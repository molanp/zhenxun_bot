"""DSH 桌面回环连接、内存鉴权与单次请求传输。"""

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
from pathlib import Path
import struct
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import httpx
import psutil
from websockets.legacy.client import connect
import yaml

from zhenxun.services.ai.core.exceptions import LLMException, RunInterrupted


class DshUnavailable(LLMException):
    """桌面连接或专用能力尚未就绪。"""


def loopback_url(value: str) -> str:
    """校验仅包含回环 IP 与显式端口的桌面 HTTP 地址。"""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if host == "localhost":
            host = "127.0.0.1"
        if (
            parsed.scheme != "http"
            or not ipaddress.ip_address(host).is_loopback
            or not parsed.port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        authority = f"[{host}]" if ":" in host else host
        return f"http://{authority}:{parsed.port}"
    except ValueError as exc:
        raise DshUnavailable("桌面地址必须是本机回环 HTTP 地址和端口。") from exc


def desktop_endpoints() -> dict[int, tuple[int, float]]:
    """从桌面进程监听套接字读取端口与进程启动身份。"""
    endpoints = {}
    for process in psutil.process_iter(["name", "exe", "create_time"]):
        try:
            if (process.info["name"] or "").lower() != "deepseek harness.exe":
                continue
            executable = Path(process.info["exe"] or "")
            if not (executable.parent / "resources" / "app.asar").is_file():
                continue
            for connection in process.net_connections(kind="tcp"):
                if (
                    connection.status == psutil.CONN_LISTEN
                    and connection.laddr
                    and ipaddress.ip_address(connection.laddr.ip).is_loopback
                ):
                    endpoints[connection.laddr.port] = (
                        process.pid,
                        process.info["create_time"],
                    )
        except (psutil.Error, OSError, ValueError):
            continue
    return endpoints


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def desktop_version(identity: tuple[int, float]) -> str | None:
    """从已确认的桌面进程安装包读取协议版本，不读取账号数据。"""
    try:
        process = psutil.Process(identity[0])
        if process.create_time() != identity[1]:
            return None
        archive = Path(process.exe()).parent / "resources" / "app.asar"
        with archive.open("rb") as stream:
            header = struct.unpack("<4I", stream.read(16))
            if header[3] > 32 * 1024 * 1024:
                return None
            index = json.loads(stream.read(header[3]))
            node = index
            for key in (
                "dsh",
                "node_modules",
                "@deepseek-ai",
                "dsh-api-session-controller",
                "package.json",
            ):
                node = node["files"][key]
            stream.seek(8 + header[1] + int(node["offset"]))
            return json.loads(stream.read(node["size"])).get("version")
    except (OSError, psutil.Error, ValueError, KeyError, TypeError):
        return None


def browser_cookie(home: Path, endpoint: str) -> str:
    """仅使用桌面浏览器签名授权记录生成短期内存 Cookie。"""
    try:
        records = yaml.safe_load((home / ".credentials.yaml").read_text("utf-8"))
        grant = records["records"]["client-connection/browser-session"]
        payload = grant["payload"]
        secret = payload["secret"]
        key = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
        if grant["kind"] != "grant" or payload["version"] != 1 or len(key) != 32:
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DshUnavailable("桌面本机授权不可用，请先启动并登录 DSH。") from exc
    authority = urlsplit(endpoint).netloc
    now = int(time.time() * 1000)
    body = _b64(
        json.dumps(
            {
                "version": 1,
                "authority": authority,
                "issuedAt": now,
                "expiresAt": now + 60 * 60 * 1000,
            },
            separators=(",", ":"),
        ).encode()
    )
    signature = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    name = "dsh-auth-" + _b64(hashlib.sha256(authority.encode()).digest())
    return f"{name}=v1.{body}.{signature}"


class DesktopClient:
    """不重连、不重放的单次桌面连接；销毁仅释放客户端资源。"""

    def __init__(self, endpoint: str, home: Path, identity: tuple[int, float]):
        self.endpoint, self.home, self.identity = endpoint, home, identity
        self.version = desktop_version(identity)
        self.cookie = browser_cookie(home, endpoint)
        self.http = httpx.AsyncClient(
            base_url=endpoint,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
            headers={"Cookie": self.cookie},
        )

    @classmethod
    async def open(cls, address: str, home: str) -> "DesktopClient":
        endpoints = await asyncio.to_thread(desktop_endpoints)
        if address.strip():
            endpoint = loopback_url(address.strip())
            port = urlsplit(endpoint).port
        else:
            if len(endpoints) != 1:
                raise DshUnavailable("无法唯一识别 DSH 桌面端口，请填写 desktop_url。")
            port = next(iter(endpoints))
            endpoint = f"http://127.0.0.1:{port}"
        if port not in endpoints:
            raise DshUnavailable("该端口不属于可确认的 DSH 桌面进程。")
        async with httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=5
        ) as probe:
            response = await probe.post(
                endpoint + "/api/settings/describe",
                json={
                    "type": "client-request",
                    "rpcId": uuid.uuid4().hex,
                    "method": "settings/describe",
                    "payload": {"args": {}},
                },
            )
            if response.status_code not in {200, 401, 403}:
                raise DshUnavailable("监听端口没有返回 DSH 桌面协议。")
            if response.status_code == 200:
                try:
                    if response.json().get("type") != "server-response":
                        raise ValueError
                except (ValueError, AttributeError) as exc:
                    raise DshUnavailable("监听端口没有返回 DSH 桌面协议。") from exc
        client = cls(
            endpoint,
            Path(home).expanduser() if home else Path.home() / ".dsh",
            endpoints[port],
        )
        try:
            await client.rpc("settings/describe", args={})
        except BaseException:
            await client.close()
            raise
        return client

    async def rpc(self, method: str, request=None, *, args=None, timeout=10) -> Any:
        """发送一次 RPC，校验请求关联与协议结果，不重试不跟随重定向。"""
        await self.verify_process()
        request_id = uuid.uuid4().hex
        if args is None:
            args = {
                "_request" if method == "session/list" else "request": request or {}
            }
        try:
            response = await self.http.post(
                "/api/" + method,
                timeout=timeout,
                json={
                    "type": "client-request",
                    "rpcId": request_id,
                    "method": method,
                    "payload": {"args": args},
                },
            )
            if response.status_code != 200:
                raise DshUnavailable(
                    f"桌面接口不可用（HTTP {response.status_code}），未重试。"
                )
            envelope = response.json()
            if (
                envelope.get("type") != "server-response"
                or envelope.get("rpcId") != request_id
            ):
                raise ValueError
            result = envelope["result"]
            if result.get("ok") is not True:
                code = str((result.get("error") or {}).get("code", "unknown"))
                raise DshUnavailable(f"桌面拒绝操作（{code}）。")
            return result.get("value")
        except (httpx.TransportError, asyncio.TimeoutError) as exc:
            raise RunInterrupted("DSH 桌面通信中断，操作结果待核验。") from exc
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            raise RunInterrupted("DSH 返回不可确认的协议结果。") from exc

    def mux(self):
        """创建无自动重连的事件连接。"""
        return connect(
            self.endpoint.replace("http://", "ws://", 1) + "/api/remote.mux",
            extra_headers={"Cookie": self.cookie},
            open_timeout=10,
            close_timeout=2,
            ping_interval=15,
            ping_timeout=10,
            max_size=16 * 1024 * 1024,
            max_queue=128,
        )

    async def verify_process(self):
        """每次发送凭据前复核监听进程及启动身份，不跨桌面重启接管任务。"""

        def check():
            try:
                process = psutil.Process(self.identity[0])
                return process.create_time() == self.identity[1] and any(
                    item.status == psutil.CONN_LISTEN
                    and item.laddr
                    and item.laddr.port == urlsplit(self.endpoint).port
                    for item in process.net_connections(kind="tcp")
                )
            except psutil.Error:
                return False

        if not await asyncio.to_thread(check):
            raise RunInterrupted("DSH 桌面进程已变化")

    async def stop(self, session_id: str, request_id: str) -> str:
        """移除本轮待执行输入并取消，直到被动状态确认空闲或窗口耗尽。"""

        async def drain():
            snapshot = await self.rpc("session/state", {"sessionId": session_id})
            for item in snapshot.get("queue") or []:
                if item.get("rpcId") == request_id:
                    await self.rpc(
                        "session/updateQueue",
                        {
                            "sessionId": session_id,
                            "itemId": item["id"],
                            "action": {"kind": "remove"},
                        },
                    )
            await self.rpc("session/cancel", {"sessionId": session_id})
            while True:
                state = await self.rpc("session/state", {"sessionId": session_id})
                if (
                    state.get("running") is False
                    and not state.get("queue")
                    and not state.get("pendingInput")
                ):
                    return "confirmed"
                await asyncio.sleep(0.2)

        try:
            return await asyncio.wait_for(drain(), 9)
        except (Exception, asyncio.CancelledError, RunInterrupted):
            return "unknown"

    async def close(self):
        await self.http.aclose()
        self.cookie = ""

    async def workspace_snapshot(self) -> dict:
        """读取工作区事件流的首个快照，不激活会话。"""
        stream_id = uuid.uuid4().hex
        async with self.mux() as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "open",
                        "streamId": stream_id,
                        "endpoint": "workspace/follow",
                        "payload": {"args": {}},
                    }
                )
            )
            while True:
                frame = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if frame.get("streamId") != stream_id:
                    continue
                if frame.get("type") in {"error", "end"}:
                    raise DshUnavailable("工作区快照暂不可读取。")
                value = frame.get("value") or {}
                if value.get("type") == "baseline":
                    return value["value"]
