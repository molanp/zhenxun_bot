"""Bot 到独立 SDK 宿主的本机连接、单轮转发与结果确认。"""

import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

import psutil

from zhenxun.services.ai.core.exceptions import RunInterrupted
from zhenxun.services.ai.run.dsh_desktop import DshUnavailable


def eligible_providers():
    """返回可供独立 SDK 使用的官方服务商，凭据仅在后端消费。"""
    from zhenxun.services.ai.llm.manager import get_configured_providers

    providers = []
    for provider in get_configured_providers():
        try:
            address = urlsplit(provider.base_url)
        except ValueError:
            continue
        if (
            provider.api_keys
            and address.scheme == "https"
            and address.hostname == "api.deepseek.com"
            and any(model.enabled for model in provider.models)
        ):
            providers.append(provider)
    return providers


class SdkClient:
    transport = "sdk"
    _starting = asyncio.Lock()

    def __init__(self, home: Path, endpoint: dict, mode: str):
        self.home, self.endpoint, self.mode = home, endpoint, mode
        self.run_id = None
        self.message_id = None
        self.context = None
        self.closed = False

    @staticmethod
    def home_for(settings) -> Path:
        from .dsh import DSH_PATH

        return Path(settings.sdk_home or DSH_PATH / "sdk-host").expanduser().resolve()

    @classmethod
    async def open(cls, settings, mode="agent"):
        home = cls.home_for(settings)
        try:
            endpoint = json.loads(
                (home / "zhenxun-control" / "host.json").read_text(encoding="utf8")
            )
            if (
                endpoint.get("contract") != 1
                or not 0 < int(endpoint["port"]) < 65536
                or not isinstance(endpoint.get("token"), str)
            ):
                raise ValueError
            process = psutil.Process(endpoint["pid"])
            argv = process.cmdline()
            if not any(Path(arg).name == "dsh_sdk_host.py" for arg in argv):
                raise ValueError
            index = argv.index("--home")
            if Path(argv[index + 1]).resolve() != home:
                raise ValueError
            client = cls(home, endpoint, mode)
            await client.call("status")
            return client
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            psutil.Error,
            asyncio.TimeoutError,
            RunInterrupted,
        ) as exc:
            raise DshUnavailable(
                "SDK 宿主未连接；请在 DSH 页显式启动或检查配置。"
            ) from exc

    async def call(self, method, params=None, timeout=40):
        writer = None
        try:

            async def request():
                nonlocal writer
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", int(self.endpoint["port"]), limit=8 * 1024 * 1024
                )
                writer.write(
                    (
                        json.dumps(
                            {
                                "token": self.endpoint["token"],
                                "method": method,
                                "params": {"mode": self.mode, **(params or {})},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode()
                )
                await writer.drain()
                result = json.loads(await reader.readline())
                if result.get("boot") != self.endpoint["boot"]:
                    raise DshUnavailable("SDK 宿主代际已变化；不会恢复原运行。")
                if result.get("error"):
                    raise DshUnavailable(result["error"])
                return result["result"]

            return await asyncio.wait_for(request(), timeout)
        except (OSError, asyncio.TimeoutError, ValueError, KeyError) as exc:
            raise RunInterrupted("SDK 宿主通信") from exc
        finally:
            if writer:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()

    @classmethod
    def credentials(cls, settings) -> dict:
        """只从已配置的官方 DeepSeek 提供商取密钥，不读取桌面账号。"""
        eligible = [
            p
            for p in eligible_providers()
            if not settings.sdk_provider or p.name == settings.sdk_provider
        ]
        if len(eligible) != 1:
            raise DshUnavailable(
                "请在 sdk_provider 中指定一个已配置的官方 DeepSeek 提供商。"
            )
        provider = eligible[0]
        models = [m.name for m in provider.models if m.enabled]
        model = settings.sdk_model or (models[0] if models else "")
        if not model or model not in models:
            raise DshUnavailable("SDK 模型必须来自该提供商已启用的模型列表。")
        executable = shutil.which(settings.sdk_bin) or settings.sdk_bin
        if not executable or not Path(executable).expanduser().is_file():
            raise DshUnavailable(
                "未找到 SDK 运行文件，请配置 sdk_bin；不会自动安装或退回桌面。"
            )
        return {
            "bin": str(Path(executable).expanduser().resolve()),
            "api_key": provider.api_keys[0],
            "model": model,
            "thinking": settings.sdk_thinking,
        }

    @classmethod
    async def ensure(cls, settings, mode):
        async with cls._starting:
            parameters = cls.credentials(settings)
            try:
                client = await cls.open(settings, mode)
            except DshUnavailable:
                home = cls.home_for(settings)
                home.mkdir(parents=True, exist_ok=True)
                options = (
                    {"start_new_session": True}
                    if os.name != "nt"
                    else {
                        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                    }
                )
                await asyncio.to_thread(
                    subprocess.Popen,
                    [
                        sys.executable,
                        str(Path(__file__).with_name("dsh_sdk_host.py")),
                        "--home",
                        str(home),
                    ],
                    cwd=home,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **options,
                )
                deadline = time.monotonic() + 10
                while True:
                    try:
                        client = await cls.open(settings, mode)
                        break
                    except DshUnavailable:
                        if time.monotonic() >= deadline:
                            raise
                        await asyncio.sleep(0.15)
            if cls.needs_restart(await client.call("status")):
                raise DshUnavailable(
                    "SDK 宿主连接契约不匹配或无法确认。请确认后台任务后，"
                    "在更多操作中停止 SDK 宿主，再重新连接；不会自动停止任务。"
                )
            await client.call("start", parameters, timeout=65)
            return client

    @staticmethod
    def needs_restart(status):
        """核对当前宿主及已启动 Worker 的连接契约，不自动重启。"""
        from .dsh_sdk_host import SDK_API_BASE, SDK_CONNECTION_REVISION

        return status.get("connection_revision") != SDK_CONNECTION_REVISION or any(
            worker.get("api_base") != SDK_API_BASE
            for worker in status.get("workers", {}).values()
            if worker.get("ready")
        )

    @classmethod
    async def status(cls, settings):
        result = {
            "transport": "sdk",
            "connected": False,
            "version": None,
            "native_ready": False,
            "standard_ready": False,
            "presets": {},
            "mapping_trusted": True,
            "login_state": "api-key",
            "capabilities": {
                "memory": False,
                "knowledge": False,
                "native_agent": True,
                "images": False,
            },
            "retry_policy": {"max_attempts": 5, "delays": [2, 5, 10, 20]},
            "restart_required": False,
        }
        try:
            client = await cls.open(settings)
            data = await client.call("status")
            workers = data["workers"]
            restart_required = cls.needs_restart(data)
            result.update(
                connected=True,
                host_boot=data["boot"],
                native_ready=workers.get("agent", {}).get("ready", False),
                standard_ready=workers.get("agent", {}).get("ready", False),
                presets={"chat": workers.get("chat", {}).get("ready", False)},
                version="SDK 协议 "
                + str(
                    workers.get("agent", workers.get("chat", {})).get("version")
                    or "待启动"
                ),
                reason="独立宿主保活；Bot 退出只结束原生 Agent 转发，不自动补发。",
                restart_required=restart_required,
                restart_reason=(
                    "SDK 宿主连接契约不匹配或无法确认。请确认没有需要保留的后台任务，"
                    "再通过更多操作停止 SDK 宿主并重新连接；不会自动重启。"
                    if restart_required
                    else ""
                ),
            )
        except (DshUnavailable, RunInterrupted) as exc:
            result["reason"] = str(exc)
        return result

    async def rpc(self, method, args=None):
        args = args or {}
        if method == "session/list":
            return await self.call("control", {"method": "list"})
        if method == "session/state":
            return await self.call("control", {"method": "state", "args": args})
        raise DshUnavailable("当前 SDK 不提供此桌面接口，未当作普通 prompt 执行。")

    @classmethod
    async def start_turn(cls, context, binding, settings, approve, ask):
        from .dsh import BOOT_ID, DshRuntime, DshSessions, _Turn

        records = DshSessions.for_binding(binding)
        home = str(cls.home_for(settings))
        if binding.get("sdk_home") not in {None, home}:
            raise DshUnavailable("SDK home 已变化，请显式选择此后端的会话。")
        if not binding.get("native") and (
            binding.get("inflight") or binding.get("quarantined")
        ):
            records.rotate_remote(binding)
        client = await cls.ensure(settings, binding["mode"])
        client.context = context
        client.run_id = context.run.run_id
        turn = _Turn(context, binding, client, approve, ask)
        turn.proof = {"transport": "sdk", "contract": 1}
        state = await client.rpc("session/state", {"sessionId": binding["remote_id"]})
        if binding.get("inflight") and not state.get("loaded"):
            raise DshUnavailable(
                "原 SDK 执行结果待核验；可显式新建会话，不会重放旧输入。"
            )
        if state.get("running") or state.get("queue"):
            raise DshUnavailable("此 SDK 会话仍在运行，请等待或显式停止。")
        records.update(
            binding,
            sdk_home=home,
            inflight={
                "boot": BOOT_ID,
                "run_id": context.run.run_id,
                "request_id": turn.request_id,
            },
        )
        DshRuntime._turns[context.run.run_id] = turn
        return turn

    async def prompt(self, turn, content, timeout):
        from .dsh import DshRuntime, DshSessions

        if any(block.get("type") != "text" for block in content):
            raise DshUnavailable("SDK 图片契约尚未通过本接入验证，请改发文字。")
        receipt = await self.call(
            "prompt",
            {
                "run_id": self.run_id,
                "session_id": turn.binding["remote_id"],
                "inherited": turn.binding["created"],
                "content": content,
                "model": turn.binding.get("model")
                if not turn.binding.get("created") and not turn.binding.get("inherited")
                else None,
            },
        )
        self.message_id = receipt["message_id"]
        if not self.message_id:
            raise RunInterrupted("SDK 输入收据未知，未重发")
        DshSessions.for_binding(turn.binding).update(turn.binding, created=True)
        cursor = receipt["cursor"]
        deadline = time.monotonic() + timeout
        answered = set()
        while time.monotonic() < deadline:
            batch = await self.call(
                "poll",
                {"run_id": self.run_id, "cursor": cursor},
                timeout=min(20, max(0.1, deadline - time.monotonic())),
            )
            cursor = batch["cursor"]
            for question in batch["questions"]:
                if question["id"] in answered:
                    continue
                answered.add(question["id"])
                task = asyncio.create_task(self.answer(turn, question))
                turn.pending.add(task)
            for task in list(turn.pending):
                if task.done():
                    turn.pending.remove(task)
                    task.result()
            for envelope in batch["events"]:
                params = envelope["params"]
                if (
                    envelope["method"] != "session.event"
                    or params.get("sessionId") != turn.binding["remote_id"]
                ):
                    continue
                event = params.get("event") or {}
                data = event.get("data") or {}
                if (
                    event.get("type") == "user/message"
                    and (data.get("message") or data).get("id") == self.message_id
                ):
                    turn.admitted = True
                outcome = DshRuntime.consume(turn, event)
                if outcome is not None:
                    if outcome != "completed":
                        raise RunInterrupted("SDK 本轮未完成")
                    return (
                        "\n\n".join(t for t in turn.responses.values() if t),
                        turn.usage,
                        outcome,
                    )
        raise RunInterrupted("SDK 本轮超时")

    async def answer(self, turn, question):
        from zhenxun.services.ai.utils.runtime import PermissionUtils

        PermissionUtils.ensure_current(turn.context)
        params = question["params"]
        if question["method"] == "approval/request":
            allowed = await turn.approve(
                str(params.get("toolName", "")), str(params.get("reason", ""))
            )
            result = {"outcome": "allowed-once" if allowed else "rejected"}
        else:
            answers = []
            for item in params.get("questions", []):
                text = await turn.ask(str(item.get("question", "")))
                answers.append({"id": item["id"], "selected": [], "custom": text})
            result = {"answers": answers}
        PermissionUtils.ensure_current(turn.context)
        await self.call(
            "answer", {"run_id": self.run_id, "id": question["id"], "answer": result}
        )

    async def stop(self, session_id, request_id):
        del request_id
        state = await self.call(
            "control",
            {
                "method": "cancel",
                "args": {"sessionId": session_id, "messageId": self.message_id},
            },
        )
        if not state.get("confirmed"):
            return "unknown"
        return (
            "target_removed"
            if state.get("reason") == "another_activity"
            else "confirmed"
        )

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.run_id:
            with contextlib.suppress(Exception):
                await self.call(
                    "detach",
                    {
                        "run_id": self.run_id,
                        "outcome": self.context.run.outcome or "unknown",
                    },
                    timeout=3,
                )
