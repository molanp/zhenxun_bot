"""独立 SDK stdio 宿主；不导入 Bot，不随转发客户端退出。"""

import argparse
import asyncio
from collections import deque
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

import httpx
import psutil
import yaml

LIMIT = 8 * 1024 * 1024
SDK_API_BASE = "https://api.deepseek.com/anthropic"
SDK_CONNECTION_REVISION = 2
SHELL_TIMEOUT = {"timeoutMs": 300_000, "maxTimeoutMs": 1_800_000}


def private_directory(path: Path) -> None:
    """仅保护本集成的凭据目录，不调整用户工作目录或系统权限。"""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        user = psutil.Process().username()
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"{user}:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
                "/remove:g",
                "*S-1-1-0",
                "*S-1-5-11",
                "*S-1-5-32-545",
            ],
            check=True,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        path.chmod(0o700)


def atomic(path: Path, value: dict) -> None:
    """原子保存无模型密钥的宿主状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex)
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class Worker:
    def __init__(self, home: Path, mode: str, permission: str = "sandbox"):
        self.home, self.mode = home, mode
        self.permission = permission if mode == "agent" else "sandbox"
        self.process = None
        self.tasks = set()
        self.pending = {}
        self.questions = {}
        self.events = deque(maxlen=4000)
        self.sequence = 0
        self.changed = asyncio.Event()
        self.lock = asyncio.Lock()
        receipt_path = home / "forwarding-receipts.json"
        previous = (
            json.loads(receipt_path.read_text(encoding="utf8"))
            if receipt_path.exists()
            else {"runs": []}
        )
        if not isinstance(previous, dict) or not isinstance(previous.get("runs"), list):
            raise ValueError("SDK 转发收据损坏，不能当作空记录。")
        self.runs = {
            row["run_id"]: {**row, "attached": False} for row in previous["runs"]
        }
        self.children = {}
        self.loaded = set()
        self.control_token = secrets.token_urlsafe(32)
        self.control_port = None
        self.control_identity = None
        self.version = None
        self.last_error = None
        self.runtime_identity = None
        self.transport_closed = False
        self.credential_hash = None
        self.defaults = None
        self.api_base = None

    async def control(self, method, args=None):
        if self.process is None or self.process.returncode is not None:
            raise ValueError("SDK 进程已退出；不会重放原输入。")
        if self.control_identity:
            try:
                process = psutil.Process(self.control_identity[0])
                if process.create_time() != self.control_identity[1]:
                    raise ValueError("SDK 控制进程已变化。")
            except psutil.Error as error:
                raise ValueError("SDK 控制进程已退出。") from error
        async with httpx.AsyncClient(
            timeout=10, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.post(
                f"http://127.0.0.1:{self.control_port}/control",
                headers={"Authorization": "Bearer " + self.control_token},
                json={"method": method, "args": args or {}},
            )
            if response.status_code != 200:
                self.last_error = {"control": response.json()}
                raise ValueError("SDK 控制操作未完成，请检查会话状态。")
            return response.json()["result"]

    async def launch(self, params):
        """启动独立工具组合，握手和控制契约通过后才允许提交输入。"""
        executable = Path(params["bin"]).resolve(strict=True)
        if self.process and self.process.returncode is None:
            if self.transport_closed:
                raise ValueError("SDK 通道已关闭，请核验后显式停止宿主。")
            if self.runtime_identity != str(executable):
                raise ValueError("运行文件已变化，请显式停止宿主后重新启动。")
            if (
                self.credential_hash
                != hashlib.sha256(params["api_key"].encode()).hexdigest()
            ):
                raise ValueError(
                    "SDK 凭据已变化，请显式停止宿主后重新启动；原任务未被修改。"
                )
            if self.defaults != (params["model"], params.get("thinking") or ""):
                await self.initialize(params)
            return
        if self.process is not None:
            raise ValueError("SDK 已退出，请先核对未决记录再显式重启宿主。")
        self.home.mkdir(parents=True, exist_ok=True)
        root = self.home / "zhenxun-control"
        root.mkdir(exist_ok=True)
        (root / "control.json").unlink(missing_ok=True)
        patch = [
            {
                "insert": [
                    {
                        "id": "zhenxun-sdk-control",
                        "name": str(Path(__file__).with_name("dsh_sdk_guard.mjs")),
                        "config": {"root": str(root), "mode": self.mode},
                    }
                ]
            },
            {"id": "bash-sandbox", "config": SHELL_TIMEOUT},
            {"id": "pwsh-sandbox", "config": SHELL_TIMEOUT},
        ]
        patch_path = root / "profile.patch.yml"
        patch_path.write_text(
            yaml.safe_dump(patch, allow_unicode=True), encoding="utf8"
        )
        env = {
            **os.environ,
            "DSH_HOME": str(self.home),
            "DEEPSEEK_API_KEY": params["api_key"],
            "DEEPSEEK_BASE_URL": SDK_API_BASE,
            "ZHENXUN_SDK_CONTROL_TOKEN": self.control_token,
            "DSH_PERMISSION_MODE": "danger-full-access"
            if self.permission == "full"
            else "workspace-write",
        }
        command = [str(executable), "--profile", "sdk", "--patch", str(patch_path)]
        if executable.suffix == ".py":
            command.insert(0, sys.executable)
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.home,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=LIMIT,
        )
        self.runtime_identity = str(executable)
        self.api_base = SDK_API_BASE
        self.credential_hash = hashlib.sha256(params["api_key"].encode()).hexdigest()
        self.tasks = {
            asyncio.create_task(self.read()),
            asyncio.create_task(self.drain_stderr()),
        }
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise ValueError("SDK 启动失败。")
                try:
                    endpoint = json.loads(
                        (root / "control.json").read_text(encoding="utf8")
                    )
                    owned = {
                        self.process.pid,
                        *(
                            p.pid
                            for p in psutil.Process(self.process.pid).children(
                                recursive=True
                            )
                        ),
                    }
                    if endpoint["pid"] in owned and endpoint["contract"] == 1:
                        self.control_port = int(endpoint["port"])
                        self.control_identity = (
                            endpoint["pid"],
                            psutil.Process(endpoint["pid"]).create_time(),
                        )
                        break
                except (OSError, ValueError, KeyError, psutil.Error):
                    pass
                await asyncio.sleep(0.1)
            if self.control_port is None:
                raise ValueError("当前 SDK 未提供所需控制契约，未发送模型请求。")
            capabilities = await self.control("capabilities")
            if (
                capabilities.get("contract") != 1
                or not capabilities.get("preserveSelection")
                or (self.mode == "chat" and not capabilities.get("toolsRestricted"))
            ):
                raise ValueError("SDK 会话或权限能力不满足要求。")
            await self.initialize(params)
        except BaseException:
            await self.close()
            raise

    async def initialize(self, params):
        """更新新会话的默认路由，已有会话保留各自的持久化选择。"""
        info = await self.rpc(
            "initialize",
            {
                "cwd": str(self.home / "workspace"),
                "provider": "deepseek-official",
                "model": params["model"],
                **(
                    {"reasoningEffort": params["thinking"]}
                    if params.get("thinking")
                    else {}
                ),
                "approvalRequests": True,
                "capabilities": {"userQuestions": True},
            },
        )
        if info.get("serverInfo", {}).get("name") != "deepseek-harness-sdk-runtime":
            raise ValueError("SDK 握手身份不匹配。")
        self.version = info["serverInfo"]["version"]
        self.defaults = (params["model"], params.get("thinking") or "")

    async def drain_stderr(self):
        # 运行时可能输出用户提示或凭据，宿主不持久化原始诊断。
        while await self.process.stderr.read(8192):
            pass

    async def send(self, value):
        if (
            not self.process
            or self.process.returncode is not None
            or self.transport_closed
        ):
            raise ValueError("SDK 已断开。")
        self.process.stdin.write(
            (json.dumps(value, ensure_ascii=False) + "\n").encode()
        )
        await self.process.stdin.drain()

    async def rpc(self, method, params):
        ident = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[ident] = future
        try:
            await self.send(
                {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
            )
            return await asyncio.wait_for(future, 30)
        finally:
            self.pending.pop(ident, None)

    async def read(self):
        try:
            while line := await self.process.stdout.readline():
                value = json.loads(line)
                if not value.get("method"):
                    future = self.pending.get(value.get("id"))
                    if future and not future.done():
                        if "error" in value:
                            self.last_error = value["error"]
                            future.set_exception(
                                ValueError("SDK 拒绝请求；未自动重复提交。")
                            )
                        else:
                            future.set_result(value.get("result") or {})
                    continue
                method, params = value["method"], value.get("params") or {}
                if "id" in value:
                    task = asyncio.create_task(self.question(value))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                    continue
                if method == "subagent.started":
                    self.children[params.get("childSessionId")] = params.get(
                        "parentSessionId"
                    )
                if method == "user-question.cancelled":
                    for request in self.questions.values():
                        if request["params"].get("requestId") == params.get(
                            "requestId"
                        ):
                            request["answer"].cancel()
                self.sequence += 1
                self.events.append(
                    {"sequence": self.sequence, "method": method, "params": params}
                )
                self.changed.set()
        except (OSError, ValueError, asyncio.LimitOverrunError):
            pass
        finally:
            self.transport_closed = True
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ValueError("SDK 通信中断，执行结果待核验。"))
            self.changed.set()

    async def question(self, value):
        ident = str(value["id"])
        params, method = value.get("params") or {}, value["method"]
        session = params.get("sessionId")
        visited = set()
        while session in self.children and session not in visited:
            visited.add(session)
            session = self.children[session]
        run = next(
            (
                r
                for r in self.runs.values()
                if r["session_id"] == session
                and r.get("attached")
                and r.get("expires", 0) > time.time()
            ),
            None,
        )
        answer = asyncio.get_running_loop().create_future()
        self.questions[ident] = {
            "params": params,
            "method": method,
            "answer": answer,
            "run_id": run["run_id"] if run else None,
        }
        self.changed.set()
        try:
            if not run or method not in {"approval/request", "user-question/ask"}:
                raise ValueError("No active recipient")
            response = await asyncio.wait_for(answer, 1800)
            await self.send({"jsonrpc": "2.0", "id": value["id"], "result": response})
        except (ValueError, asyncio.TimeoutError, asyncio.CancelledError):
            with contextlib.suppress(Exception):
                await self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": value["id"],
                        **(
                            {"result": {"outcome": "rejected"}}
                            if method == "approval/request"
                            else {
                                "error": {
                                    "code": -32000,
                                    "message": "Question recipient unavailable",
                                }
                            }
                        ),
                    }
                )
        finally:
            self.questions.pop(ident, None)

    async def prompt(self, params):
        async with self.lock:
            run_id = params["run_id"]
            signature = hashlib.sha256(
                json.dumps(
                    {
                        key: params.get(key)
                        for key in ("session_id", "content", "model")
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if run_id in self.runs:
                if self.runs[run_id].get("signature") not in {None, signature}:
                    raise ValueError("同一运行标识不能提交不同输入。")
                return self.runs[run_id]
            session_id = params["session_id"]
            state = await self.control("state", {"sessionId": session_id})
            if state.get("running") or state.get("queue"):
                raise ValueError("SDK 会话仍有任务；不会追加或重放输入。")
            if params.get("inherited") and session_id not in self.loaded:
                await self.rpc("session/resume", {"sessionId": session_id})
                self.loaded.add(session_id)
            if params.get("model"):
                await self.control(
                    "prepare", {"sessionId": session_id, "model": params["model"]}
                )
            if not params.get("inherited") and session_id not in self.loaded:
                workspace = self.home / "workspace" / session_id
                workspace.mkdir(parents=True, exist_ok=True)
                await self.rpc(
                    "session/working-directory/set",
                    {"sessionId": session_id, "path": str(workspace)},
                )
                self.loaded.add(session_id)
            run = {
                "signature": signature,
                "run_id": run_id,
                "session_id": session_id,
                "cursor": self.sequence,
                "attached": True,
                "expires": time.time() + 15,
                "message_id": None,
                "outcome": "unknown",
            }
            self.runs[run_id] = run
            self.save_receipts()
            receipt = await self.rpc(
                "session/prompt",
                {"sessionId": session_id, "contentBlocks": params["content"]},
            )
            if not receipt.get("messageId"):
                raise ValueError("SDK 输入收据缺失，禁止自动重发。")
            run["message_id"] = receipt["messageId"]
            self.save_receipts()
            return dict(run)

    def save_receipts(self):
        atomic(
            self.home / "forwarding-receipts.json",
            {
                "runs": [
                    {k: v for k, v in row.items() if k not in {"expires", "attached"}}
                    for row in self.runs.values()
                ]
            },
        )

    async def poll(self, params):
        run = self.runs.get(params["run_id"])
        if not run or not run.get("attached"):
            raise ValueError("原转发已结束，不会重新投递旧消息。")
        cursor = int(params["cursor"])
        run["expires"] = time.time() + 30
        self.changed.clear()
        if self.sequence <= cursor:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.changed.wait(), 15)
        if self.events and cursor < self.events[0]["sequence"] - 1:
            raise ValueError("事件缓冲已淘汰，送达结果待核验；不会补发。")
        if self.process.returncode is not None or self.transport_closed:
            raise ValueError("SDK 进程已退出，结果待核验。")
        events = [e for e in self.events if e["sequence"] > cursor][:200]
        questions = [
            {"id": ident, "method": q["method"], "params": q["params"]}
            for ident, q in self.questions.items()
            if q["run_id"] == run["run_id"] and not q["answer"].done()
        ]
        return {
            "events": events,
            "cursor": events[-1]["sequence"] if events else self.sequence,
            "questions": questions,
        }

    async def close(self):
        if self.process and self.process.returncode is None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.rpc("shutdown", {}), 7)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.process.wait(), 2)
            if self.process.returncode is None:
                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.Error:
                    children = []
                for child in reversed(children):
                    with contextlib.suppress(psutil.Error):
                        child.terminate()
                self.process.terminate()
                await asyncio.to_thread(psutil.wait_procs, children, timeout=2)
            await self.process.wait()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class Host:
    def __init__(self, home: Path):
        self.home = home
        self.token = secrets.token_urlsafe(32)
        self.boot = uuid.uuid4().hex
        self.workers = {}
        self.closed = asyncio.Event()

    async def dispatch(self, method, params):
        if method == "status":
            return {
                "boot": self.boot,
                "pid": os.getpid(),
                "contract": 1,
                "connection_revision": SDK_CONNECTION_REVISION,
                "workers": {
                    mode: {
                        "ready": not w.transport_closed
                        and w.process is not None
                        and w.process.returncode is None
                        and bool(w.version),
                        "version": w.version,
                        "api_base": w.api_base,
                        "permission": w.permission,
                    }
                    for mode, w in self.workers.items()
                },
            }
        if method == "shutdown":
            for worker in self.workers.values():
                if worker.process is None or worker.process.returncode is not None:
                    continue
                for row in (await worker.control("list"))["items"]:
                    if row.get("running"):
                        raise ValueError(
                            "SDK 仍有活动任务，请在确认后台工作后显式处理。"
                        )
            self.closed.set()
            return {"stopping": True}
        mode = params.get("mode")
        if mode not in {"chat", "agent"}:
            raise ValueError("无效的 SDK 模式。")
        if method == "start":
            permission = params.get("permission") or "sandbox"
            worker = self.workers.get(mode)
            if worker and mode == "agent" and worker.permission != permission:
                async with worker.lock:
                    if worker.process and worker.process.returncode is None:
                        if any(r.get("attached") for r in worker.runs.values()) or any(
                            row.get("running")
                            for row in (await worker.control("list"))["items"]
                        ):
                            raise ValueError(
                                "Agent 权限已修改，但 SDK 仍有任务在运行；"
                                "请等待任务结束后再试。"
                            )
                        await worker.close()
                worker = None
            if worker is None:
                worker = self.workers[mode] = Worker(self.home / mode, mode, permission)
            async with worker.lock:
                await worker.launch(params)
            return await self.dispatch("status", {})
        worker = self.workers.get(mode)
        if worker is None:
            raise ValueError("此 SDK 工具组合尚未启动。")
        if method == "prompt":
            return await worker.prompt(params)
        if method == "poll":
            return await worker.poll(params)
        if method in {"run_status", "stop_run"}:
            run = worker.runs.get(params.get("run_id"))
            if (
                not run
                or run["session_id"] != params.get("session_id")
                or not run.get("message_id")
            ):
                return {"state": "unknown", "reason": "原运行的可信收据不可用"}
            args = {"sessionId": run["session_id"], "messageId": run["message_id"]}
            state = await worker.control("inspect", args)
            if method == "run_status":
                return state
            if not state.get("known"):
                return {"state": "unknown", "reason": "无法在当前执行端确认原输入"}
            stopped = await worker.control("cancel", args)
            outcome = (
                (
                    "target_removed"
                    if stopped.get("reason") == "another_activity"
                    else "foreground_confirmed"
                )
                if stopped.get("confirmed")
                else "unknown"
            )
            return {
                "state": outcome,
                "foreground_only": True,
                "reason": stopped.get("reason"),
            }
        if method == "answer":
            question = worker.questions.get(params["id"])
            run = worker.runs.get(params["run_id"])
            if (
                not question
                or question["run_id"] != params["run_id"]
                or question["answer"].done()
                or not run
                or not run["attached"]
            ):
                raise ValueError("问题已失效。")
            question["answer"].set_result(params["answer"])
            return {"accepted": True}
        if method == "detach":
            if run := worker.runs.get(params["run_id"]):
                run["attached"] = False
                run["outcome"] = params.get("outcome", "detached")
                worker.save_receipts()
                for question in worker.questions.values():
                    if question["run_id"] == params["run_id"]:
                        question["answer"].cancel()
            return {"detached": True}
        if method == "control":
            return await worker.control(params["method"], params.get("args"))
        raise ValueError("不支持的宿主请求。")

    async def connection(self, reader, writer):
        try:
            value = json.loads(await asyncio.wait_for(reader.readline(), 10))
            if not hmac.compare_digest(str(value.get("token", "")), self.token):
                raise ValueError("宿主认证失败。")
            result = await self.dispatch(value["method"], value.get("params") or {})
            response = {"result": result, "boot": self.boot}
        except Exception as error:
            # 上游请求原文和安装诊断不发给聊天客户端。
            message = (
                str(error)
                if isinstance(error, ValueError)
                else "SDK 宿主请求未完成，结果待核验。"
            )
            response = {"error": message[:250], "boot": self.boot}
        try:
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve(self):
        self.home.mkdir(parents=True, exist_ok=True)
        secure = self.home / "zhenxun-control"
        await asyncio.to_thread(private_directory, secure)
        lock = (self.home / "host.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            server = await asyncio.start_server(
                self.connection, "127.0.0.1", 0, limit=LIMIT
            )
            atomic(
                secure / "host.json",
                {
                    "port": server.sockets[0].getsockname()[1],
                    "pid": os.getpid(),
                    "token": self.token,
                    "boot": self.boot,
                    "contract": 1,
                },
            )
            async with server:
                await self.closed.wait()
        finally:
            await asyncio.gather(
                *(worker.close() for worker in self.workers.values()),
                return_exceptions=True,
            )
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    args = parser.parse_args()
    asyncio.run(Host(Path(args.home).resolve()).serve())
