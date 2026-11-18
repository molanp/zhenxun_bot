"""真寻专用 SDK 的独立安装、协议校验及安装任务状态。"""

import asyncio
import contextlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time

from filelock import FileLock, Timeout
import psutil

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.installer_network import (
    installer_environment,
    resolver_failure,
    uv_binary_options,
    uv_command,
    uv_executable,
)
from zhenxun.services.log import logger
from zhenxun.utils.atomic_json import read_json_locked, write_json_locked

SDK_VERSION = "0.2.2rc2"
ACTIVE_PHASES = {"checking", "creating", "installing", "verifying", "stopping"}


class SdkInstallError(RuntimeError):
    """安装无法安全继续，保留旧安装及明确诊断。"""


class SdkInstaller:
    """管理一个固定版本的独立环境，不写入 Bot 或桌面依赖。"""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.task = None
        self.process = None
        self.process_identity = None
        self.lock = None
        self.record = {}
        self.closing = False
        self.driver = None

    def bind_lifecycle(self, driver):
        """每个驱动只注册一次安装器的启动与关闭回调。"""
        if self.driver is driver:
            return
        driver.on_startup(self.startup)
        driver.on_shutdown(self.close)
        self.driver = driver

    async def startup(self):
        """开启新生命周期的安装准入，不恢复中断任务。"""
        if (not self.task or self.task.done()) and not self._alive(
            self.process_identity
        ):
            self.closing = False

    @property
    def record_path(self):
        return self.root / "installation.json"

    def _read(self):
        if not self.root.exists():
            return {}
        value = read_json_locked(self.record_path, {}, strict=True)
        if not isinstance(value, dict) or (
            value
            and (
                value.get("schema") != 1
                or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("id", "")))
            )
        ):
            raise SdkInstallError("安装记录无法读取，请检查目录权限或恢复记录。")
        current = value.get("current")
        if current is not None and (
            not isinstance(current, dict)
            or not isinstance(current.get("path"), str)
            or not isinstance(current.get("version"), str)
        ):
            raise SdkInstallError("已安装 SDK 的记录不完整。")
        child = value.get("child")
        if child is not None and (
            not isinstance(child, list)
            or len(child) != 2
            or type(child[0]) is not int
            or not isinstance(child[1], int | float)
        ):
            raise SdkInstallError("安装进程记录无法确认。")
        if value and value.get("phase") not in ACTIVE_PHASES | {
            "ready",
            "failed",
            "cancelled",
        }:
            raise SdkInstallError("安装阶段记录无法确认。")
        return value

    def _save(self, **changes):
        self.record.update(changes, updated_at=time.time())
        write_json_locked(self.record_path, self.record)

    @staticmethod
    def _alive(identity):
        if not identity:
            return False
        try:
            return psutil.Process(identity[0]).create_time() == identity[1]
        except psutil.NoSuchProcess:
            return False
        except (psutil.Error, TypeError, IndexError):
            return True

    def _owned_path(self, value):
        base = self.root / "environments"
        if base.resolve() != base or base.is_symlink():
            raise SdkInstallError("安装环境目录不能链接到其他位置。")
        path = Path(value).resolve()
        if not path.is_relative_to(base) or path == base:
            raise SdkInstallError("安装记录的程序位置不属于真寻管理目录。")
        return path

    def status(self):
        """只读取安装证据，不执行用户填写的程序或自动恢复安装。"""
        try:
            record = self._read()
            current = record.get("current") or {}
            path = self._owned_path(current["path"]) if current else None
            installed = bool(path and path.is_file())
            active = bool(self.task and not self.task.done())
            if not active and self.root.exists():
                probe = FileLock(self.root / ".install.lock", thread_local=False)
                try:
                    probe.acquire(timeout=0)
                except Timeout:
                    active = True
                else:
                    probe.release()
            phase = record.get("phase", "missing")
            interrupted = not active and phase in ACTIVE_PHASES
            child_alive = self._alive(record.get("child")) and not active
            if interrupted:
                phase = "interrupted"
            missing = phase == "ready" and not installed
            if missing:
                phase = "failed"
            return {
                "phase": phase,
                "active": active,
                "operation_id": record.get("id"),
                "installed": installed,
                "path": str(path) if installed else "",
                "version": current.get("version") if installed else None,
                "target_version": SDK_VERSION,
                "can_install": not self.closing and not active and not child_alive,
                "message": (
                    "上次安装已中断，安装进程仍在退出，暂不能重试。"
                    if child_alive
                    else "上次安装已中断，可重新安装；不会自动继续。"
                    if interrupted
                    else "SDK 程序已缺失，请修复安装后重新选择程序位置。"
                    if missing
                    else record.get(
                        "message", "可安装托管 SDK，也可使用 PATH 或自定义程序连接。"
                    )
                ),
                "detail": record.get("detail", ""),
                "cleanup_pending": record.get("cleanup_pending", False),
                "cleanup_detail": record.get("cleanup_detail", ""),
                "retained_environments": record.get("retained_environments", 0),
            }
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {
                "phase": "unavailable",
                "active": False,
                "installed": False,
                "can_install": False,
                "target_version": SDK_VERSION,
                "message": "安装状态无法确认，请检查安装记录和目录权限。",
                "detail": type(error).__name__,
            }

    def start(self, operation_id: str, expected_operation: str | None):
        """同一操作查询原任务，新的显式安装使用独立候选目录。"""
        if self.closing:
            raise SdkInstallError("Bot 正在关闭，不能开始安装。")
        if not re.fullmatch(r"[0-9a-f]{32}", operation_id):
            raise SdkInstallError("安装操作标识无效。")
        status = self.status()
        if status.get("operation_id") == operation_id or status["active"]:
            return status
        if not status["can_install"]:
            raise SdkInstallError(status["message"])
        self.root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.root / ".install.lock", thread_local=False)
        try:
            lock.acquire(timeout=0)
        except Timeout as error:
            raise SdkInstallError("已有安装正在进行，请刷新状态。") from error
        try:
            old = self._read()
            if old.get("id") != expected_operation:
                raise SdkInstallError("安装状态已变化，请刷新后再操作。")
            if self._alive(old.get("child")):
                raise SdkInstallError("上次安装进程尚未退出，不能重复安装。")
            self.lock = lock
            self.record = {
                "schema": 1,
                "id": operation_id,
                "current": old.get("current"),
                "phase": "checking",
                "message": "正在检查安装环境。",
            }
            self._save()
            self.task = asyncio.create_task(self._install(), name="dsh-sdk-install")
            self.task.add_done_callback(self._finished)
        except BaseException:
            lock.release()
            raise
        return self.status()

    def _finished(self, task):
        try:
            if task.cancelled() and task is self.task and self.lock:
                try:
                    self._save(phase="cancelled", message="安装已取消。")
                finally:
                    self.lock.release()
                    self.lock = None
            elif not task.cancelled():
                error = task.exception()
                if error is not None:
                    raise error
        except Exception as error:
            detail = resolver_failure(str(error))["detail"][-2000:]
            logger.error(f"SDK 安装收尾异常：{type(error).__name__} {detail}", "DSH")

    @staticmethod
    def _configured_paths():
        """读取当前配置引用的程序及数据目录。"""
        from zhenxun.configs.config import Config

        from .dsh import DshSettings

        settings = DshSettings(**(Config.get_config("ai_chat", "DSH") or {}))
        executable = (
            Path(shutil.which(settings.sdk_bin) or settings.sdk_bin)
            .expanduser()
            .resolve()
        )
        return {
            executable,
            *(
                Path(path).expanduser().resolve()
                for path in (settings.sdk_home, settings.home)
                if path
            ),
        }

    def _environment_references(self):
        """核对配置、PATH 和当前用户进程的引用；当前用户进程无法确认时保留旧环境。"""
        paths = [Path(self.record["current"]["path"]).resolve()]
        paths.extend(self._configured_paths())
        user = psutil.Process().username()
        for process in psutil.process_iter(["name", "exe", "cmdline", "username"]):
            info = process.info
            if info["username"] != user:
                # 其他用户的进程无法读取 Bot 数据目录内的私有环境
                continue
            name = (info["name"] or "").lower()
            relevant = any(
                part in name for part in ("dsh", "deepseek", "python", "node", "uv")
            )
            if not name or (relevant and (not info["exe"] or info["cmdline"] is None)):
                raise SdkInstallError("无法确认 SDK 相关进程的程序位置，旧环境已保留。")
            for value in [info["exe"], *(info["cmdline"] or [])]:
                if isinstance(value, str) and Path(value).is_absolute():
                    paths.append(Path(value).resolve())
        return set(paths)

    async def _cleanup_environments(self):
        """只清理管理目录内无配置及运行引用的旧候选环境。

        持有启动锁核对引用并改名退役，删除在锁外进行，不阻塞 SDK 连接。
        """
        from .dsh_sdk import SdkClient

        retained = 0
        failed = False
        detail = ""
        retired: list[Path] = []
        try:
            async with SdkClient._starting:
                references = await asyncio.to_thread(self._environment_references)
                parent = (self.root / "environments").resolve()
                current = self._owned_path(self.record["current"]["path"])
                for candidate in list(parent.iterdir()):
                    if not re.fullmatch(
                        r"(?:\.obsolete-)?[0-9a-f]{32}", candidate.name
                    ):
                        continue
                    attributes = candidate.lstat()
                    if (
                        candidate.is_symlink()
                        or getattr(attributes, "st_file_attributes", 0)
                        & stat.FILE_ATTRIBUTE_REPARSE_POINT
                        or not candidate.is_dir()
                        or candidate.resolve().parent != parent
                    ):
                        retained += 1
                        continue
                    candidate = self._owned_path(candidate)
                    if current.is_relative_to(candidate):
                        continue
                    if any(
                        path.is_relative_to(candidate)
                        for path in references | self._configured_paths()
                    ):
                        retained += 1
                        continue
                    if not candidate.name.startswith(".obsolete-"):
                        target = parent / (".obsolete-" + candidate.name)
                        try:
                            if target.exists():
                                raise OSError("旧环境清理目录已存在。")
                            candidate.rename(target)
                        except OSError:
                            failed = True
                            retained += 1
                            continue
                        candidate = target
                    retired.append(candidate)
            for candidate in retired:
                deletion = asyncio.create_task(
                    asyncio.to_thread(shutil.rmtree, candidate)
                )
                try:
                    await asyncio.shield(deletion)
                except asyncio.CancelledError:
                    with contextlib.suppress(OSError):
                        await deletion
                    raise
                except OSError:
                    failed = True
                    retained += 1
            if failed:
                detail = "部分旧环境清理未完成，下次修复安装时重新检查。"
        except (OSError, ValueError, RuntimeError, psutil.Error) as error:
            failed = True
            detail = "旧环境引用无法确认，已保留；下次修复安装时重新检查。"
            logger.warning(f"SDK 旧环境清理未完成：{type(error).__name__}", "DSH")
        self._save(
            cleanup_pending=failed,
            cleanup_detail=detail,
            retained_environments=retained,
        )

    def _remember(self, process):
        self.process = process
        self.process_identity = [process.pid, psutil.Process(process.pid).create_time()]
        self._save(child=self.process_identity)

    async def _terminate(self):
        """停止本任务拥有的子进程树，并核对退出结果。"""
        process = self.process
        if not process or process.returncode is not None:
            return True
        if not self._alive(self.process_identity):
            return True
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        _, alive = await asyncio.to_thread(
            psutil.wait_procs, [*children, parent], timeout=3
        )
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        _, alive = await asyncio.to_thread(psutil.wait_procs, alive, timeout=2)
        if not alive:
            await process.wait()
        return not alive

    async def _command(self, command, environment, timeout=600):
        tail = bytearray()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.root,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **(
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {"start_new_session": True}
            ),
        )
        self._remember(process)

        async def drain():
            while chunk := await process.stdout.read(4096):
                tail.extend(chunk)
                del tail[:-16384]
            return await process.wait()

        try:
            code = await asyncio.wait_for(drain(), timeout)
        except BaseException:
            await self._terminate()
            raise
        if code:
            failure = resolver_failure(tail.decode("utf-8", errors="replace"))
            self._save(detail=failure["detail"][-4000:])
            reason = {
                "dependency_disk_full": "磁盘空间不足，请清理后重试。",
                "dependency_proxy_auth": "安装代理认证失败，请检查代理配置。",
                "dependency_index_auth": "软件源拒绝访问，请检查软件源认证。",
                "dependency_tls": "软件源证书校验失败，请检查网络或证书配置。",
                "dependency_network": "软件下载失败，请检查网络后重试。",
                "dependency_python": "当前 Python 版本不满足 SDK 要求。",
                "dependency_source_required": "当前系统没有可用的 SDK 二进制安装包。",
                "dependency_conflict": "未找到兼容的 SDK 依赖组合，请查看详情。",
                "dependency_resolver_configuration": "uv 参数不兼容，请检查 uv 版本。",
            }.get(failure["code"], "SDK 安装失败，请查看详情后重试。")
            raise SdkInstallError(reason)
        return tail.decode("utf-8", errors="replace")

    async def _verify(self, executable: Path, candidate: Path):
        from .dsh_sdk_host import Worker

        worker = Worker(candidate / "verification", "chat")
        launch = asyncio.create_task(
            asyncio.wait_for(
                worker.launch(
                    {
                        "bin": str(executable),
                        "api_key": "sk-install-verification-not-a-key",
                        "model": "deepseek-chat",
                        "thinking": "",
                    }
                ),
                40,
            )
        )
        try:
            while not launch.done():
                if worker.process is not None:
                    self._remember(worker.process)
                    break
                await asyncio.sleep(0.05)
            await asyncio.wait_for(launch, 40)
            return worker.version
        finally:
            if not launch.done():
                launch.cancel()
            await asyncio.gather(launch, return_exceptions=True)
            try:
                await asyncio.wait_for(worker.close(), 10)
            except (asyncio.TimeoutError, OSError):
                if not await self._terminate():
                    raise SdkInstallError(
                        "校验进程退出未确认，请检查安装详情。"
                    ) from None

    async def _install(self):
        candidate = self.root / "environments" / self.record["id"]
        created = False
        published = False
        previous = self.record.get("current")
        try:
            if sys.version_info < (3, 10) or not uv_executable():
                raise SdkInstallError("需要 Python 3.10 或更高版本，并安装 uv。")
            if platform.system() not in {"Windows", "Linux", "Darwin"}:
                raise SdkInstallError("当前操作系统暂不支持托管安装。")
            machine = platform.machine().lower()
            if machine not in {"amd64", "x86_64", "arm64", "aarch64"} or (
                platform.system() == "Windows" and machine not in {"amd64", "x86_64"}
            ):
                raise SdkInstallError("当前系统架构没有匹配的 SDK 安装包。")
            if shutil.disk_usage(self.root).free < 512 * 1024 * 1024:
                raise SdkInstallError("安装目录可用空间不足，请至少预留 512 MiB。")
            self._owned_path(candidate)
            if candidate.exists():
                raise SdkInstallError("候选目录已存在，请刷新后重新安装。")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.mkdir()
            created = True
            environment = installer_environment()
            environment.update(
                UV_CACHE_DIR=str(self.root / "cache"),
                UV_PYTHON_DOWNLOADS="never",
                UV_NO_PROGRESS="1",
            )
            self._save(phase="creating", message="正在创建独立运行环境。")
            await self._command(
                uv_command("venv", "--python", sys.executable, str(candidate)),
                environment,
                60,
            )
            scripts = candidate / ("Scripts" if os.name == "nt" else "bin")
            python = scripts / ("python.exe" if os.name == "nt" else "python")
            executable = scripts / ("dsh.exe" if os.name == "nt" else "dsh")
            self._save(phase="installing", message="正在下载并安装 SDK。")
            await self._command(
                uv_command(
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    *uv_binary_options(wheels_only=True),
                    "deepseek-harness-sdk==" + SDK_VERSION,
                ),
                environment,
            )
            self._save(
                phase="verifying", message="正在验证版本及连接协议，不会调用模型。"
            )
            metadata = await self._command(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import importlib.metadata as m,json; "
                    "print('ZHENXUN_SDK_METADATA=' + json.dumps("
                    "[m.version('deepseek-harness-sdk'),"
                    "m.version('deepseek-harness-runtime-bin')]))",
                ],
                environment,
                20,
            )
            markers = [
                line.removeprefix("ZHENXUN_SDK_METADATA=")
                for line in metadata.splitlines()
                if line.startswith("ZHENXUN_SDK_METADATA=")
            ]
            try:
                versions = json.loads(markers[0]) if len(markers) == 1 else None
            except ValueError as error:
                raise SdkInstallError(
                    "SDK 版本校验输出无法识别，未启用该安装。"
                ) from error
            if versions != [SDK_VERSION, SDK_VERSION] or not executable.is_file():
                raise SdkInstallError("SDK 文件或版本与预期不符，未启用该安装。")
            protocol = await self._verify(executable, candidate)
            self._save(
                phase="ready",
                message="SDK 已安装，可以保存并连接。",
                child=None,
                current={
                    "path": str(executable.resolve()),
                    "version": SDK_VERSION,
                    "protocol": protocol,
                },
                detail="",
            )
            published = True
            await self._cleanup_environments()
        except asyncio.CancelledError:
            self._save(
                phase="ready" if published else "cancelled",
                message="SDK 已安装，旧环境清理未完成。"
                if published
                else "安装已取消，原有 SDK 未变更。",
                current=self.record["current"] if published else previous,
                cleanup_pending=published,
            )
        except Exception as error:
            self._save(
                phase="ready" if published else "failed",
                message=(
                    "SDK 已安装，旧环境清理未完成。"
                    if published
                    else str(error)
                    if isinstance(error, SdkInstallError)
                    else "安装未完成：" + type(error).__name__
                ),
                current=self.record["current"] if published else previous,
                cleanup_pending=published,
                detail=self.record.get("detail", ""),
            )
        finally:
            try:
                stopped = await self._terminate()
                if not published:
                    if stopped and created and candidate.exists():
                        await asyncio.to_thread(
                            shutil.rmtree, self._owned_path(candidate)
                        )
                    self._save(
                        cleanup_pending=not stopped,
                        child=None if stopped else self.process_identity,
                    )
            except Exception:
                self._save(cleanup_pending=True)
            finally:
                if self.lock:
                    self.lock.release()
                    self.lock = None
                self.process = None

    async def cancel(self, operation_id):
        """取消当前安装，完成子进程清理后才返回结果。"""
        if self.record.get("id") != operation_id or not self.task or self.task.done():
            raise SdkInstallError("安装任务已变化，请刷新状态。")
        self._save(phase="stopping", message="正在停止安装并清理临时文件。")
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(self.task)
        return self.status()

    async def close(self):
        self.closing = True
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self.task), 10)


sdk_installer = SdkInstaller(DATA_PATH / "ai" / "dsh" / "sdk-install")
