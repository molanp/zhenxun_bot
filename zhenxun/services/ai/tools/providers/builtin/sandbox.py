import asyncio
import re
from typing import ClassVar

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.sandbox.docker import (
    LANGUAGES,
    DockerSandbox,
    normalize_language,
)
from zhenxun.services.ai.sandbox.manager import SandboxManager
from zhenxun.services.ai.tools.core.decorators import Rules, tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolkitConfig, ToolOptions, ToolResult
from zhenxun.services.ai.utils.logger import log_tool as logger

_FENCE = re.compile(r"^```[\w+-]*\r?\n(.*?)\r?\n?```$", re.DOTALL)


def sandbox_instructions() -> str:
    """按当前配置生成沙箱使用说明

    返回:
        str: 交给模型的说明
    """
    settings = get_llm_config().sandbox
    network = "可以联网" if settings.network else "不能联网，无法下载或安装依赖"
    return (
        "## 沙箱\n"
        f"你有一个独立的 Linux 容器（镜像 {settings.docker_image}，"
        f"{settings.cpus:g} 核 CPU、{settings.memory_mb}MB 内存，{network}），"
        "工作目录 /workspace 中的文件在本会话内保留，闲置一段时间后容器会被删除。\n"
        f"1. 短时任务用 `execute_code` 或 `execute_terminal_command`，"
        f"单次最长 {settings.exec_timeout} 秒；代码中不要使用 input() 等交互输入，"
        "变量不会在两次调用之间保留。\n"
        "2. 需要交互或长时间运行的程序：先用 `write_sandbox_file` 保存代码，"
        "再用 `execute_terminal_command(is_interactive=True)` 启动，"
        "用 `send_sandbox_input` / `read_sandbox_screen` 交互，"
        "卡住时用 `interrupt_sandbox` 中断。"
    )


class SandboxToolkit(BaseToolkit):
    default_prefix = ""
    _default_config: ClassVar[ToolkitConfig] = ToolkitConfig(
        shared_options=Rules.superuser()
    )
    """默认仅超级用户可见，实例化时传入 shared_options 可覆盖"""

    def __init__(self, **kwargs):
        kwargs.setdefault("instructions", sandbox_instructions())
        super().__init__(**kwargs)

    @staticmethod
    async def _sandbox(context: RunContext) -> DockerSandbox:
        """获取当前会话的沙箱"""
        return await SandboxManager.get(context.session_id or "default")

    @staticmethod
    def _terminal(context: RunContext):
        """获取当前会话正在使用的终端"""
        sandbox = SandboxManager.find(context.session_id or "default")
        return sandbox.terminal if sandbox else None

    @tool(
        name="execute_code",
        description=(
            "在沙箱中执行一段代码并返回输出。language 可选 "
            + "、".join(LANGUAGES)
            + "（javascript 需要镜像中有 node）。"
        ),
    )
    async def execute_code(
        self, code: str, language: str, context: RunContext
    ) -> ToolResult:
        lang = normalize_language(language)
        if lang is None:
            return ToolResult(
                output=f"不支持的语言 {language}，可用: {'、'.join(LANGUAGES)}。"
            ).as_error()
        code = code.strip()
        if match := _FENCE.match(code):
            code = match.group(1)
        started = False
        try:
            sandbox = await self._sandbox(context)
            timeout = get_llm_config().sandbox.exec_timeout
            started = True
            result = await sandbox.run_code(lang, code, timeout)
        except Exception as e:
            logger.warning(f"沙箱调用未完成: {type(e).__name__}")
            return ToolResult(output="沙箱执行未完成，请核查结果。").as_error(
                False, uncertain=started
            )
        logger.info(f"沙箱执行 {lang} 代码 {len(code)} 字符，退出码 {result.exit_code}")
        text = result.text()
        if result.exit_code == 127:
            text += f"\n镜像中可能没有 {LANGUAGES[lang][1][0]}。"
        return ToolResult(output=text)

    @tool(
        name="execute_terminal_command",
        description=(
            "在沙箱中执行 shell 命令。普通命令保持 is_interactive=False，"
            "等待结束后返回输出；需要交互输入或长期运行的程序设为 True，"
            "在虚拟终端中启动并返回当前画面。"
        ),
    )
    async def execute_terminal_command(
        self, command: str, context: RunContext, is_interactive: bool = False
    ) -> ToolResult:
        started = False
        try:
            sandbox = await self._sandbox(context)
            started = True
            if not is_interactive:
                timeout = get_llm_config().sandbox.exec_timeout
                result = await sandbox.exec(["sh", "-c", command], timeout)
                return ToolResult(output=result.text())
            terminal = await sandbox.open_terminal(command)
        except Exception as e:
            logger.warning(f"沙箱命令未完成: {type(e).__name__}")
            return ToolResult(output="沙箱命令未完成，请核查结果。").as_error(
                False, uncertain=started
            )
        await asyncio.sleep(1.5)
        return ToolResult(output=f"终端已启动，当前画面:\n{terminal.screen()}")

    @tool(
        name="send_sandbox_input",
        description="向沙箱终端中运行的程序发送输入，需要回车时在末尾加 \\n。",
    )
    async def send_sandbox_input(self, text: str, context: RunContext) -> ToolResult:
        terminal = self._terminal(context)
        if not terminal:
            return ToolResult(
                output="没有打开的终端，请先调用 execute_terminal_command"
                "(is_interactive=True)。"
            ).as_error()
        await terminal.send(text.replace("\\n", "\n").replace("\\t", "\t"))
        await asyncio.sleep(1.5)
        return ToolResult(output=f"已发送，当前画面:\n{terminal.screen()}")

    @tool(
        name="read_sandbox_screen",
        rules=ToolOptions(metadata={"read_only": True}),
        description="读取沙箱终端的当前画面。",
    )
    async def read_sandbox_screen(self, context: RunContext) -> ToolResult:
        terminal = self._terminal(context)
        if not terminal:
            return ToolResult(output="没有打开的终端。").as_error()
        return ToolResult(output=f"当前画面:\n{terminal.screen()}")

    @tool(
        name="interrupt_sandbox",
        description="向沙箱终端发送 Ctrl+C，中断卡住或死循环的程序。",
    )
    async def interrupt_sandbox(self, context: RunContext) -> ToolResult:
        terminal = self._terminal(context)
        if not terminal:
            return ToolResult(output="没有打开的终端。").as_error()
        await terminal.interrupt()
        await asyncio.sleep(1)
        return ToolResult(output=f"已发送 Ctrl+C，当前画面:\n{terminal.screen()}")

    @tool(
        name="write_sandbox_file",
        description="把文本写入沙箱文件，相对路径位于 /workspace 下。",
    )
    async def write_sandbox_file(
        self, path: str, content: str, context: RunContext
    ) -> ToolResult:
        started = False
        try:
            sandbox = await self._sandbox(context)
            started = True
            await sandbox.write_file(path, content.encode("utf-8"))
        except Exception as e:
            logger.warning(f"沙箱写入未完成: {type(e).__name__}")
            return ToolResult(output="文件写入未完成，请核查内容。").as_error(
                False, uncertain=started
            )
        return ToolResult(output=f"已写入 {path}（{len(content)} 字符）")

    @tool(
        name="read_sandbox_file",
        rules=ToolOptions(metadata={"read_only": True}),
        description="读取沙箱文件的文本内容（最多 1MB），相对路径位于 /workspace 下。",
    )
    async def read_sandbox_file(self, path: str, context: RunContext) -> ToolResult:
        try:
            sandbox = await self._sandbox(context)
            data = await sandbox.read_file(path)
        except Exception as e:
            return ToolResult(output=f"读取失败：{e or type(e).__name__}").as_error()
        return ToolResult(output=data.decode("utf-8", "replace"))
