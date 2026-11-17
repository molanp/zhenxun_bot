import asyncio
import base64
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
import re
from typing import Any, Literal

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession  # type: ignore
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import PaginatedRequestParams
from pydantic import ValidationError

from zhenxun.services.ai.core.models import ToolDefinition
from zhenxun.services.ai.core.stream_events import UserCustomEvent
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolkitConfig, ToolResult
from zhenxun.services.ai.utils.logger import log_tool as logger
from zhenxun.services.network_proxy import ManagedAsyncClient
from zhenxun.utils.lifespan import LifespanManager
from zhenxun.utils.pydantic_compat import model_dump

_MESSAGE_START_CHARS = {"{", "["}
_LITERAL_PREFIXES: tuple[str, ...] = ("true", "false", "null")


class _RunTransport(StreamableHTTPTransport):
    """MCP 请求响应按调用上下文决定是否允许断流续接。"""

    def active_contexts(self):
        return ()

    async def handle_get_stream(self, client, read_stream_writer):
        """通知流断开时中断受影响 AI 运行，其他调用保留 SDK 的连接恢复规则。"""
        from mcp.client.streamable_http import (
            DEFAULT_RECONNECTION_DELAY_MS,
            MAX_RECONNECTION_ATTEMPTS,
        )
        from mcp.shared._httpx_utils import sse_within_origin

        attempt, last_event_id, retry_ms = 0, None, DEFAULT_RECONNECTION_DELAY_MS
        while attempt < MAX_RECONNECTION_ATTEMPTS:
            if not self.session_id:
                return
            headers = self._prepare_headers()
            if last_event_id:
                headers["Last-Event-ID"] = last_event_id
            try:
                async with sse_within_origin(
                    client,
                    self.url,
                    headers=headers,
                    max_event_size=self.max_sse_event_size,
                ) as source:
                    if source.response.status_code == 405:
                        return
                    source.response.raise_for_status()
                    async for event in source:
                        last_event_id = event.id or last_event_id
                        retry_ms = event.retry if event.retry is not None else retry_ms
                        await self._handle_sse_event(event, read_stream_writer)
                attempt = 0
            except Exception:
                attempt += 1
            active = list(self.active_contexts())
            strict = [ctx for ctx in active if ctx and ctx.run.fail_fast]
            for ctx in strict:
                ctx.run.operations["interruption"] = "mcp_notification_stream"
                token = ctx.run.cancellation_token
                while token and token.parent:
                    token = token.parent
                if token:
                    token.cancel("network")
            if strict and len(strict) == len(active):
                return
            if attempt < MAX_RECONNECTION_ATTEMPTS:
                await anyio.sleep(retry_ms / 1000)

    async def _handle_reconnection(
        self, ctx, last_event_id, retry_interval_ms=None, attempt=0
    ):
        from zhenxun.services.ai.run.context import fail_fast_enabled

        if fail_fast_enabled():
            await self._resolve_abandoned_request(
                ctx.read_stream_writer,
                ctx.session_message.message.id,
                "AI run interrupted: response stream ended",
            )
            return
        await super()._handle_reconnection(
            ctx, last_event_id, retry_interval_ms, attempt
        )


@asynccontextmanager
async def _streamable_transport(url: str, http_client, active_contexts):
    """沿用 SDK 传输与发送者上下文，只覆盖 AI 请求的断流恢复策略。"""
    from mcp.shared._context_streams import create_context_streams

    transport = _RunTransport(url)
    transport.active_contexts = active_contexts
    incoming, reader = create_context_streams[SessionMessage | Exception](0)
    writer, outgoing = create_context_streams[SessionMessage](0)
    async with incoming, reader, writer, outgoing, anyio.create_task_group() as group:

        def start_get_stream():
            group.start_soon(transport.handle_get_stream, http_client, incoming)

        group.start_soon(
            transport.post_writer,
            http_client,
            outgoing,
            incoming,
            writer,
            start_get_stream,
            group,
        )
        try:
            yield reader, writer
        finally:
            if transport.session_id:
                await transport.terminate_session(http_client)
            group.cancel_scope.cancel()


def _protocol_value(value: Any, name: str, alias: str, default: Any = None) -> Any:
    """读取 MCP 模型字段及其协议别名。"""
    return getattr(value, name, getattr(value, alias, default))


def _should_ignore_exception(exc: Exception) -> bool:
    """
    判断该异常是否是由非 JSON 的脏数据标准输出引起的。
    如果是脏数据，则可以安全忽略。
    """
    if not isinstance(exc, ValidationError):
        return False

    errors = exc.errors()
    first = next(iter(errors), None)
    if not first or first.get("type") != "json_invalid":
        return False

    input_value = first.get("input")
    if not isinstance(input_value, str):
        return False

    stripped = input_value.strip()
    if not stripped:
        return True

    first_char = stripped[0]
    lowered = stripped.lower()

    if first_char in _MESSAGE_START_CHARS or any(
        lowered.startswith(prefix) for prefix in _LITERAL_PREFIXES
    ):
        return False

    return True


@asynccontextmanager
async def filtered_stdio_client(
    server_name: str, server: StdioServerParameters
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    """
    包裹官方的 stdio_client，拦截并过滤掉非 JSON 格式的标准输出噪音。
    """
    async with stdio_client(server=server) as (read_stream, write_stream):
        filtered_send, filtered_recv = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)

        async def _forward_stdout() -> None:
            try:
                async with read_stream:
                    async for item in read_stream:
                        if isinstance(item, Exception) and _should_ignore_exception(
                            item
                        ):
                            if isinstance(item, ValidationError):
                                err_input = item.errors()[0].get("input", "")
                                logger.debug(
                                    f"🔇 [MCP Stdout Filter] {server_name} 忽略脏数据: "
                                    f"{str(err_input)[:100].strip()}..."
                                )
                            continue
                        await filtered_send.send(item)
            except anyio.ClosedResourceError:
                pass
            finally:
                await filtered_send.aclose()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_forward_stdout)
            try:
                yield filtered_recv, write_stream
            finally:
                tg.cancel_scope.cancel()


class MCPRemoteTool(BaseTool):
    """远端 MCP 工具在本地的代理对象"""

    @property
    def identity(self) -> str:
        """MCP 服务及原始工具名组成的执行身份。"""
        return f"mcp:{self.toolkit.server_name}:{self.original_tool_name}"

    def __init__(
        self,
        name: str,
        original_tool_name: str,
        description: str,
        parameters: dict,
        toolkit: "MCPToolkit",
    ):
        """
        初始化远端 MCP 工具在本地的代理对象。

        参数:
            name: 工具在本地注册的唯一标识名称。
            original_tool_name: 该工具在远端 MCP 服务器上的原始名称。
            description: 工具的描述信息，用于指导大模型选用此工具。
            parameters: 工具参数的 JSON 模式描述。
            toolkit: 所属的 MCPToolkit 工具箱实例。
        """
        super().__init__(name=name, description=description)
        self.original_tool_name = original_tool_name
        self.parameters = parameters
        self.toolkit = toolkit
        if toolkit.admin_level > 0:
            from zhenxun.services.ai.tools.core.decorators import require_admin_level

            self.settings = self.settings.merge(
                require_admin_level(toolkit.admin_level)
            )

    @property
    def effective_ttl(self) -> float:
        """获取当前工具关联的有效生存期 TTL"""
        try:
            from zhenxun.services.ai.config import get_llm_config

            mcp_ttl = get_llm_config().agent_settings.mcp_cleanup_timeout
            return 31536000.0 if mcp_ttl <= 0 else float(mcp_ttl)
        except Exception:
            return 31536000.0 if self.toolkit.ttl <= 0 else float(self.toolkit.ttl)

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            metadata=self.metadata or {},
        )
        if context and self.settings.capabilities:
            from zhenxun.services.ai.capabilities.base import CombinedCapability

            combined_cap = CombinedCapability(self.settings.capabilities)
            defs = await combined_cap.prepare_tools(context, [tool_def])
            if not defs:
                return None
            tool_def = defs[0]
        return tool_def

    async def execute(self, context: RunContext | None = None, **kwargs) -> ToolResult:
        await self.toolkit.lifespan_manager.touch(
            self.toolkit.server_name, self.toolkit.effective_ttl
        )
        try:
            result = await self.toolkit.call_tool(
                self.original_tool_name, kwargs, context
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            from mcp.shared.exceptions import MCPError

            if isinstance(error, MCPError) and error.code in {-32600, -32601, -32602}:
                return ToolResult(
                    output=f"远端拒绝了 MCP 请求，请检查工具名或参数（{error.code}）。"
                ).as_error(False)
            from zhenxun.services.ai.run.context import interrupt_network

            interrupt_network("mcp_call", error)
            logger.warning(f"MCP 工具 {self.name} 结果未确认: {type(error).__name__}")
            return ToolResult(
                output="MCP 操作结果未确认，未自动重试；请核查外部结果。"
            ).as_error(False, uncertain=True)

        if _protocol_value(result, "is_error", "isError", False):
            return ToolResult(output=str(result.content)).as_error(False)

        from zhenxun.services.ai.core.messages import ImagePart, TextPart

        output_content = []
        img_count = 0

        for item in result.content:
            item_type = getattr(item, "type", "text")

            if item_type == "image":
                b64_data = getattr(item, "data", "")
                mime_type = _protocol_value(item, "mime_type", "mimeType", "image/png")
                if b64_data:
                    try:
                        img_bytes = base64.b64decode(b64_data)
                        output_content.append(
                            ImagePart(raw=img_bytes, mime_type=mime_type)
                        )
                        img_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"MCP 图片 Base64 解码失败: {e}")

                output_content.append(TextPart(text="[图片解码失败]"))

            elif item_type == "text":
                text = getattr(item, "text", str(item))
                output_content.append(TextPart(text=text))

                md_images = re.findall(r"!\[.*?\]\((https?://[^\)]+)\)", text)
                for img_url in md_images:
                    output_content.append(ImagePart(url=img_url))
                    img_count += 1
            else:
                dumped = model_dump(item) if hasattr(item, "model_dump") else str(item)
                output_content.append(TextPart(text=str(dumped)))

        tool_result = ToolResult(output=output_content)
        logger.info(
            f"获取到 {len(output_content)} 条返回数据，提取了 {img_count} 张图片"
        )

        if img_count > 0 and context and context.run.event_bus:
            await context.run.event_bus.emit(UserCustomEvent(display=output_content))

        return tool_result


class MCPToolkit(BaseToolkit):
    """模型上下文协议 (MCP) 的工具箱封装 (支持声明式挂载与动态隔离)"""

    def __init__(
        self,
        server_name: str,
        prefix: str | None = None,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        install_command: str | None = None,
        timeout: int = 30,
        admin_level: int = 0,
        header_provider: Callable[[RunContext], dict[str, str]] | None = None,
        env_provider: Callable[[RunContext], dict[str, str]] | None = None,
        ttl: int = 600,
    ):
        """
        初始化模型上下文协议 (MCP) 的工具箱封装。

        参数:
            server_name: MCP 服务器的唯一标识名称。
            prefix: 工具名称的前缀，防止工具命名冲突。
            transport: 连接 MCP 服务器的通信传输协议，可选 "stdio", "sse", "streamable-http"。
            command: 用于 stdio 模式启动服务器的可执行命令。
            args: 启动服务器时附加的命令行参数。
            url: 用于 sse 或 streamable-http 模式的服务器连接 URL。
            headers: 静态配置的 HTTP 请求头。
            env: 启动服务器时的进程环境变量。
            cwd: 启动服务器时的进程工作目录。
            install_command: 首次启动前执行的环境热装配/依赖安装命令。
            timeout: 初始化和请求网络接口时的超时秒数，默认 30。
            admin_level: 调用工具所需的管理权限等级，默认 0 (无限制)。
            header_provider: 动态生成 HTTP 头部信息的工厂函数。
            env_provider: 动态生成进程环境变量的工厂函数。
            ttl: 闲置清理的生存时间 (秒)，默认 600。
        """  # noqa: E501
        super().__init__(
            config=ToolkitConfig(prefix=prefix) if prefix is not None else None
        )
        self.server_name = server_name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.url = url
        self.headers = headers or {}
        self.env = env or {}
        self.cwd = cwd
        self.install_command = install_command
        self.timeout = timeout
        self.admin_level = admin_level
        self.header_provider = header_provider
        self.env_provider = env_provider
        self.ttl = ttl

        self._shared_session: ClientSession | None = None
        self._tools: list[BaseTool] = []
        self._is_initialized = False

        self.lifespan_manager = LifespanManager()

        self._shared_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._ready_event: asyncio.Event = asyncio.Event()
        self._init_exception: Exception | None = None

    @property
    def effective_ttl(self) -> float:
        """获取当前工具箱的有效闲置生存期 TTL"""
        try:
            from zhenxun.services.ai.config import get_llm_config

            mcp_ttl = get_llm_config().agent_settings.mcp_cleanup_timeout
            return 31536000.0 if mcp_ttl <= 0 else float(mcp_ttl)
        except Exception:
            return 31536000.0 if self.ttl <= 0 else float(self.ttl)

    async def _run_install_command(self, force: bool = False):
        """在本地运行预热依赖安装命令以准备 MCP 服务器环境"""
        if not self.install_command or not self.cwd:
            return

        from pathlib import Path

        marker_file = Path(self.cwd) / ".zx_installed"

        heuristic_missing = False
        if (
            "npm" in self.install_command
            and not (Path(self.cwd) / "node_modules").exists()
        ):
            heuristic_missing = True

        if not force and not heuristic_missing and marker_file.exists():
            return

        reason = "强制触发自愈重装" if force else "首次启动或检测到环境缺失"
        logger.info(
            f"🔧 [{self.server_name}] {reason}，正在执行: `{self.install_command}` ..."
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                self.install_command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err_msg = (
                    stderr.decode("utf-8", errors="ignore")
                    if stderr
                    else stdout.decode("utf-8", errors="ignore")
                )
                logger.error(
                    f"❌ [{self.server_name}] 环境安装失败!\n错误输出:\n{err_msg}"
                )
                raise RuntimeError(f"安装命令执行失败: {self.install_command}")
            else:
                marker_file.touch()
                logger.info(f"✅ [{self.server_name}] 环境装配成功！")
        except Exception as e:
            logger.error(f"执行预热安装异常: {e}")
            raise e

    async def get_session(
        self, context: RunContext | None = None
    ) -> ClientSession | None:
        """获取或建立与 MCP 服务器的会话连接，并刷新其生存状态"""
        await self.lifespan_manager.register(
            self.server_name,
            ttl=self.effective_ttl,
            cleanup_callback=self.release_resource,
        )
        if not self._is_initialized:
            await self.initialize(context)
        return self._shared_session

    async def call_tool(
        self, name: str, arguments: dict, context: RunContext | None = None
    ) -> Any:
        """登记当前调用身份，让共享连接失效只中断对应的 AI 运行。"""
        if not hasattr(self, "_active_calls"):
            self._active_calls = {}
        task = asyncio.current_task()
        self._active_calls[task] = context
        try:
            return await self._call_tool(name, arguments, context)
        finally:
            self._active_calls.pop(task, None)

    async def _call_tool(
        self, name: str, arguments: dict, context: RunContext | None = None
    ) -> Any:
        """调用远端工具，连接中断时立即抛出异常

        参数:
            name: 远端工具原始名称
            arguments: 工具参数
            context: 运行上下文

        返回:
            Any: MCP 工具调用结果
        """
        session = await self.get_session(context)
        task = self._shared_task
        if not session or not task:
            raise ConnectionError(f"MCP 服务器 '{self.server_name}' 未连接")
        call = asyncio.ensure_future(session.call_tool(name, arguments))
        try:
            await asyncio.wait({call, task}, return_when=asyncio.FIRST_COMPLETED)
            if call.done():
                return call.result()
            raise ConnectionError(f"MCP 服务器 '{self.server_name}' 连接已断开")
        finally:
            if not call.done():
                call.cancel()
            await asyncio.gather(call, return_exceptions=True)

    async def _spawn_session_task(self, dynamic_headers: dict, dynamic_env: dict):
        """在后台异步拉起 MCP 传输协议通道并进行服务初始化"""
        max_attempts = 2 if (self.transport == "stdio" and self.install_command) else 1
        from zhenxun.services.ai.run.context import fail_fast_enabled

        if fail_fast_enabled():
            max_attempts = 1

        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    if self.transport == "stdio":
                        await self._run_install_command(force=(attempt > 1))

                    async with AsyncExitStack() as stack:
                        read_stream: Any = None
                        write_stream: Any = None
                        if self.transport == "stdio":
                            if not self.command:
                                raise ValueError("stdio requires 'command'")

                            params = StdioServerParameters(
                                command=self.command,
                                args=self.args,
                                env=dynamic_env,
                                cwd=self.cwd,
                            )
                            transport_ctx = filtered_stdio_client(
                                self.server_name, params
                            )
                        elif self.transport == "sse":
                            if not self.url:
                                raise ValueError("sse requires 'url'")
                            transport_ctx = sse_client(
                                url=self.url,
                                headers=dynamic_headers,
                                timeout=self.timeout,
                            )
                        elif self.transport == "streamable-http":
                            if not self.url:
                                raise ValueError("streamable-http requires 'url'")

                            http_client = ManagedAsyncClient(
                                headers=dynamic_headers, timeout=self.timeout
                            )
                            await stack.enter_async_context(http_client)
                            transport_ctx = _streamable_transport(
                                url=self.url,
                                http_client=http_client,
                                active_contexts=lambda: tuple(
                                    getattr(self, "_active_calls", {}).values()
                                ),
                            )
                        else:
                            raise ValueError(f"Unknown transport: {self.transport}")

                        transport = await stack.enter_async_context(transport_ctx)
                        read_stream, write_stream = transport[0], transport[1]

                        client_session = await stack.enter_async_context(
                            ClientSession(read_stream, write_stream)
                        )
                        await client_session.initialize()

                        self._shared_session = client_session

                        if not self._is_initialized:
                            mcp_tools_res = await client_session.list_tools()
                            tools_list = list(mcp_tools_res.tools)
                            cursor = _protocol_value(
                                mcp_tools_res, "next_cursor", "nextCursor"
                            )

                            while cursor:
                                mcp_tools_res = await client_session.list_tools(
                                    params=PaginatedRequestParams(cursor=cursor)
                                )
                                tools_list.extend(mcp_tools_res.tools)
                                cursor = _protocol_value(
                                    mcp_tools_res, "next_cursor", "nextCursor"
                                )

                            for t in tools_list:
                                t_name = t.name.replace("-", "_")
                                final_name = (
                                    f"{self.config.prefix}{t_name}"
                                    if self.config.prefix
                                    else t_name
                                )
                                self._tools.append(
                                    MCPRemoteTool(
                                        name=final_name,
                                        original_tool_name=t.name,
                                        description=t.description or "",
                                        parameters=_protocol_value(
                                            t, "input_schema", "inputSchema"
                                        ),
                                        toolkit=self,
                                    )
                                )
                            self._is_initialized = True

                        self._ready_event.set()
                        logger.info(
                            f"成功连接 MCP 服务器: {self.server_name}, "
                            f"获取了 {len(self._tools)} 个工具"
                        )

                        await self._stop_event.wait()
                        return

                except Exception as e:
                    if attempt < max_attempts:
                        logger.warning(
                            f"⚠️ [{self.server_name}] 进程启动或运行异常崩溃，"
                            "疑似环境损坏。触发自愈机制 (准备重试)..."
                        )
                        self._init_exception = e
                        self._shared_session = None
                        if not self._is_initialized:
                            self._is_initialized = False
                        continue

                    raise e

        except Exception as e:
            while getattr(e, "exceptions", None):
                e = e.exceptions[0]  # type: ignore
            self._init_exception = e
            logger.error(
                f"MCP 服务器 '{self.server_name}' "
                f"初始化连接失败（模式: {self.transport}）。"
                f"错误原因: {e}"
            )
            if "Connection closed" in str(e):
                logger.error(
                    "提示：若是使用沙箱隧道，请进入 Docker Desktop 检查容器的日志。"
                    "多半是因为 npx 命令执行出错"
                    "（例如：镜像中缺少 Node 环境或国内网络无法连接 npm 等）。"
                )
        finally:
            self._is_initialized = False
            self._shared_session = None
            self._ready_event.set()

    async def initialize(self, context: RunContext | None = None):
        """清空当前工具集并异步执行 MCP 服务端的全套初始化流程"""
        if self._is_initialized:
            return

        self._tools.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._init_exception = None

        dynamic_headers = self.headers.copy()
        if self.header_provider and context:
            dynamic_headers.update(self.header_provider(context))

        dynamic_env = self.env.copy()
        if self.env_provider and context:
            dynamic_env.update(self.env_provider(context))

        self._shared_task = asyncio.create_task(
            self._spawn_session_task(dynamic_headers, dynamic_env)
        )

        await self._ready_event.wait()

        if self._init_exception:
            from zhenxun.services.ai.run.context import interrupt_network

            interrupt_network("mcp_initialize", self._init_exception)
            raise self._init_exception

    async def get_tools(self, context: RunContext | None = None) -> dict[str, BaseTool]:
        await self.lifespan_manager.register(
            self.server_name,
            ttl=self.effective_ttl,
            cleanup_callback=self.release_resource,
        )
        if not self._is_initialized:
            await self.initialize(context)

        tools_dict = {}
        for t in self._tools:
            key = t.name if t.name not in tools_dict else t.identity
            tools_dict[key] = t
        return tools_dict

    async def release_resource(self, resource_id: str):
        await self.close()

    async def close(self):
        """彻底关闭 MCP 工具箱，注销生存期并释放底层 WebSocket/进程通道"""
        current_task = asyncio.current_task()

        if (
            self.lifespan_manager._watchdog_task
            and self.lifespan_manager._watchdog_task is not current_task
        ):
            await self.lifespan_manager.stop()

        self._stop_event.set()
        self._is_initialized = False
        self._shared_session = None
        self._tools.clear()

        if (
            self._shared_task
            and not self._shared_task.done()
            and self._shared_task is not current_task
        ):
            try:
                await asyncio.wait_for(self._shared_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"MCP 服务器 '{self.server_name}' 的任务执行超时，正在取消。"
                )
                self._shared_task.cancel()
        self._shared_task = None
