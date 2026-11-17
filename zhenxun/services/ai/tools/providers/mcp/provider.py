from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import re
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.core.protocols.tool import (
    ToolExecutable,
    ToolProvider,
    ToolResolvable,
)
from zhenxun.services.ai.tools.models import ResolvedToolPayload
from zhenxun.services.ai.utils.logger import log_tool as logger
from zhenxun.utils.pydantic_compat import model_dump, model_validate, model_validator

if TYPE_CHECKING:
    from .toolkit import MCPToolkit


MCP_PATH = DATA_PATH / "ai" / "mcp.json"


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "sse", "streamable-http"] = Field(default="stdio")
    """传输协议类型：stdio / sse / streamable-http"""
    url: str | None = Field(default=None)
    """远端地址（用于 sse 或 streamable-http）"""
    headers: dict[str, str] | None = Field(default=None)
    """HTTP 请求头（用于 sse 或 streamable-http 的接口鉴权）"""
    timeout: int = Field(default=30)
    """请求超时时间(秒)"""
    command: str | None = None
    """启动命令（用于 stdio）"""
    args: list[str] = Field(default_factory=list)
    """命令参数列表"""
    env: dict[str, str] | None = None
    """启动子进程的环境变量（可选）"""
    cwd: str | None = Field(default=None)
    """进程的工作目录"""
    install_command: str | None = Field(default=None)
    """首次运行前的安装/预热命令"""
    description: str | None = None
    """服务描述信息（用于配置可读性与展示）"""
    enabled: bool = Field(default=True)
    """是否全局启用"""
    admin_level: int = Field(default=0)
    """执行该服务器工具所需的群管等级"""

    @model_validator(mode="before")
    @classmethod
    def _normalize_config(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if "type" in values and "transport" not in values:
            values["transport"] = values.pop("type")

        transport_val = values.get("transport")
        if isinstance(transport_val, str) and transport_val.lower() == "streamablehttp":
            values["transport"] = "streamable-http"

        headers = values.get("headers")
        if isinstance(headers, dict):
            new_headers = {}
            for k, v in headers.items():
                if k.lower() == "authorization" and isinstance(v, str):
                    if not any(
                        v.startswith(prefix)
                        for prefix in ("Bearer ", "Basic ", "Digest ")
                    ):
                        v = f"Bearer {v}"
                new_headers[k] = v
            values["headers"] = new_headers

        return values


class MCPToolsConfig(BaseModel):
    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class GlobalMCPProvider(ToolProvider):
    def __init__(self):
        self._config: MCPToolsConfig | None = None
        self._toolkits: dict[str, MCPToolkit] = {}
        self._discovery_lock = asyncio.Lock()
        self._discovered_tools: dict[str, ToolExecutable] | None = None
        self.env_provider: Callable[[Any], dict[str, str]] | None = None
        self.header_provider: Callable[[Any], dict[str, str]] | None = None
        self._code_registered_servers: dict[str, tuple[MCPServerConfig, bool]] = {}
        self._known_tools: dict[str, dict[str, str]] = {}

    def servers(self) -> dict[str, MCPServerConfig]:
        """mcp.json 中的服务配置，尚未加载时为空

        返回:
            dict[str, MCPServerConfig]: 服务名 -> 配置
        """
        return dict(self._config.mcpServers) if self._config else {}

    def known_tools(self) -> dict[str, dict[str, str]]:
        """已连接过的服务提供的工具

        返回:
            dict[str, dict[str, str]]: 服务名 -> {工具名: 描述}
        """
        return {k: dict(v) for k, v in self._known_tools.items() if k in self._toolkits}

    def _remember(self, server: str, tools: dict[str, ToolExecutable]) -> None:
        """记录服务提供的工具名与描述"""
        self._known_tools[server] = {
            name: str(getattr(t, "description", "") or "") for name, t in tools.items()
        }

    async def save_server(self, name: str, conf: MCPServerConfig) -> None:
        """新增或替换 mcp.json 中的服务，按新配置重建连接；已知工具保留到下次连接时刷新

        参数:
            name: 服务名
            conf: 服务配置
        """
        await self.initialize()
        assert self._config is not None
        self._config.mcpServers[name] = conf
        self._save_config()
        if tk := self._toolkits.pop(name, None):
            await tk.close()
        self._setup_toolkit(name, conf)
        self._discovered_tools = None

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        """启用或停用服务，停用时关闭已建立的连接

        参数:
            name: 服务名
            enabled: 是否启用

        返回:
            bool: 服务是否存在
        """
        await self.initialize()
        assert self._config is not None
        if not (conf := self._config.mcpServers.get(name)):
            return False
        if conf.enabled != enabled:
            conf.enabled = enabled
            if tk := self._toolkits.pop(name, None):
                await tk.close()
            self._setup_toolkit(name, conf)
            self._discovered_tools = None
            self._save_config()
        return True

    async def reload(self) -> None:
        """关闭所有连接并重新读取 mcp.json"""
        await self.shutdown()
        self._config = None
        self._discovered_tools = None
        self._known_tools.clear()
        await self.initialize()

    async def test(
        self, name: str, conf: MCPServerConfig, timeout: float = 30
    ) -> dict[str, str]:
        """用临时连接测试服务配置，返回服务提供的工具

        参数:
            name: 服务名，决定工具名前缀
            conf: 服务配置
            timeout: 超时秒数

        返回:
            dict[str, str]: 工具名 -> 描述
        """
        from .toolkit import MCPToolkit

        toolkit = MCPToolkit(
            server_name=f"{name}_test",
            prefix=self.prefix(name),
            transport=conf.transport,
            command=conf.command,
            args=conf.args,
            url=conf.url,
            headers=conf.headers,
            env=conf.env,
            cwd=conf.cwd,
            install_command=conf.install_command,
            timeout=conf.timeout,
        )
        try:
            tools = await asyncio.wait_for(toolkit.get_tools(), timeout)
        finally:
            await toolkit.close()
        self._remember(name, cast(dict[str, ToolExecutable], tools))
        return self._known_tools[name]

    @staticmethod
    def prefix(name: str) -> str:
        """服务的工具名前缀"""
        return f"mcp_{re.sub(r'[^a-zA-Z0-9]', '_', name)}_"

    def register_server(
        self, name: str, config: MCPServerConfig, persist: bool = False
    ) -> None:
        """代码优先：动态注册 MCP 服务 (支持无痕内存级注册与热重载)"""
        self._code_registered_servers[name] = (config, persist)

        if self._config is not None:
            if persist:
                if name not in self._config.mcpServers:
                    self._config.mcpServers[name] = config
                    self._save_config()
                else:
                    json_conf = self._config.mcpServers[name]
                    code_dict = model_dump(config, exclude_unset=True)
                    json_dict = model_dump(json_conf, exclude_unset=True)
                    merged_dict = {**code_dict, **json_dict}
                    merged_conf = model_validate(MCPServerConfig, merged_dict)

                    if model_dump(merged_conf) != model_dump(json_conf):
                        self._config.mcpServers[name] = merged_conf
                        self._save_config()

            if name not in self._toolkits:
                self._setup_toolkit(name, config)
            logger.info(
                f"🔄 MCP 代理 '{name}' 已完成动态热重载注册 (持久化: {persist})"
            )

    async def unregister_server(self, name: str) -> None:
        """动态注销 MCP 服务，关闭底层进程并从配置文件中移除"""
        if name in self._code_registered_servers:
            self._code_registered_servers.pop(name)
        if self._config and name in self._config.mcpServers:
            self._config.mcpServers.pop(name)
            self._save_config()
        if tk := self._toolkits.pop(name, None):
            await tk.close()
        self._known_tools.pop(name, None)
        self._discovered_tools = None
        logger.debug(f"🛑 MCP 代理 '{name}' 已注销并销毁所有连接")

    def _save_config(self) -> None:
        """将当前配置持久化到 JSON 文件"""
        if not self._config:
            return
        try:
            with MCP_PATH.open("w", encoding="utf-8") as f:
                data = model_dump(self._config, exclude_unset=True)
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存 MCP 配置失败: {e}", e=e)

    async def initialize(self) -> None:
        if self._config is not None:
            return

        if not MCP_PATH.exists():
            MCP_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._config = MCPToolsConfig(
                mcpServers={
                    "amap-maps": MCPServerConfig(
                        command="npx",
                        args=["-y", "@amap/amap-maps-mcp-server"],
                        env={"AMAP_MAPS_API_KEY": "Your_API_Key"},
                        enabled=False,
                    ),
                    "bilibili-search": MCPServerConfig(
                        command="npx",
                        args=["bilibili-mcp"],
                    ),
                    "fetch": MCPServerConfig(
                        command="uvx",
                        args=["mcp-server-fetch"],
                    ),
                    "bingcn": MCPServerConfig(
                        command="npx",
                        args=["bing-cn-mcp"],
                    ),
                }
            )
            with MCP_PATH.open("w", encoding="utf-8") as f:
                data = model_dump(self._config, exclude_unset=True)
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            try:
                with MCP_PATH.open("r", encoding="utf-8") as f:
                    self._config = model_validate(MCPToolsConfig, json.load(f))
            except Exception as e:
                logger.error(f"加载 MCP 工具配置文件失败: {e}", e=e)
                self._config = MCPToolsConfig()

        config_changed = False
        for name, (code_conf, persist) in self._code_registered_servers.items():
            if persist:
                if name not in self._config.mcpServers:
                    self._config.mcpServers[name] = code_conf
                    config_changed = True
                else:
                    json_conf = self._config.mcpServers[name]
                    code_dict = model_dump(code_conf, exclude_unset=True)
                    json_dict = model_dump(json_conf, exclude_unset=True)
                    merged_dict = {**code_dict, **json_dict}
                    merged_conf = model_validate(MCPServerConfig, merged_dict)

                    if model_dump(merged_conf) != model_dump(json_conf):
                        self._config.mcpServers[name] = merged_conf
                        config_changed = True

        if config_changed or not MCP_PATH.exists():
            self._save_config()

        for name, conf in self._config.mcpServers.items():
            self._setup_toolkit(name, conf)

        for name, (code_conf, persist) in self._code_registered_servers.items():
            if not persist and name not in self._toolkits:
                self._setup_toolkit(name, code_conf)

    def _setup_toolkit(self, name: str, conf: MCPServerConfig) -> None:
        """辅助方法：装载单个 MCP Toolkit"""
        from .toolkit import MCPToolkit

        self._toolkits[name] = MCPToolkit(
            server_name=name,
            prefix=self.prefix(name),
            transport=conf.transport,
            command=conf.command,
            args=conf.args,
            url=conf.url,
            headers=conf.headers,
            env=conf.env,
            cwd=conf.cwd,
            install_command=conf.install_command,
            timeout=conf.timeout,
            admin_level=conf.admin_level,
            header_provider=self.header_provider,
            env_provider=self.env_provider,
        )

    async def shutdown(self):
        for tk in self._toolkits.values():
            await tk.close()
        self._toolkits.clear()

    async def get_tools_for_server(self, server_name: str) -> dict[str, ToolExecutable]:
        return await self.discover_tools(allowed_servers=[server_name])

    async def discover_tools(
        self,
        allowed_servers: list[str] | None = None,
        excluded_servers: list[str] | None = None,
    ) -> dict[str, ToolExecutable]:
        async with self._discovery_lock:
            if not self._config:
                await self.initialize()

            if not self._config:
                return {}

            if (
                self._discovered_tools is not None
                and allowed_servers is None
                and excluded_servers is None
            ):
                return self._discovered_tools

            servers_to_query: list[str] = []
            for name, conf in self._config.mcpServers.items():
                if allowed_servers and name not in allowed_servers:
                    continue
                if excluded_servers and name in excluded_servers:
                    continue
                if not conf.enabled and not allowed_servers:
                    continue
                servers_to_query.append(name)

            for name, (conf, persist) in self._code_registered_servers.items():
                if persist:
                    continue
                if allowed_servers and name not in allowed_servers:
                    continue
                if excluded_servers and name in excluded_servers:
                    continue
                if not conf.enabled and not allowed_servers:
                    continue
                if name not in servers_to_query:
                    servers_to_query.append(name)

            all_tools: dict[str, ToolExecutable] = {}
            names = [s for s in servers_to_query if s in self._toolkits]
            results = await asyncio.gather(
                *(self._toolkits[s].get_tools() for s in names),
                return_exceptions=True,
            )
            for s_name, res in zip(names, results):
                if isinstance(res, BaseException):
                    logger.error(f"MCP 服务 {s_name} 发现工具失败: {res}")
                    continue
                tools = cast(dict[str, ToolExecutable], res)
                self._remember(s_name, tools)
                all_tools.update(
                    {f"{s_name}:{name}": tool for name, tool in tools.items()}
                )

            if allowed_servers is None and excluded_servers is None:
                self._discovered_tools = all_tools
            return all_tools


class MCPSource(BaseModel):
    """显式定义的 MCP 工具源 (动态解析器)"""

    server_name: str | None = None
    """目标 MCP 服务名（走全局注册表模式时使用）"""
    namespace: str | None = None
    """可选命名空间标识（用于上层路由或分组）"""
    config: MCPServerConfig | None = None
    """内联临时配置 (MCPServerConfig 实例)"""

    fetch_all_enabled: bool = Field(default=False)
    """内部标识：是否拉取所有已启用的 MCP 服务"""
    exclude_servers: list[str] | None = Field(default=None)
    """排除的服务名列表（仅当 fetch_all_enabled 为 True 时生效）"""

    def __hash__(self):
        return hash((self.server_name, self.namespace))

    @classmethod
    def all_enabled(cls, exclude: list[str] | None = None) -> "MCPSource":
        """
        声明式获取所有已启用的 MCP 服务器下的工具。

        参数:
            exclude: 需要显式排除的 MCP 服务名称列表。
        """
        return cls(fetch_all_enabled=True, exclude_servers=exclude)

    @model_validator(mode="after")
    def validate_source(self) -> "MCPSource":
        if not self.server_name and not self.config and not self.fetch_all_enabled:
            raise ValueError(
                "MCPSource 必须提供 server_name 或 config，或者开启 fetch_all_enabled"
            )
        return self

    async def resolve(self, context: Any | None = None) -> Any:
        tools = []
        if self.fetch_all_enabled:
            server_tools_dict = await mcp_provider.discover_tools(
                excluded_servers=self.exclude_servers
            )
            for t in server_tools_dict.values():
                p = await cast(ToolResolvable, t).resolve(context)
                if p:
                    tools.extend(p.tools)
            return ResolvedToolPayload(tools=tools)

        elif self.config:
            import uuid

            from .toolkit import MCPToolkit

            temp_server_name = self.server_name or f"inline_mcp_{uuid.uuid4().hex[:8]}"
            inline_toolkit = MCPToolkit(
                server_name=temp_server_name,
                transport=self.config.transport,
                command=self.config.command,
                args=self.config.args,
                url=self.config.url,
                headers=self.config.headers,
                env=self.config.env,
                cwd=self.config.cwd,
                install_command=self.config.install_command,
                timeout=self.config.timeout,
                admin_level=self.config.admin_level,
            )
            return await inline_toolkit.resolve(context)

        else:
            assert self.server_name is not None
            server_tools = await mcp_provider.get_tools_for_server(self.server_name)
            for t in server_tools.values():
                p = await cast(ToolResolvable, t).resolve(context)
                if p:
                    tools.extend(p.tools)

            return ResolvedToolPayload(tools=tools)


mcp_provider = GlobalMCPProvider()

__all__ = ["GlobalMCPProvider", "MCPSource", "mcp_provider"]
