import inspect
from typing import Any, ClassVar

from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.models import (
    ResolvedToolPayload,
    ToolkitConfig,
    ToolOptions,
)
from zhenxun.utils.pydantic_compat import model_dump

from .tool import BaseTool


class BaseToolkit:
    """
    状态工具箱基类。继承此类的对象可以维持自身状态，
    其内部被 @tool 标记的方法将被自动解析为工具。
    """

    default_instructions: str = ""
    config: ToolkitConfig
    _default_config: ClassVar[ToolkitConfig] = ToolkitConfig()

    @property
    def name(self) -> str:
        """工具箱的默认名称标识（取类名），主要用于全局注册表的 Hash 与展示"""
        return self.__class__.__name__

    def __init__(
        self,
        prefix: str | None = None,
        shared_options: ToolOptions | None = None,
        config: ToolkitConfig | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ):
        """
        初始化工具箱实例。

        参数:
            prefix: 工具名称的前缀。如果不指定，则不添加前缀。
            shared_options: 所有子工具默认继承的配置项，覆盖类上的默认配置。
            config: 显式传递的配置对象，与类上的默认配置合并。
            instructions: 当前工具箱实例的系统提示词补充说明。
        """
        merged = model_dump(self._default_config, exclude_unset=True)
        if config is not None:
            merged.update(model_dump(config, exclude_unset=True))
        if prefix is not None:
            merged["prefix"] = prefix
        if shared_options is not None:
            merged["shared_options"] = shared_options
        self.config = ToolkitConfig(**merged)
        if self.config.prefix is None:
            self.config.prefix = ""

        self._cached_tools: dict[str, BaseTool] | None = None
        self._instance_instructions = (
            instructions
            if instructions is not None
            else getattr(self, "default_instructions", "")
        )

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        """实现 ToolResolvable 协议，解析内部工具并返回标准 Payload"""
        resolved_tools = []
        for t in (await self.get_tools(context)).values():
            payload = await t.resolve(context)
            resolved_tools.extend(payload.tools)

        builders = []
        if instructions := self.get_instructions():
            required = frozenset(t.name for t in resolved_tools)
            builders.append(
                lambda names: [instructions] if required and required <= names else []
            )

        return ResolvedToolPayload(tools=resolved_tools, instruction_builders=builders)

    async def get_tools(self, context: RunContext | None = None) -> dict[str, BaseTool]:
        """获取该工具箱中所有合法注册的工具实例映射表"""
        if self._cached_tools is not None:
            return self._cached_tools

        tools_dict: dict[str, BaseTool] = {}
        for name, member in inspect.getmembers(self.__class__):
            if hasattr(member, "__toolkit_tool__"):
                bound_tool = getattr(self, name)
                original_name = getattr(member, "__tool_original_name__", name)
                final_tool_name = f"{self.config.prefix}{original_name}"
                bound_tool.name = final_tool_name
                if self.config.shared_options:
                    bound_tool.settings = self.config.shared_options.merge(
                        bound_tool.settings
                    )
                tools_dict[final_tool_name] = bound_tool

        self._cached_tools = tools_dict
        return tools_dict

    def get_instructions(self) -> str | None:
        """
        获取工具箱的系统提示词说明。
        """
        text = self._instance_instructions
        return text or None
