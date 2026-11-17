from collections.abc import Callable
import copy
from typing import Any, Literal

from pydantic import ValidationError

from zhenxun.services.ai.capabilities.base import CombinedCapability
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
    ToolFatalError,
    ToolRetryError,
)
from zhenxun.services.ai.core.models import ToolDefinition
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.models import (
    ResolvedToolPayload,
    ToolOptions,
    ToolResult,
)
from zhenxun.services.ai.utils.logger import log_tool as logger
from zhenxun.services.ai.utils.utils import wrap_to_async
from zhenxun.utils.pydantic_compat import model_dump, model_validate

from .schema import _parse_docstring, build_schema_hint, build_tool_model


class BaseTool:
    """
    面向对象的工具基类。支持多模态契约与纯 Pydantic V2 参数验证。

    参数:
        name: 工具名称。大模型将基于此识别并调用。如果不提供，默认使用类名。
        description: 工具的说明文档。大模型会根据此描述理解工具用途。
        settings: 配置选项（权限拦截器与元数据）。
    """

    name: str
    description: str
    settings: ToolOptions
    execution_side: Literal["client", "server"] = "client"
    _dynamic_def: Any = None
    _base_schema: dict[str, Any] | None = None
    _param_model: Any = None

    @property
    def identity(self) -> str:
        """工具的稳定来源标识，不随本轮参数或权限变化。"""
        target = self.get_signature_target()
        return (
            f"local:{getattr(target, '__module__', type(target).__module__)}:"
            f"{getattr(target, '__qualname__', type(target).__qualname__)}:"
            f"{self.name.lower()}"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self.settings.metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self.settings.metadata = value

    def get_execute_target(self) -> Callable:
        """获取依赖注入和实际执行的目标函数。子类可重写（如 FunctionTool）。"""
        return self.run

    def get_signature_target(self) -> Callable:
        """获取依赖注入和 Schema 生成的原始函数目标。子类可重写。"""
        return self.get_execute_target()

    async def run(self, **kwargs: Any) -> Any:
        """供第三方开发者重写的核心执行逻辑。完美支持 Inject 依赖注入。"""
        raise NotImplementedError("子类必须实现 run 方法")

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        settings: ToolOptions | None = None,
    ):
        """
        初始化工具基类。

        参数:
            name: 工具名，供大模型识别调用，默认取类名。
            description: 工具功能说明，供大模型理解，默认取 __doc__ 说明。
            settings: 工具的配置选项（权限拦截器与元数据），默认 None。
        """
        self.name = name or getattr(self, "name", self.__class__.__name__)
        self.description = description or getattr(
            self, "description", self.__doc__ or "未提供描述"
        )
        self.settings = settings or getattr(self, "settings", ToolOptions())

    async def _handle_validation_error(self, e: ValidationError, kwargs: dict) -> None:
        """将 Pydantic 参数校验失败转化为 ToolRetryError"""
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "root"
            msg = err.get("msg", "")
            ctx = err.get("ctx", {})
            ctx_str = f" (期望要求: {ctx})" if ctx else ""
            error_msgs.append(f"参数字段 `{loc}`: {msg}{ctx_str}")

        err_msg = "\n".join(error_msgs)
        schema_hint = build_schema_hint(self._param_model)
        raise ToolRetryError(
            f"参数验证未通过，请严格根据以下错误信息修正参数后重试：\n{err_msg}\n"
            f"（附：你刚才传入的非法参数为 {kwargs}）{schema_hint}"
        )

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        """获取工具的 JSON Schema 定义。支持动态修改 Schema 甚至隐藏工具。"""
        if self._dynamic_def is not None:
            return self._dynamic_def

        schema = (
            self._base_schema.copy()
            if isinstance(self._base_schema, dict)
            else {"type": "object", "properties": {}}
        )
        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
            metadata=self.settings.metadata.copy(),
        )

        if context and self.settings.capabilities:
            combined_cap = CombinedCapability(self.settings.capabilities)
            defs = await combined_cap.prepare_tools(context, [tool_def])
            if not defs:
                return None
            tool_def = defs[0]

        return tool_def

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        """实现 ToolResolvable 协议，将自身解析为标准 Payload"""
        definition = await self.get_definition(context)
        if definition is None:
            return ResolvedToolPayload()

        run_scoped_tool = copy.copy(self)
        run_scoped_tool._dynamic_def = definition
        return ResolvedToolPayload(tools=[run_scoped_tool])

    async def validate_args(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """验证参数，返回序列化后的合法字典。如果校验失败则抛出异常。"""
        if self._param_model:
            try:
                validated_model = model_validate(self._param_model, kwargs)
                return model_dump(validated_model, exclude_unset=True)
            except ValidationError as e:
                await self._handle_validation_error(e, kwargs)
        return kwargs

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        """工具执行的总入口：注入依赖并运行，成功时清零该工具的连续出错计数"""
        from zhenxun.services.ai.tools.engine.executor import run_native_tool

        context = context or RunContext()
        try:
            result = await run_native_tool(self, context, **kwargs)
        except Exception as e:
            if not isinstance(e, ControlFlowExit):
                logger.error(f"工具 {self.name} 执行抛出异常，将交由底层引擎处理: {e}")
            raise
        if not result.is_error:
            context.run.tool_retries[self.name] = 0
        return result


class ServerSideTool(BaseTool):
    """
    云端原生工具的抽象基类。
    代表那些由各大模型厂商在云端原生实现的能力（如 Google Search, Code Execution）。
    此类工具只产生 Schema 提交给大模型用于触发路由，禁止在本地执行。
    """

    execution_side: Literal["client", "server"] = "server"
    type_id: str = ""

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        raise ToolFatalError(
            f"[{self.name}] 是服务端工具，禁止在本地层执行引擎中运行。"
        )

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        if self._dynamic_def is not None:
            return self._dynamic_def
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}},
            metadata=self.settings.metadata.copy(),
        )


class FunctionTool(BaseTool):
    """
    将普通 Python 函数包装为 BaseTool 的实现。

    参数:
        func: 实际执行的 Python 异步或同步函数。
        name: 覆盖函数名的工具名称。
        description: 覆盖函数 docstring 的工具描述。
        settings: 配置选项。
    """

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        settings: ToolOptions | None = None,
    ):
        """
        初始化函数式包装工具。

        参数:
            func: 被包装的底层 Python 同步或异步函数。
            name: 覆盖函数名的自定义工具名，默认 None。
            description: 覆盖 docstring 的自定义工具描述，默认 None。
            settings: 覆盖默认选项的自定义工具配置项，默认 None。
        """
        super().__init__(name=name, description=description, settings=settings)

        self._original_func = func
        self.__name__ = self.name
        self._func = wrap_to_async(func)
        self._schema_built = False

        main_desc, _ = _parse_docstring(self._original_func.__doc__)
        if not description or description == "未提供描述":
            self.description = main_desc or "未提供描述"

    def get_execute_target(self) -> Callable:
        """FunctionTool 的执行目标和依赖注入目标是其包装的原始函数"""
        return self._func

    def get_signature_target(self) -> Callable:
        return self._original_func

    def _ensure_schema(self):
        """JIT 懒加载：推迟到第一次请求定义或验证参数时再构建 Schema (耗时操作)"""
        if not self._schema_built:
            self._base_schema, self._param_model = build_tool_model(
                self.get_signature_target()
            )
            self._schema_built = True

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        self._ensure_schema()
        return await super().get_definition(context)

    async def validate_args(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._ensure_schema()
        return await super().validate_args(kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """允许像普通函数一样被直接调用。对其他插件完全透明。"""
        return self._original_func(*args, **kwargs)
