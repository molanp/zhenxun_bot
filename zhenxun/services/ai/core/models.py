"""
模型自身设定域类型定义
"""

import asyncio
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

TReq = TypeVar("TReq", bound=BaseModel)
"""泛型：LLM 请求体模型约束 (必须继承自 BaseModel)"""
TRes = TypeVar("TRes", bound=BaseModel)
"""泛型：LLM 响应体模型约束 (必须继承自 BaseModel)"""


class CancellationToken:
    """全局取消令牌，用于在异步链路中传递中止信号"""

    def __init__(self, parent: "CancellationToken | None" = None):
        self._cancelled = False
        self._futures: list[asyncio.Future] = []
        self.parent = parent
        self.reason = ""
        self._children: set[CancellationToken] = set()
        if parent is not None:
            parent._children.add(self)
            if parent.is_cancelled():
                self.cancel(parent.reason)

    def cancel(self, reason: str = "cancelled") -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self.reason = reason
        for child in tuple(self._children):
            child.cancel(reason)
        for f in tuple(self._futures):
            if not f.done():
                f.cancel()

    def detach(self) -> None:
        """子运行完成后释放父级关联。"""
        if self.parent is not None:
            self.parent._children.discard(self)
        self._futures.clear()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError(
                "任务已被主动取消 (CancellationToken triggered)"
            )

    def link_future(self, future: asyncio.Future) -> None:
        if self._cancelled:
            future.cancel()
        elif future not in self._futures:
            self._futures.append(future)
            future.add_done_callback(self._unlink)

    def _unlink(self, future: asyncio.Future) -> None:
        if future in self._futures:
            self._futures.remove(future)


class LLMContext(BaseModel, Generic[TReq, TRes]):
    """LLM 执行上下文，用于在能力包装链中传递请求状态"""

    request: TReq
    """强类型的各模态请求对象实体。"""

    cancellation_token: CancellationToken | None = Field(default=None)
    """全局取消令牌。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolDefinition(BaseModel):
    """结构化的工具定义模型"""

    name: str = Field(...)
    """工具名称"""
    description: str = Field(...)
    """工具描述"""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema 参数"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """元数据"""


class ToolChoice(BaseModel):
    """工具选择配置"""

    mode: Literal["auto", "none", "any", "required"] = Field(default="auto")
    """工具选择模式"""
    allowed_function_names: list[str] | None = Field(default=None)
    """允许调用的函数名称列表"""


class ModelModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
"""统一的推理强度等级，从低到高排列"""


class ModelProfile(BaseModel):
    """模型能力档案，决定请求如何编码以及哪些任务可用"""

    tasks: set[str] = Field(default_factory=lambda: {"chat"})
    """可承担的任务：chat / embedding / rerank / tts / stt / image"""
    input_modalities: set[ModelModality] = Field(
        default_factory=lambda: {ModelModality.TEXT}
    )
    """支持的输入模态"""
    output_modalities: set[ModelModality] = Field(
        default_factory=lambda: {ModelModality.TEXT}
    )
    """支持的输出模态"""
    tools: bool = True
    """是否支持函数调用"""
    structured_output: Literal["json_schema", "json_object", "none"] = "json_object"
    """原生结构化输出方式"""
    reasoning_levels: list[str] = Field(default_factory=list)
    """可选的推理强度等级，为空表示只能开关或不支持调节"""
    reasoning_off: bool = False
    """是否允许关闭推理"""
    reasoning_param: str = "none"
    """推理参数的写法，取值由协议决定 (如 openai / thinking / level / adaptive)"""
    context_window: int = Field(default=128000, gt=0)
    """上下文窗口 Token 数"""
    max_output_tokens: int | None = Field(default=None, gt=0)
    """单次最大输出 Token 数"""
    native_tools: set[str] = Field(default_factory=set)
    """确认支持的服务端内置工具 (web_search / code_execution / url_context 等)"""
    native_tools_off: set[str] = Field(default_factory=set)
    """确认不支持的服务端内置工具，未列入两者的内置工具按请求照常发送"""
    mixed_tools: bool = False
    """是否允许内置工具与函数工具混用"""
    default_voice: str | None = None
    """语音合成的默认音色"""
    sampling: bool = True
    """是否接受 temperature / top_p 等采样参数"""
    sources: dict[str, str] = Field(default_factory=dict)
    """各字段的取值来源：user / observed / discovery / catalog / hint / default，
    内置工具按 native_tools.<类型> 记录"""

    def supports_task(self, task: str) -> bool:
        """判断模型是否可承担指定任务"""
        return task in self.tasks

    def accepts_input(self, modality: ModelModality | str) -> bool:
        """判断模型是否接受某种输入模态"""
        return ModelModality(modality) in self.input_modalities


__all__ = [
    "REASONING_LEVELS",
    "CancellationToken",
    "LLMContext",
    "ModelModality",
    "ModelProfile",
    "ToolChoice",
    "ToolDefinition",
]
