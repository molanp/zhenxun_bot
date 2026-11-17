"""
工具系统域类型定义
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import ToolCallPart, UsageInfo


class ToolResult(BaseModel):
    """结构化的工具执行结果模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = Field(...)
    """大模型实际看到的执行结果，可以是字符串、字典等序列化对象。"""
    usage: UsageInfo | None = Field(default=None)
    """子智能体或复杂工具消耗的 Token 统计，将向主流程冒泡累加。"""
    is_error: bool = Field(default=False)
    """是否发生了业务级别的错误"""
    is_retryable: bool = Field(default=True)
    """标记该错误是否允许大模型进行自愈反思重试"""
    uncertain: bool = False
    """外部操作可能已经执行，不能按普通失败重放。"""
    reused: bool = False
    """返回本轮已经完成的操作结果，没有再次执行。"""

    def as_error(
        self, is_retryable: bool = True, *, uncertain: bool = False
    ) -> "ToolResult":
        """链式方法：标记此结果为错误，并引导大模型在下一轮进行重试自愈"""
        self.is_error = True
        self.is_retryable = is_retryable
        self.uncertain = uncertain
        return self


class ToolOptions(BaseModel):
    """工具的配置选项"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    capabilities: list[Any] = Field(default_factory=list)
    """当前工具专属的生命周期拦截器 (Capabilities) 列表。"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """额外扩展元数据字典，可供其他系统或自定义中间件读取。"""

    def merge(self, other: "ToolOptions | None") -> "ToolOptions":
        """合并另一个配置：拦截器按顺序追加，元数据由 other 覆盖

        参数:
            other: 另一个配置

        返回:
            ToolOptions: 合并后的新配置
        """
        if not other:
            return self
        return ToolOptions(
            capabilities=self.capabilities + other.capabilities,
            metadata={**self.metadata, **other.metadata},
        )


class ToolkitConfig(BaseModel):
    """工具箱配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prefix: str | None = Field(default=None)
    """工具名称前缀"""
    shared_options: ToolOptions | None = Field(default=None)
    """所有子工具默认继承的配置项"""


class ValidatedToolCall(BaseModel):
    """工具调用验证结果载体"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: ToolCallPart = Field(...)
    """原始工具调用部件"""
    tool: Any | None = Field(default=None)
    """匹配到的目标工具实例"""
    args_valid: bool = Field(default=False)
    """参数是否成功通过校验"""
    validated_args: dict[str, Any] | None = Field(default=None)
    """通过验证并反序列化后的参数字典"""
    validation_error: BaseException | None = Field(default=None)
    """验证失败时的异常信息"""


@dataclass
class ResolvedToolPayload:
    """解析后的工具上下文包"""

    tools: list[Any] = field(default_factory=list)
    injected_prompts: list[str] = field(default_factory=list)
    """不随运行变化的说明，进入系统提示词"""
    reminders: list[str] = field(default_factory=list)
    """随运行变化的提示（如检索结果），只贴在本轮用户消息末尾"""
    instruction_builders: list[Callable[[frozenset[str]], list[str]]] = field(
        default_factory=list
    )
    """根据最终可用工具生成操作说明，不持久化回调。"""
    injection_sources: list[dict[str, Any]] = field(default_factory=list)
    """来源、长度与生命周期摘要，不包含注入正文。"""

    def merge(self, other: "ResolvedToolPayload") -> "ResolvedToolPayload":
        """将另一个解析载荷合并到当前载荷中"""
        self.tools.extend(other.tools)
        self.injected_prompts.extend(other.injected_prompts)
        self.reminders.extend(other.reminders)
        self.instruction_builders.extend(other.instruction_builders)
        self.injection_sources.extend(other.injection_sources)
        return self


__all__ = [
    "ResolvedToolPayload",
    "ToolOptions",
    "ToolResult",
    "ToolkitConfig",
    "ValidatedToolCall",
]
