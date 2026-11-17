from __future__ import annotations

"""
运行时（Run）相关核心类型定义
"""

from collections.abc import AsyncIterator
from typing import Any, Generic, cast

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.exceptions import ControlFlowExit
from zhenxun.services.ai.core.messages import LLMMessage, UsageInfo
from zhenxun.services.ai.core.messages.types import OutputDataT
from zhenxun.services.ai.core.stream_events import (
    AgentStreamEvent,
    EventBus,
)


class AgentRunResult(BaseModel, Generic[OutputDataT]):
    """Agent 单次无状态运行的结果"""

    output: OutputDataT
    """最终输出数据"""
    messages: list[LLMMessage] = Field(default_factory=list)
    """本次运行产生/更新的历史消息"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """本次运行的Token消耗总计"""
    stopped: ControlFlowExit | None = None
    outcome: str = "completed"
    reason: str | None = None
    """运行被中止的原因（审批被拒、消息已追加、任务被中止等），此时 output 为空"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentRunStart(AgentStreamEvent):
    """智能体运行开始"""

    agent_name: str
    """启动运行的智能体名称"""


class AgentRunError(AgentStreamEvent):
    """智能体运行发生异常"""

    error: BaseException
    """智能体运行过程中抛出的异常"""


class AgentRunEnd(AgentStreamEvent):
    """智能体运行完全结束"""

    result: AgentRunResult[Any]
    """智能体运行结束后的完整结果"""


class StreamedRunResult(Generic[OutputDataT]):
    """
    智能体流式运行的结果代理对象。
    可逐个消费底层事件，或等待运行结束取得完整结果。
    """

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._result: AgentRunResult[OutputDataT] | None = None

    async def stream_events(self) -> "AsyncIterator[AgentStreamEvent]":
        """获取底层的所有原始事件（包含工具调用过程等）"""

        async for event in self._event_bus:
            if isinstance(event, AgentRunEnd):
                self._result = cast(AgentRunResult[OutputDataT], event.result)
            elif isinstance(event, AgentRunError):
                raise event.error.with_traceback(None) from None
            yield event

    async def get_run_result(self) -> AgentRunResult[OutputDataT]:
        """等待运行结束，返回完整的运行结果（包含 Token 消耗等）"""
        if self._result is None:
            async for _ in self.stream_events():
                pass
        if self._result is None:
            raise RuntimeError("Agent 运行异常结束，未产生最终结果。")
        return self._result


__all__ = [
    "AgentRunResult",
    "OutputDataT",
    "StreamedRunResult",
]
