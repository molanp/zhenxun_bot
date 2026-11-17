"""
记忆域类型定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import LLMMessage


@dataclass(frozen=True)
class HistorySettlement:
    """历史后端确认的本轮可见输入，不以未抛异常代替收据。"""

    status: str = "unconfirmed"
    visible_claims: frozenset[str] = field(default_factory=frozenset)


class BaseChatContext(ABC):
    """短期对话历史记忆接口"""

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[LLMMessage]:
        """获取会话的历史消息"""
        ...

    @abstractmethod
    async def add_messages(self, session_id: str, messages: list[LLMMessage]) -> None:
        """向会话追加消息"""
        ...

    async def settle_run(self, session_id, run_id, revision, delivery, started_at):
        """结算支持投递状态的聊天历史。"""
        raise NotImplementedError("当前历史后端不支持投递结算")

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """清空会话的历史消息"""
        ...


class BaseMemoryReducer(ABC):
    """记忆压缩器基类"""

    @abstractmethod
    async def reduce(
        self,
        messages: list[LLMMessage],
        current_tokens: int,
        model_name: str,
        base_overhead: int = 0,
    ) -> tuple[list[LLMMessage], bool, int]:
        """
        执行记忆压缩处理，精简或提炼对话上下文以降低 Token 消耗。

        参数:
            messages: 需要进行压缩的原始 LLM 消息历史列表。
            current_tokens: 压缩前消息列表的当前 Token 总数。
            model_name: 用于判定压缩阈值或计算 Token 的底层大模型名称。
            base_overhead: 基础系统提示词等静态开销的 Token 计数。

        返回:
            tuple[list[LLMMessage], bool, int]: 包含压缩后的新消息历史列表、
                本次是否实际触发了压缩的布尔标记、以及压缩后的新 Token 总数。
        """
        ...


class ShortTermConfig(BaseModel):
    """短期对话记忆配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable: bool = Field(default=True)
    """是否启用短期对话记忆上下文"""
    backend: BaseChatContext | None = Field(default=None)
    """短期记忆存储后端，为空使用数据库后端"""


class ContextCompressionConfig(BaseModel):
    """上下文压缩与管理配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float | None = Field(default=None)
    """(局部重写) 触发记忆压缩的 Token 阈值"""
    max_history_turns: int | None = Field(default=None)
    """(局部重写) 触发记忆压缩的对话轮数上限。设为 0 表示不限制轮数。"""
    vision_window: int | None = Field(default=None)
    """多模态滑动窗口大小。0表示关闭该功能，>0表示仅保留最近N轮包含多模态数据的消息，None表示跟随全局配置。"""
    policy: list[BaseMemoryReducer] | None = Field(default=None)
    """核心记忆压缩策略管线 (List[BaseMemoryReducer])。为 None 时将应用全局默认策略。"""


class MemoryConfig(BaseModel):
    """统一的记忆配置项声明"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    """短期对话记忆配置"""
    compression: ContextCompressionConfig = Field(
        default_factory=ContextCompressionConfig
    )
    """上下文压缩与管理配置"""

    @classmethod
    def resolve(cls, memory: "bool | MemoryConfig | None") -> "MemoryConfig":
        """把 Agent 的 memory 参数转换为记忆配置

        参数:
            memory: 布尔值表示是否开启短期记忆，也可以直接传入配置

        返回:
            MemoryConfig: 记忆配置
        """
        if isinstance(memory, MemoryConfig):
            return memory
        return cls(short_term=ShortTermConfig(enable=bool(memory)))
