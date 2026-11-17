from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses
import inspect
from typing import Any, Generic, cast, get_origin
from typing_extensions import TypeVar
import uuid

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher, current_bot, current_event, current_matcher
from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.capabilities.base import AbstractCapability
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.core.protocols.tool import ToolExecutable
from zhenxun.services.ai.core.stream_events import AgentStreamEvent, EventBus
from zhenxun.services.ai.utils.runtime import ContextUtils
from zhenxun.utils.utils import infer_plugin_namespace

AgentDepsT = TypeVar("AgentDepsT", default=Any)
"""泛型类型变量：外部环境依赖对象 (Agent Dependencies)。"""


@dataclasses.dataclass(frozen=True)
class CallerIdentity:
    """由入口绑定的调用者与目标，不接收模型提供的身份字段。"""

    source: str
    platform: str
    bot_id: str
    user_id: str | None
    group_id: str | None


class NoneBotDeps(BaseModel):
    """NoneBot 环境下的标准依赖容器。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: Bot | None = Field(default=None)
    """当前触发事件的 Bot 实例"""

    event: Event | None = Field(default=None)
    """当前触发的事件实例"""

    matcher: Matcher | None = Field(default=None)
    """当前处理该事件的 Matcher 实例"""

    @classmethod
    def get_current(cls) -> "NoneBotDeps | None":
        """利用 NoneBot 原生魔法，基于 ContextVars 隐式提取当前执行上下文"""
        bot = current_bot.get(None)
        event = current_event.get(None)
        matcher = current_matcher.get(None)

        if bot or event:
            return cls(bot=bot, event=event, matcher=matcher)

        return None


class ScheduledDeps(BaseModel):
    """后台/定时任务环境下的标准依赖容器。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: Bot | None = Field(default=None)
    """机器人实例"""
    group_id: str | None = Field(default=None)
    """群组/频道 ID"""
    user_id: str | None = Field(default=None)
    """用户 ID"""
    schedule_id: int | str | None = Field(default=None)
    """定时任务/调度任务 ID"""
    platform: str | None = Field(default=None)
    """适配器平台标识"""


@dataclasses.dataclass
class SessionContext(Generic[AgentDepsT]):
    """
    生命周期：Session（会话）层。
    跨越多次对话轮次，负责保存长线状态与物理隔离信息。
    """

    session_id: str
    """核心会话标识符，用于区分不同用户或群组的上下文隔离。"""
    deps: AgentDepsT
    """强类型的外部依赖注入对象（如 Bot, Event），供跨工具共享。"""
    shared_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    """共享状态字典：全局引用穿透，用于主智能体与嵌套子智能体之间的数据通信。"""
    namespace: str = "global"
    """触发事件的插件命名空间"""


@dataclasses.dataclass
class AgentRunContext(Generic[AgentDepsT]):
    """
    生命周期：Run（运行）层。
    伴随 Agent 的单次执行 (run_stream)，保存大模型推理时的状态与原生消息历史。
    """

    session: SessionContext[AgentDepsT]
    """指向所属 Session 层上下文的引用。"""
    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    """状态字典：用于在当前 Agent 执行轮次、工具和中间件中透传动态变量。"""
    agent_name: str | None = None
    """当前正在执行的 Agent 名称。"""
    current_model: str | None = None
    """当前实际调用的底层大模型名称 (Provider/Model)。"""
    user_input: str | None = None
    """当前轮次用户的原始文本输入。"""
    messages: list[LLMMessage] = dataclasses.field(default_factory=list)
    """大模型原生上下文 (LLMMessage 列表)，与执行器中的执行历史保持内存引用同步。"""
    tool_retries: dict[str, int] = dataclasses.field(default_factory=dict)
    """记录当前轮次内各个工具的累积失败重试次数，用于系统熔断。"""
    tool_max_retries: int = 1
    """工具未单独配置 max_retries 时允许的连续出错次数（取自 reflexion_retries）。"""
    cancellation_token: CancellationToken | None = None
    """全局级联取消令牌，用于跨 Agent 的协程挂起中断。"""
    event_bus: EventBus | None = None
    """底层的事件流发射器 (EventBus)，由执行引擎在运行时挂载。"""
    source: str | None = None
    """运行来源，由入口设置；设置后中止原因由入口自行提示。"""
    follow_up: bool = True
    """是否接受同一发送者在工具执行期间追加的输入。"""
    timeout: float | None = None
    """整轮时限（秒），从进入会话锁开始计算。"""
    budget: dict[str, float] = dataclasses.field(default_factory=dict)
    """整轮预算：tokens 为已知模型用量，deadline 为截止时间；子代理共享同一对象。"""
    history_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    """写入短期记忆时附加到本轮消息上的元数据。"""
    user_notice: str | None = None
    """可直接展示给用户的失败原因。"""
    caller: CallerIdentity | None = None
    management_check: Any = None
    tool_ceiling: frozenset[str] | None = None
    available_tools: frozenset[str] = frozenset()
    bound_tools: dict[str, Any] = dataclasses.field(default_factory=dict)
    delegated_agent_tools: list[Any] | None = None
    operations: dict[str, Any] = dataclasses.field(default_factory=dict)
    """根运行共享的副作用结果、互斥与诊断，不跨对话轮次复用。"""
    root_session_id: str | None = None
    root_revision: int | None = None
    run_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    parent_run_id: str | None = None
    phase: str = "queued"
    outcome: str | None = None
    fail_fast: bool = False
    connection_id: str | None = None
    on_accept: Any = None
    prepare_follow_up: Any = None
    """由当前运行在追加输入消费边界准备背景，不重新执行整套检索。"""

    dynamic_prompts: dict[str, str] = dataclasses.field(default_factory=dict)
    """动态提示词字典（保持插入顺序并去重）。
    仅在 HTTP 请求前 JIT 渲染，不会污染持久化的上下文对话历史。"""

    def add_system_prompt(self, prompt: str, key: str | None = None) -> None:
        """动态追加临时系统提示词到大模型上下文中（实时生效且不污染历史）。"""
        dict_key = key or prompt
        if prompt:
            self.dynamic_prompts[dict_key] = prompt

    async def emit(self, event: AgentStreamEvent) -> None:
        """安全地向事件总线发射流式事件（内置判空逻辑，消解业务层的样板代码）"""
        if self.event_bus:
            await self.event_bus.emit(event)


@dataclasses.dataclass
class ToolCallContext(Generic[AgentDepsT]):
    """
    生命周期：Call（工具调用）层。
    单次工具调用分配的绝对私有状态，彻底消灭并发调用时的属性污染。
    """

    run: AgentRunContext[AgentDepsT]
    """指向所属 Run 层上下文的引用。"""
    tool_call_id: str = "unknown"
    """大模型为本次工具调用分配的唯一 ID。"""
    tool_name: str = "unknown"
    """本次调用的工具名称。"""
    retry_count: int = 0
    """当前工具调用的重试序号 (第几次重试)。"""
    current_tool: ToolExecutable | None = None
    """当前工具的可执行实例 (ToolExecutable)。"""


@dataclasses.dataclass
class RunContext(Generic[AgentDepsT]):
    """
    依赖注入容器（DI Container），保留原有上下文信息的同时提升获取类型的能力。
    """

    session_id: str | None = None
    """当前运行所在的会话ID，用于区分不同用户的独立上下文"""

    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    """状态字典：用于在会话轮次、工具和中间件中透传动态变量"""

    shared_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    """共享状态字典：全局引用穿透，用于主智能体与嵌套子智能体之间的数据通信"""

    capabilities: list[AbstractCapability] = dataclasses.field(default_factory=list)
    """当前上下文绑定的拦截器 (Capabilities) 链，用于生命周期拦截"""

    deps: AgentDepsT = dataclasses.field(default=cast(AgentDepsT, None))
    """强类型的外部依赖注入对象，用于跨工具共享业务状态或配置"""

    session: SessionContext[AgentDepsT] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    """会话层上下文：承载跨轮次共享的依赖与共享状态引用。"""
    run: AgentRunContext[AgentDepsT] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    """运行层上下文：承载当前 Agent 执行轮次的模型状态与运行时元信息。"""
    call: ToolCallContext[AgentDepsT] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    """调用层上下文：承载单次工具调用的私有状态与执行引用。"""

    def get_bot(self) -> Bot | None:
        """强类型安全地提取 Bot 实例"""
        if not self.deps:
            return None
        bot = getattr(self.deps, "bot", None)
        return bot if isinstance(bot, Bot) else None

    def get_event(self) -> Event | None:
        """强类型安全地提取 Event 实例"""
        if not self.deps:
            return None
        event = getattr(self.deps, "event", None)
        return event if isinstance(event, Event) else None

    def get_user_id(self) -> str | None:
        """安全提取当前触发任务的用户 ID"""
        if (run := getattr(self, "run", None)) and run.caller:
            return run.caller.user_id
        return ContextUtils.extract_user_id(self.deps)

    def get_group_id(self) -> str | None:
        """安全提取当前触发任务的群组 ID（私聊则为 None）"""
        if (run := getattr(self, "run", None)) and run.caller:
            return run.caller.group_id
        return ContextUtils.extract_group_id(self.deps)

    def get_platform(self) -> str:
        """安全提取当前连接的适配器平台标识"""
        return ContextUtils.extract_platform(self.deps)

    def __post_init__(self):
        if self.deps is None:
            self.deps = cast(AgentDepsT, NoneBotDeps.get_current())

        if not self.session_id and self.get_bot():
            self.session_id = ContextUtils.default_session_id(self.deps)

        ns = "global"
        if self.deps:
            ns = getattr(self.deps, "namespace", None) or (
                self.deps.get("namespace") if isinstance(self.deps, dict) else None
            )
        if not ns:
            ns = infer_plugin_namespace()

        self.session = SessionContext(
            session_id=self.session_id or "default_session",
            deps=self.deps,
            shared_state=self.shared_state,
            namespace=ns,
        )

        self.run = AgentRunContext(session=self.session, state=self.state)
        self.call = ToolCallContext(run=self.run)

    def clone_for_tool_call(self, tool_call_id: str, tool_name: str) -> "RunContext":
        """
        为每个并发的工具调用派生绝对独立的 ToolCallContext。消除多工具并行时的属性污染。
        """
        new_ctx = dataclasses.replace(cast(Any, self))
        new_ctx.session = self.session
        new_ctx.run = self.run
        new_ctx.call = ToolCallContext(
            run=new_ctx.run,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            retry_count=new_ctx.run.tool_retries.get(tool_name, 0),
        )
        return new_ctx


_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "current_run_context", default=None
)


def get_current_run_context() -> RunContext | None:
    """
    [全局逃生舱] 获取当前运行中的上下文对象。
    适用于深层嵌套业务逻辑，无需层层透传 context 参数。
    """
    return _CURRENT_RUN_CONTEXT.get()


def fail_fast_enabled() -> bool:
    """仅当前 AI 运行及继承其上下文的调用采用首次失败即停。"""
    context = get_current_run_context()
    return bool(context and context.run.fail_fast)


def interrupt_network(stage: str, error: BaseException | None = None) -> None:
    """中断根运行并取消兄弟执行，不把通信失败降级成可重试工具结果。"""
    from zhenxun.services.ai.core.exceptions import RunInterrupted
    from zhenxun.services.ai.run.session import session_manager

    context = get_current_run_context()
    if context is None or not context.run.fail_fast:
        return
    context.run.operations["interruption"] = stage
    context.run.outcome = "interrupted"
    session_manager.interrupt_siblings(context, "network")
    raise RunInterrupted(stage) from error


def interrupt_transport(error: BaseException, stage: str) -> None:
    """按异常类型识别传输失败；完整业务拒绝仍交给工具正常返回。"""
    import asyncio
    import ssl

    import anyio
    import httpx

    pending = [error]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(
            item,
            httpx.TransportError
            | asyncio.TimeoutError
            | ConnectionError
            | ssl.SSLError
            | anyio.EndOfStream
            | anyio.BrokenResourceError
            | anyio.ClosedResourceError,
        ):
            interrupt_network(stage, error)
        if item.__cause__:
            pending.append(item.__cause__)
        pending.extend(getattr(item, "exceptions", ()))


@contextmanager
def set_run_context(ctx: RunContext):
    """[内部 API] 挂载当前上下文至全局"""
    from zhenxun.utils.decorator.retry import network_fail_fast

    token = _CURRENT_RUN_CONTEXT.set(ctx)
    network_token = network_fail_fast.set(bool(ctx and ctx.run.fail_fast))
    try:
        yield
    finally:
        network_fail_fast.reset(network_token)
        _CURRENT_RUN_CONTEXT.reset(token)


def _is_run_context_type(annotation: Any) -> bool:
    """判断类型标注是否为 RunContext 类型或其子类。"""
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    if inspect.isclass(origin) and issubclass(origin, RunContext):
        return True
    if inspect.isclass(annotation) and issubclass(annotation, RunContext):
        return True
    if "RunContext" in str(annotation):
        return True
    return False


__all__ = [
    "AgentDepsT",
    "AgentRunContext",
    "NoneBotDeps",
    "RunContext",
    "ScheduledDeps",
    "SessionContext",
    "ToolCallContext",
    "get_current_run_context",
    "set_run_context",
]
