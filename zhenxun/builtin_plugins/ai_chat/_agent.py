import asyncio
import json
import time
from typing import Any
import uuid

from zhenxun.configs.config import Config
from zhenxun.models.ai_agent_call import AIAgentCall
from zhenxun.models.ai_agent_run import AIAgentRun
from zhenxun.services.ai.capabilities.base import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from zhenxun.services.ai.core.exceptions import ControlFlowExit, LLMException
from zhenxun.services.ai.core.messages import TextPart
from zhenxun.services.ai.llm import get_default_model
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.sandbox.manager import SandboxManager
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ResolvedToolPayload, ToolResult
from zhenxun.services.ai.utils.runtime import PermissionUtils
from zhenxun.services.log import logger

from ._data_source import MODULE, PROACTIVE_SOURCES
from ._reply import parse_silent

RUN_ID_KEY = "_agent_run_id"
"""运行上下文 state 中的运行id"""
AGENT_ACTIVE_KEY = "_agent_active"
"""入口是否请求启用或明确禁用敏感工具，实际授权仍在执行前核验。"""
AGENT_TOOLS_KEY = "_agent_tools"
"""运行上下文 state 中本次运行提供的 Agent 工具名"""


SOURCES = {
    "qq": {"privileged": True, "follow_up": True},
    "webui": {"privileged": True, "follow_up": True},
    "task": {"privileged": True, "follow_up": False},
    "interject": {"privileged": False, "follow_up": False},
}
"""聊天来源及其策略：privileged 为可用管理与持久化修改类工具，
follow_up 为接受同一发送者追加的输入"""
INTERJECT_TOOLS = {
    "web_search",
    "search_web",
    "search_knowledge",
    "recall_memory",
    "find_stickers",
    "send_sticker",
    "send_message",
}
"""自动插话可用的工具"""


def source_policy(context: RunContext) -> dict[str, bool]:
    """从入口绑定的身份读取来源，缺少身份时不提供特权能力

    参数:
        context: 运行上下文

    返回:
        dict[str, bool]: 来源策略
    """
    caller = context.run.caller
    return SOURCES.get(caller.source if caller else "", SOURCES["interject"])


def bind_source(context: RunContext, source: str, *, management_check=None) -> None:
    """为运行绑定聊天来源，设置是否接受追加输入与整轮时限，
    无特权的来源不启用 Agent

    参数:
        context: 运行上下文
        source: 来源：qq、webui、task 或 interject
    """
    if source not in SOURCES:
        raise ValueError(f"未知聊天来源: {source}")
    PermissionUtils.bind_caller(context, source, management_check)
    context.run.source = source
    context.run.fail_fast = True
    if bot := context.get_bot():
        context.run.connection_id = vars(bot).get("_zhenxun_connection_id")
    context.run.follow_up = SOURCES[source]["follow_up"]
    context.run.timeout = max(1, int(Config.get_config(MODULE, "TURN_TIMEOUT") or 300))
    if not SOURCES[source]["privileged"]:
        context.state[AGENT_ACTIVE_KEY] = False


class ReplyCapability(AbstractCapability):
    """无特权来源只提供插话工具；模型输出写入历史前先清理并过滤"""

    async def prepare_tools(self, context, tool_defs):
        if not source_policy(context)["privileged"]:
            return [t for t in tool_defs if t.name in INTERJECT_TOOLS]
        return tool_defs

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        if (
            not source_policy(context)["privileged"]
            and tool_name not in INTERJECT_TOOLS
        ):
            return ToolResult(output="自动插话不能执行管理或持久化修改。").as_error(
                False
            )
        return await handler(arguments)

    async def wrap_model_request(
        self, context, llm_context, handler: WrapModelRequestHandler
    ):
        from ._reply import clean_output
        from ._safety import Safety

        response = await handler(llm_context)
        parts = response.content_parts
        text = "".join(p.text for p in parts if isinstance(p, TextPart))
        safe, _ = Safety.filter_output(clean_output(text))
        response.content_parts = ([TextPart(text=safe)] if safe else []) + [
            p for p in parts if not isinstance(p, TextPart)
        ]
        return response


def agent_enabled() -> bool:
    """是否为超级用户开启 Agent

    返回:
        bool: 是否开启
    """
    return bool(Config.get_config(MODULE, "AGENT_ENABLED"))


USER_KEY = "_agent_user"
"""运行上下文 state 中的发起人，用于没有 NoneBot 事件的运行（如 WebUI 控制台）"""


def start_run(context: RunContext) -> str:
    """为一次运行分配运行id

    参数:
        context: 运行上下文

    返回:
        str: 运行id
    """
    run_id = context.run.run_id
    context.state[RUN_ID_KEY] = run_id
    return run_id


async def record_run(
    context: RunContext,
    source: str,
    persona: str | None,
    text: str,
    started: float,
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    """保存所有来源的生成、投递及已知用量，主动未发言时状态为 silent

    参数:
        context: 运行上下文
        source: 来源：qq、webui 或主动发言来源（interject、task）
        persona: 人设名称
        text: 输入文本
        started: 开始时间（time.monotonic）
        result: 运行结果
        error: 运行失败时的异常
    """
    run_id = context.state.get(RUN_ID_KEY)
    proactive = source in PROACTIVE_SOURCES
    if not run_id:
        return
    status, message = "ok", ""
    if error is not None:
        status = "stopped" if isinstance(error, asyncio.CancelledError) else "error"
        message = (
            context.run.user_notice or error.user_friendly_message
            if isinstance(error, LLMException)
            else f"{type(error).__name__}: {error}"
        )
    elif result is not None and result.stopped:
        status, message = "stopped", str(result.stopped)
    elif proactive and result is not None and parse_silent(str(result.output)) is None:
        status = "silent"
    from zhenxun.services.ai.run.session import session_manager

    if context.run.outcome and context.run.outcome != "completed":
        # failed 只表示投递失败，执行失败记为 error
        status = "error" if context.run.outcome == "failed" else context.run.outcome
    try:
        from ._safety import Safety

        delivery = context.state.get("_delivery", {})
        if status in {"ok", "silent"} and delivery.get("status") in {
            "failed",
            "partial",
            "unknown",
        }:
            status = delivery["status"]
        values = {
            "source": source,
            "session_id": context.session_id or "",
            "user_id": context.get_user_id() or context.state.get(USER_KEY) or "",
            "group_id": context.get_group_id(),
            "persona": persona,
            "model": context.run.current_model or get_default_model("chat"),
            "input": _shorten(text, 2000),
            "output": _shorten(
                Safety.filter_output(str(result.output) if result else "")[0], 4000
            ),
            "tools": await AIAgentCall.filter(run_id=run_id).count()
            + len(context.run.operations.get("dsh_tools", {})),
            "tokens": context.run.budget.get(
                "tokens", result.usage.total_tokens if result else 0
            ),
            "status": "settling" if run_id in session_manager.runs else status,
            "error": _shorten(Safety.filter_output(message)[0], 2000),
            "details": {
                **session_manager.details(context),
                "delivery": delivery,
                "execution_status": status,
            },
            "duration": int((time.monotonic() - started) * 1000),
        }
        _, created = await AIAgentRun.get_or_create(run_id=run_id, defaults=values)
        if not created:
            await AIAgentRun.filter(
                run_id=run_id,
                status__in=["queued", "running", "delivering", "settling", "stopping"],
            ).update(**values)
    except Exception as e:
        logger.warning(f"保存 Agent 运行记录失败: {e}", MODULE)


def _shorten(value: Any, limit: int) -> str:
    """把参数或结果转为不超过 limit 字符的文本

    参数:
        value: 参数或结果
        limit: 最大字符数

    返回:
        str: 文本
    """
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    return text if len(text) <= limit else f"{text[:limit]}…"


class _Resolved:
    """已解析的工具载荷，原样提供给 Agent"""

    def __init__(self, payload: ResolvedToolPayload):
        self.payload = payload

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        return self.payload


class AgentCapability(AbstractCapability):
    """超级用户在 WebUI 开启 Agent 后，为其提供 HTTP 请求、沙箱、MCP 与提问工具及
    提示词，这些工具调用直接执行并记录"""

    async def active(self, context: RunContext) -> bool:
        """当前对话是否启用 Agent

        参数:
            context: 运行上下文

        返回:
            bool: 已开启 Agent 且对话者是超级用户时为 True
        """
        enabled = (
            context.state.get("_conversation", {}).get("settings", {}).get("agent")
        )
        if not (agent_enabled() if enabled is None else enabled):
            return False
        if context.state.get(AGENT_ACTIVE_KEY) is False:
            return False
        return await PermissionUtils.check_superuser(context)

    async def get_tools(self, context: RunContext) -> list[Any]:
        if not await self.active(context):
            return []
        if context.run.delegated_agent_tools is not None:
            tools = context.run.delegated_agent_tools
            context.state[AGENT_TOOLS_KEY] = {tool.name for tool in tools}
            payload = ResolvedToolPayload(tools=tools)
            self._describe(payload, context)
            return [_Resolved(payload)]
        from zhenxun.services.ai.tools.engine.registry import resolve_tools
        from zhenxun.services.ai.tools.providers.builtin.hitl import HITLToolkit
        from zhenxun.services.ai.tools.providers.builtin.rest_api import (
            RestApiToolkit,
        )
        from zhenxun.services.ai.tools.providers.mcp.provider import MCPSource

        tools: list[Any] = [HITLToolkit(), RestApiToolkit()]
        if (await SandboxManager.ready())[0]:
            from zhenxun.services.ai.tools.providers.builtin.sandbox import (
                SandboxToolkit,
            )

            tools.append(SandboxToolkit())
        tools.append(MCPSource.all_enabled())
        payload = await resolve_tools(tools, context)
        context.state[AGENT_TOOLS_KEY] = {getattr(t, "name", "") for t in payload.tools}
        self._describe(payload, context)
        return [_Resolved(payload)]

    @staticmethod
    def _describe(payload, context):
        """按最终可用的敏感工具附加操作说明。"""
        sensitive = frozenset(context.state[AGENT_TOOLS_KEY])
        payload.instruction_builders.append(
            lambda names: [
                "## Agent 能力\n当前调用者已通过敏感工具授权；"
                "仅使用本轮实际提供的工具。"
                "工具调用会直接执行，有副作用的操作应确认意图。"
                + (
                    "意图不明确时先用 `ask_user_for_help` 确认。"
                    if "ask_user_for_help" in names
                    else ""
                )
            ]
            if names & sensitive
            else []
        )

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if tool_name not in context.state.get(AGENT_TOOLS_KEY, ()):
            return await handler(arguments)
        if not await self.active(context):
            context.run.operations["permission_reason"] = context.state.get(
                "_permission_reason", "Agent 开关已关闭"
            )
            return ToolResult(output="Agent 授权或开关已变化，操作未执行。").as_error(
                False
            )
        call = AIAgentCall(
            run_id=context.state.get(RUN_ID_KEY),
            session_id=context.session_id or "",
            user_id=context.get_user_id() or context.state.get(USER_KEY) or "",
            group_id=context.get_group_id(),
            tool=tool_name,
            arguments=_shorten(arguments, 2000),
            decision="auto",
        )
        result: Any = None
        start = time.monotonic()
        try:
            result = await handler(arguments)
            return result
        except (ControlFlowExit, Exception) as e:
            result = ToolResult(output=f"{type(e).__name__}: {e}").as_error()
            raise
        finally:
            if isinstance(result, ToolResult):
                call.is_error = result.is_error
                call.result = _shorten(result.output, 2000)
            elif result is not None:
                call.result = _shorten(result, 2000)
            call.duration = int((time.monotonic() - start) * 1000)
            try:
                await call.save()
            except Exception as e:
                logger.warning(f"保存 Agent 调用记录失败: {e}", MODULE)


SUBAGENT_DEPTH_KEY = "_subagent_depth"
"""运行上下文 state 中的子代理嵌套层数"""


class SubagentToolkit(BaseToolkit):
    """人设配置的子代理：每个子代理是另一个人设（用它自己的模型与工具），
    主 Agent 通过 transfer_to_<id> 把任务交给它并取回文字结果；子代理不能再转交"""

    def __init__(self, persona_ids: list[int]):
        super().__init__()
        self.persona_ids = persona_ids

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        from zhenxun.models.ai_persona import AIPersona
        from zhenxun.services.ai.tools.core.tool import FunctionTool
        from zhenxun.services.ai.tools.models import ToolOptions

        if not self.persona_ids or (context and context.state.get(SUBAGENT_DEPTH_KEY)):
            return ResolvedToolPayload()
        personas = await AIPersona.filter(id__in=self.persona_ids).order_by("id")
        tools = []
        for persona in personas:
            brief = " ".join(persona.prompt.split())[:120]
            function_tool = FunctionTool(
                _transfer(persona.id),
                name=f"transfer_to_{persona.id}",
                description=f"把一项任务交给子代理「{persona.name}」完成，"
                f"返回它的文字结果。它的设定：{brief}",
                settings=ToolOptions(metadata={"no_timeout": True, "delegates": True}),
            )
            tools.extend((await function_tool.resolve(context)).tools)
        return ResolvedToolPayload(tools=tools)


def _transfer(persona_id: int):
    """生成转交给某个子代理的工具函数

    参数:
        persona_id: 子代理人设id

    返回:
        Callable: 工具函数
    """

    async def transfer(task: str, context: RunContext) -> ToolResult:
        """把任务交给子代理

        参数:
            task: 交给子代理的任务，写清背景与要求
        """
        from zhenxun.models.ai_persona import AIPersona
        from zhenxun.services.ai.context.memory.engine import chat_backend

        from ._data_source import AIChatManager

        persona = await AIPersona.get_or_none(id=persona_id)
        if persona is None or not persona.enabled:
            return ToolResult(output="这个子代理已不存在或停用。").as_error(False)
        agent = AIChatManager.get_agent(persona)
        child = RunContext(
            deps=context.deps,
            session_id=f"{context.session_id}:sub:{uuid.uuid4().hex}",
            shared_state=context.shared_state,
        )
        child.run.caller = context.run.caller
        child.run.management_check = context.run.management_check
        child.run.tool_ceiling = context.run.available_tools
        child.run.delegated_agent_tools = [
            tool
            for tool in context.run.bound_tools.values()
            if tool.name in context.state.get(AGENT_TOOLS_KEY, ())
        ]
        child.run.operations = context.run.operations
        child.state[RUN_ID_KEY] = child.run.run_id
        child.run.parent_run_id = context.run.run_id
        child.state["_parent_event_bus"] = context.run.event_bus
        child.run.root_session_id = context.run.root_session_id
        child.run.root_revision = context.run.root_revision
        if AGENT_ACTIVE_KEY in context.state:
            child.state[AGENT_ACTIVE_KEY] = context.state[AGENT_ACTIVE_KEY]
        if context.run.source not in SOURCES:
            return ToolResult(output="子代理缺少可信来源，未执行。").as_error(False)
        bind_source(child, context.run.source)
        AIChatManager.prepare_input(child, task)
        child.run.budget = context.run.budget
        child.run.follow_up = False
        child.state[SUBAGENT_DEPTH_KEY] = 1
        from zhenxun.services.ai.core.models import CancellationToken

        child.run.cancellation_token = CancellationToken(context.run.cancellation_token)
        from zhenxun.services.ai.run.session import session_manager

        session_manager.register(child)
        try:
            await session_manager.accept(child)
            limit = context.run.operations.setdefault(
                "child_limit", asyncio.Semaphore(2)
            )
            async with limit:
                child.run.cancellation_token.raise_if_cancelled()
                persona = await AIPersona.get_or_none(id=persona_id)
                if persona is None or not persona.enabled:
                    child.run.outcome = "skipped"
                    return ToolResult(output="这个子代理已不存在或停用。").as_error(
                        False
                    )
                agent = AIChatManager.get_agent(persona)
                AIChatManager.configuration(child, None, persona)
                child.run.current_model = child.state["_configuration"]["model"]
                result = await agent.run(task, context=child)
            if result.stopped:
                from zhenxun.services.ai.core.exceptions import RunInterrupted

                if isinstance(result.stopped, RunInterrupted):
                    raise result.stopped
                return ToolResult(output="子代理已停止。").as_error(False)
        except asyncio.CancelledError:
            if (
                child.run.cancellation_token.reason == "cancelled"
                and not context.run.cancellation_token.is_cancelled()
            ):
                return ToolResult(output="子代理已停止，父任务可以继续。").as_error(
                    False
                )
            raise
        except LLMException as e:
            return ToolResult(
                output=f"子代理执行失败: {e.user_friendly_message}"
            ).as_error(False)
        finally:
            await session_manager.finish(child)
            child.run.cancellation_token.detach()
            if backend := chat_backend(agent.memory_config):
                await backend.clear(child.session_id or "")
        return ToolResult(
            output=str(result.output).strip() or "子代理没有给出结果。",
            usage=result.usage,
        )

    return transfer
