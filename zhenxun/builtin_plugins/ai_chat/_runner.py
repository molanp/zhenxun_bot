import asyncio
import time

"""第三方执行器：选用非 local 执行器时由它代替本地 Agent 完成一轮对话，
唤醒、限流、安全过滤、额度与发送仍由 ai_chat 处理"""

from typing import Any

from zhenxun.configs.config import Config
from zhenxun.models.ai_chat_group import AIChatGroup
from zhenxun.models.ai_persona import AIPersona
from zhenxun.services.ai.core.exceptions import LLMException
from zhenxun.services.ai.core.messages import TextPart
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.run.context import RunContext, set_run_context
from zhenxun.services.ai.run.dsh import (
    DshRuntime,
    DshSessions,
    DshSettings,
    content_blocks,
)
from zhenxun.services.ai.run.models import AgentRunResult
from zhenxun.services.ai.run.session import session_manager
from zhenxun.services.ai.utils.runtime import PermissionUtils
from zhenxun.services.log import logger

from ._data_source import MODULE, AIChatManager, _time_prompt

RUNNERS = {"local": "本地 Agent", "dsh": "DeepSeek Harness"}
"""可选的执行器"""


def agent_runner(setting: AIChatGroup | None) -> str:
    """会话使用的执行器：群设置优先，其次为全局 AGENT_RUNNER

    参数:
        setting: 群设置，私聊为空

    返回:
        str: 执行器名称
    """
    runner = (setting.agent_runner if setting else None) or Config.get_config(
        MODULE, "AGENT_RUNNER", "local"
    )
    return runner if runner in RUNNERS else "local"


def dsh_settings() -> DshSettings:
    """读取聊天设置中的 DSH

    返回:
        DshSettings: dsh 执行器设置
    """
    return DshSettings(**(Config.get_config(MODULE, "DSH") or {}))


async def dsh_mode(context: RunContext, persona: AIPersona | None) -> str:
    """从当前来源、超管授权、Agent 开关和人设确定桌面权限上限。"""
    from ._agent import AgentCapability

    enabled = dsh_settings().allow_tools and (
        persona is None or persona.tools is None or "agent" in persona.tools
    )
    return "agent" if enabled and await AgentCapability().active(context) else "chat"


async def select_dsh_session(context: RunContext, persona: AIPersona | None) -> dict:
    """在排队前绑定所有者与模式，回复位置仍由本轮上下文持有。"""
    mode = await dsh_mode(context, persona)
    key = f"{persona.id}:{persona.updated_at.isoformat()}" if persona else ""
    return DshSessions.for_settings(dsh_settings()).route(context, mode, key)


async def scene_session(bot, event) -> str:
    """读取当前场景的有效桌面映射，不为停止或重置新建会话。"""
    from zhenxun.services.ai.run.context import NoneBotDeps

    from ._agent import bind_source
    from ._sessions import Conversations

    context = RunContext(deps=NoneBotDeps(bot=bot, event=event))
    scene = AIChatManager.session_id(bot, event)
    snapshot = await Conversations.snapshot(scene, create=False)
    preferences = snapshot["settings"]
    context.state["_conversation"] = snapshot
    group = context.get_group_id()
    setting = await AIChatGroup.get_setting(group) if group else None
    if (preferences.get("runner") or agent_runner(setting)) == "dsh":
        bind_source(context, "qq")
        persona = await Conversations.persona(
            context.state["_conversation"], group, context.get_user_id()
        )
        records = DshSessions.for_settings(dsh_settings())
        owner = records.owner(context, await dsh_mode(context, persona))
        item = records._read()["owners"].get(owner)
        if item:
            return item["local_id"]
    return snapshot["session_id"] or scene


async def stop_dsh_runs(context: RunContext, manage: bool) -> tuple[int, int]:
    """按本轮可信来源停止桌面运行，不依赖后来切换的执行器或模式。"""
    caller = context.run.caller
    if caller is None:
        return 0, 0
    targets = []
    for run_id, (active, task) in list(session_manager.runs.items()):
        other = active.run.caller
        binding = active.state.get("_dsh_binding")
        if (
            not binding
            or not other
            or not task
            or task.done()
            or other.source == "webui"
            or (other.platform, other.bot_id) != (caller.platform, caller.bot_id)
        ):
            continue
        own_agent = binding["mode"] == "agent" and other.user_id == caller.user_id
        same_scene = other.group_id == caller.group_id and (
            bool(caller.group_id) or other.user_id == caller.user_id
        )
        if own_agent or (same_scene and (manage or other.user_id == caller.user_id)):
            targets.append((run_id, active))
    results = await asyncio.gather(
        *(session_manager.stop(run_id) for run_id, _ in targets)
    )
    confirmed = sum(
        stopped and active.run.operations.get("dsh", {}).get("stop") != "unknown"
        for (_, active), stopped in zip(targets, results)
    )
    return confirmed, len(targets) - confirmed


async def _run_dsh(
    context: RunContext,
    parts: list[Any],
    persona: AIPersona | None,
    *,
    finalize=None,
    before_run=None,
    prepare_run=None,
) -> AgentRunResult:
    """在会话执行锁内调用桌面，等待投递和历史结算后释放归属。

    参数:
        context: 运行上下文
        parts: 本轮输入
        persona: 人设

    返回:
        AgentRunResult: 运行结果
    """
    from zhenxun.services.ai.context.memory.engine import chat_backend
    from zhenxun.services.ai.core.exceptions import RunInterrupted
    from zhenxun.services.ai.core.messages import LLMMessage
    from zhenxun.services.ai.run.hitl import HITLController

    from ._reply import clean_output
    from ._safety import Safety

    session_id = context.session_id or ""

    async def approve(tool: str, reason: str) -> bool:
        try:
            PermissionUtils.ensure_current(context)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            return False
        allowed = await PermissionUtils.check_superuser(context)
        if allowed:
            reply = await ask(
                f"DSH 请求本次批准：{tool}\n{reason or '未提供说明'}\n"
                "回复“允许”仅批准本次，其他回答视为拒绝。"
            )
            allowed = reply.strip().lower() in {"允许", "批准", "allow", "yes"}
        logger.info(
            f"dsh 请求使用 {tool}: " + ("允许" if allowed else "拒绝"),
            MODULE,
            session=context.get_user_id(),
        )
        return allowed

    async def ask(question: str) -> str:
        if callback := context.state.get("_dsh_ask"):
            return await callback(question)
        return await HITLController(context).ask_text(question, timeout=120)

    token = context.run.cancellation_token or CancellationToken()
    context.run.cancellation_token = token
    revision = session_manager.revision(session_id)
    async with session_manager.run_scope(
        session_id, token, exclusive=True, revision=revision
    ):
        await session_manager.phase(context, "running")
        deadline = time.monotonic() + float(context.run.timeout or 300)
        expired = False

        def expire():
            nonlocal expired
            expired = True

        async def turn():
            remote = None
            binding = context.state.get("_dsh_binding")
            if binding is None:
                raise LLMException("DSH 会话没有在准入时绑定。")
            try:
                DshSessions.for_binding(binding).refresh(binding)
                prepared, selected = parts, persona
                if prepare_run:
                    prepared, selected = await prepare_run()
                if before_run:
                    await before_run()
                settings = dsh_settings()
                persona_key = (
                    f"{selected.id}:{selected.updated_at.isoformat()}"
                    if selected
                    else ""
                )
                if (
                    persona_key != context.state.get("_dsh_persona_at_admission")
                    or await dsh_mode(context, selected) != binding["mode"]
                ):
                    raise LLMException("排队期间人设或 Agent 权限已变化，请重新提问。")

                async def authorization_check():
                    PermissionUtils.ensure_current(context)
                    current = (
                        await AIPersona.get_or_none(id=selected.id)
                        if selected
                        else None
                    )
                    if selected and (
                        current is None
                        or not current.enabled
                        or current.updated_at != selected.updated_at
                    ):
                        return "denied"
                    group = context.get_group_id()
                    current_setting = (
                        await AIChatGroup.get_setting(group) if group else None
                    )
                    snapshot = context.state.get("_conversation")
                    if snapshot:
                        from ._sessions import Conversations

                        await Conversations.check(snapshot)
                    if (
                        (snapshot or {}).get("settings", {}).get("runner")
                        or agent_runner(current_setting)
                    ) != "dsh":
                        return "denied"
                    return await dsh_mode(context, current)

                context.state["_dsh_authorization_check"] = authorization_check
                DshSessions.for_binding(binding).update(binding, persona=persona_key)
                context.run.budget["deadline"] = deadline
                backend = chat_backend(AIChatManager.get_agent(selected).memory_config)
                history = await backend.get_messages(session_id) if backend else []
                remote_id = binding["remote_id"]
                new = (
                    not binding["created"]
                    or binding.get("quarantined")
                    or binding.get("inflight")
                )
                remote = await DshRuntime.start(
                    context, binding, settings, approve, ask
                )
                new = new or remote_id != binding["remote_id"]
                blocks = []
                if not binding.get("native"):
                    persona_text = AIChatManager.build_instruction(selected)
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"<本轮人设>\n{persona_text}\n</本轮人设>",
                        }
                    )
                    if new:
                        transcript = []
                        for message in history[-40:]:
                            text = "".join(
                                p.text
                                for p in message.content
                                if isinstance(p, TextPart)
                            )
                            if text:
                                transcript.append(f"{message.role}: {text}")
                        if transcript:
                            blocks.append(
                                {
                                    "type": "text",
                                    "text": "以下为真寻已确认送达的历史：\n"
                                    + "\n".join(transcript),
                                }
                            )
                    blocks.append({"type": "text", "text": _time_prompt(context)})
                blocks += await content_blocks(
                    prepared, native=bool(binding.get("native"))
                )
                if backend:
                    user_message = LLMMessage.user(prepared)
                    user_message.metadata = {
                        **context.run.history_metadata,
                        **context.state.get("_dsh_input_metadata", {}),
                    }
                    await backend.add_messages(session_id, [user_message])
                raw, usage, reason = await DshRuntime.prompt(
                    remote, blocks, settings.timeout
                )
                text = Safety.filter_output(clean_output(raw))[0]
                if backend:
                    assistant = LLMMessage.assistant_text_response(text)
                    assistant.metadata = {
                        **context.run.history_metadata,
                        "chat_final": True,
                    }
                    await backend.add_messages(session_id, [assistant])
                result = AgentRunResult(output=text, usage=usage)
            except BaseException as error:
                from zhenxun.services.ai.run.dsh_desktop import DshUnavailable

                if isinstance(error, DshUnavailable):
                    context.run.user_notice = str(error)
                context.run.outcome = (
                    "interrupted"
                    if isinstance(error, asyncio.CancelledError | RunInterrupted)
                    else "failed"
                )
                if remote:
                    cleanup = asyncio.create_task(DshRuntime.interrupt(remote))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await asyncio.shield(cleanup)
                    context.run.operations["external_stop_guarantee"] = (
                        remote.stop_confirmation
                    )
                if (
                    isinstance(error, asyncio.CancelledError)
                    and expired
                    and not token.is_cancelled()
                ):
                    error = asyncio.TimeoutError("本轮执行超时")
                if isinstance(error, asyncio.TimeoutError):
                    context.run.outcome = "timeout"
                elif (
                    isinstance(error, asyncio.CancelledError)
                    and token.reason == "cancelled"
                ):
                    context.run.outcome = "cancelled"
                try:
                    if finalize:
                        await finalize(None, error)
                finally:
                    if remote:
                        await DshRuntime.release(remote)
                raise
            try:
                context.run.outcome = "completed"
                if finalize:
                    await finalize(result, None)
                delivery = context.state.get("_delivery", {})
                clean = (
                    delivery.get("status") == "sent"
                    and not delivery.get("uncertain")
                    and text == raw
                )
                DshSessions.for_binding(binding).update(
                    binding,
                    inflight=None,
                    quarantined=None
                    if clean or binding.get("native")
                    else "delivery_or_filter_changed",
                    delivery_uncertain=not clean,
                )
                return result
            except BaseException:
                DshSessions.for_binding(binding).update(
                    binding, inflight=None, quarantined="settlement_unconfirmed"
                )
                raise
            finally:
                await DshRuntime.release(remote)

        remaining = max(0.001, deadline - time.monotonic())
        timer = asyncio.get_running_loop().call_later(remaining, expire)
        try:
            return await asyncio.wait_for(turn(), remaining)
        finally:
            timer.cancel()


async def run_dsh(
    context: RunContext,
    parts: list[Any],
    persona: AIPersona | None,
    *,
    finalize=None,
    before_run=None,
    prepare_run=None,
) -> AgentRunResult:
    """登记 DSH 本地等待范围；取消不等于已确认远端执行停止。"""
    session_manager.register(context)
    with set_run_context(context):
        try:
            await session_manager.accept(context)
            return await _run_dsh(
                context,
                parts,
                persona,
                finalize=finalize,
                before_run=before_run,
                prepare_run=prepare_run,
            )
        finally:
            await session_manager.finish(context)
