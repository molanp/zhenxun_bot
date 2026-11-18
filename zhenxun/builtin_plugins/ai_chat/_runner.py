import asyncio
import hashlib
import time

"""第三方执行器：选用非 local 执行器时由它代替本地 Agent 完成一轮对话，
唤醒、限流、安全过滤、额度与发送仍由 ai_chat 处理"""

from typing import Any

from zhenxun.configs.config import Config
from zhenxun.models.ai_chat_group import AIChatGroup
from zhenxun.models.ai_persona import AIPersona
from zhenxun.services.ai.core.exceptions import LLMException, ToolFatalError
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
APPROVE_ONCE = {"允许", "批准", "allow", "yes"}
"""批准本次 DSH 请求的回复"""
APPROVE_ALL = {"全部允许", "一键允许", "allow all"}
"""批准本轮后续全部 DSH 请求的回复"""
APPROVE_HINT = "允许：仅本次；全部允许：本轮；停止：结束本轮。其他回复拒绝本次。"
"""聊天中批准请求的回复说明"""


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

    from ._attachments import collect_export, export_instruction, export_slot
    from ._reply import clean_output
    from ._safety import Safety

    session_id = context.session_id or ""
    timeout = float(dsh_settings().timeout)
    deadline = 0.0
    waiting = 0
    settling = False
    paused_at = 0.0
    approval_lock = asyncio.Lock()
    remote = None

    async def approve(tool: str, reason: str) -> bool:
        try:
            PermissionUtils.ensure_current(context)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            return False
        allowed = await PermissionUtils.check_superuser(context)
        how = ""
        if allowed:
            async with approval_lock:
                token.raise_if_cancelled()
                if dsh_settings().permission == "full":
                    how = "（完全权限）"
                elif context.state.get("_dsh_approve_all"):
                    how = "（本轮全部允许）"
                else:
                    try:
                        reply = await ask(
                            f"DSH 请求批准：{tool}\n{reason or '未提供说明'}",
                            approval=True,
                        )
                    except (ToolFatalError, asyncio.TimeoutError):
                        reply = ""
                    reply = reply.strip().lower()
                    if reply in {"停止", "stop"}:
                        token.cancel("cancelled")
                        raise asyncio.CancelledError
                    if reply in APPROVE_ALL:
                        context.state["_dsh_approve_all"] = True
                    allowed = reply in APPROVE_ONCE | APPROVE_ALL
        decision = (f"自动允许{how}" if how else "允许") if allowed else "拒绝"
        logger.info(
            f"dsh 请求使用 {tool}: {decision}",
            MODULE,
            session=context.get_user_id(),
        )
        return allowed

    async def ask(question: str, approval: bool = False) -> str:
        nonlocal waiting, paused_at, deadline
        if not waiting:
            paused_at = time.monotonic()
        waiting += 1
        try:
            if callback := context.state.get("_dsh_ask"):
                return await callback(question, approval)
            if approval:
                question += f"\n{APPROVE_HINT}"
            return await HITLController(context).ask_text(question, timeout=300)
        finally:
            waiting -= 1
            if not waiting:
                paused = time.monotonic() - paused_at
                deadline += paused
                if remote:
                    DshRuntime.extend(remote, paused)

    token = context.run.cancellation_token or CancellationToken()
    context.run.cancellation_token = token
    revision = session_manager.revision(session_id)
    async with session_manager.run_scope(
        session_id, token, exclusive=True, revision=revision
    ):
        await session_manager.phase(context, "running")
        deadline = time.monotonic() + timeout
        expired = False

        async def turn():
            nonlocal remote, settling
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
                persona_text = AIChatManager.build_instruction(selected)
                stamp = hashlib.sha256(
                    (binding["remote_id"] + "\0" + persona_text).encode()
                ).hexdigest()
                if new or binding.get("applied_persona") != stamp:
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"<本轮人设>\n{persona_text}\n</本轮人设>",
                        }
                    )
                if not binding.get("native"):
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
                    if hint := context.run.dynamic_prompts.get("proactive"):
                        blocks.append({"type": "text", "text": hint})
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
                with export_slot(context, binding) as slot:
                    if slot:
                        blocks.append(
                            {"type": "text", "text": export_instruction(slot[1])}
                        )
                    raw, usage, reason = await DshRuntime.prompt(remote, blocks)
                    settling = True
                    text, blocked = Safety.filter_output(clean_output(raw))
                    if slot and not blocked:
                        text = collect_export(context, slot, text)
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
                    context.run.user_notice = (
                        f"DSH 本轮执行超过 {int(timeout)} 秒，已停止。"
                        "等待确认的时间不计入，可在 DSH 连接设置中调整单轮超时。"
                    )
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
                attachment = delivery.get("attachment")
                clean = (
                    delivery.get("status") == "sent"
                    and not delivery.get("uncertain")
                    and (
                        attachment.get("status") == "sent"
                        if attachment
                        else text == raw
                    )
                )
                DshSessions.for_binding(binding).update(
                    binding,
                    inflight=None,
                    quarantined=None
                    if clean or binding.get("native")
                    else "delivery_or_filter_changed",
                    delivery_uncertain=not clean,
                    applied_persona=stamp,
                )
                return result
            except BaseException:
                DshSessions.for_binding(binding).update(
                    binding, inflight=None, quarantined="settlement_unconfirmed"
                )
                raise
            finally:
                await DshRuntime.release(remote)

        task = asyncio.create_task(turn())
        try:
            while not task.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0 and not waiting and not settling:
                    expired = True
                    task.cancel()
                    break
                await asyncio.wait({task}, timeout=max(remaining, 1.0))
            try:
                return await task
            except asyncio.CancelledError:
                if expired and not token.is_cancelled():
                    raise asyncio.TimeoutError("本轮执行超时") from None
                raise
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


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


async def dsh_command(context, argument: str, *, current_mode: bool = False) -> str:
    """执行 DSH 原生会话控制命令

    参数:
        context: 运行上下文
        argument: 子命令与参数
        current_mode: 按当前场景的 DSH 模式执行，否则为 Agent 模式

    返回:
        str: 执行结果
    """
    from zhenxun.services.ai.run.dsh_control import DshControl

    from ._commands import AGENT_PRIVATE, require_admin
    from ._sessions import Conversations

    action, _, value = argument.strip().partition(" ")
    settings = dsh_settings()
    records = DshSessions.for_settings(settings)
    persona = await Conversations.persona(
        context.state["_conversation"], context.get_group_id(), context.get_user_id()
    )
    mode = await dsh_mode(context, persona) if current_mode else "agent"
    if context.get_group_id() and mode == "agent":
        # 群内重置针对本群共用的聊天会话；跨场景共用的 Agent 会话只在私聊管理
        if current_mode and action == "reset":
            mode = "chat"
        elif action in {"ls", "new", "switch", "archive"}:
            raise ValueError(AGENT_PRIVATE)
    if mode == "agent":
        await require_admin(context)
    owner = records.owner(context, mode)
    current = records._read()["owners"].get(owner)
    if action in {"", "status"}:
        state = await DshRuntime.status(settings)
        return "\n".join(
            f"{key}: {state.get(key)}"
            for key in ("transport", "connected", "version", "reason", "native_ready")
        )
    if action in {"new", "ls", "reset", "archive"} and value:
        raise ValueError(
            "此 DSH 操作不接受名称、分页或额外目标；请使用完整会话 ID 切换。"
        )
    if action == "ls":
        rows = await DshControl.list(settings, owner, mode=mode)
        return (
            "\n".join(
                f"{r['remote_id']} · {r['title']} · {r.get('model') or '继承'}"
                for r in rows[:30]
            )
            or "没有可接入的会话。"
        )
    if action in {"new", "switch"}:
        if action == "switch" and not value:
            raise ValueError("请提供 /ai dsh ls 中的完整会话 ID。")
        selected = await DshControl.choose(
            context.run.caller,
            settings,
            value if action == "switch" else None,
            current["revision"] if current else 0,
            mode=mode,
        )
        if action == "new" and context.get_group_id():
            from ._buffer import ChatBuffer

            ChatBuffer.reset(context.get_bot().self_id, context.get_group_id())
        return (
            f"已选择 {selected['remote_id']}；"
            "回复到每次触发位置，原会话历史和设置保留。"
        )
    if not current:
        raise ValueError("请先新建或选择 DSH 会话。")
    if action in {"reset", "archive"}:
        await DshControl.release(settings, current, archive=action == "archive")
        if action == "reset":
            await AIChatManager.reset(current["local_id"])
        return "已解除关联；远端历史未删除。"
    if action in {"model", "thinking", "mode"}:
        return await DshControl.parameter(settings, current, action, value)
    raise ValueError("不支持的 DSH 命令；未作为聊天内容发送。")
