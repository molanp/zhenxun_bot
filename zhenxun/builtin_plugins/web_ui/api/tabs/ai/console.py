"""AI 控制台：超级用户在 WebUI 与 AI 聊天或 Agent 对话，流式输出、停止与内联审批"""

import asyncio
from collections.abc import AsyncIterator
import hashlib
import json
import time
from typing import Any
import uuid

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zhenxun.services.log import logger

from ....base_model import Result
from ....utils import authentication

router = APIRouter()

CONSOLE_PREFIX = "webui_console:"
"""控制台会话id前缀，后接人设id（默认人设为 default）"""
_tasks: dict[str, asyncio.Task] = {}


class ConsoleRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    persona_id: int | None = None
    agent: bool = False
    session_id: str | None = Field(default=None, max_length=255)


class ConsoleStopRequest(BaseModel):
    persona_id: int | None = None
    run_id: str | None = Field(default=None, max_length=32)


class ConsoleAnswerRequest(BaseModel):
    run_id: str = Field(min_length=32, max_length=32)
    question_id: str = Field(min_length=32, max_length=32)
    answer: str = Field(min_length=1, max_length=4000)


@router.post("/console/answer", dependencies=[authentication()], response_model=Result)
async def console_answer(payload: ConsoleAnswerRequest, request: Request) -> Result:
    from zhenxun.services.ai.run.session import session_manager
    from zhenxun.services.ai.utils.runtime import PermissionUtils

    subject = management_subject(request)
    entry = session_manager.runs.get(payload.run_id)
    context = entry[0] if entry else None
    if (
        not context
        or not context.run.caller
        or context.run.caller.source != "webui"
        or context.run.caller.user_id != f"webui:{subject}"
    ):
        raise HTTPException(409, detail={"code": "question_not_current"})
    try:
        PermissionUtils.ensure_current(context)
    except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
        raise HTTPException(409, detail={"code": "question_not_current"}) from exc
    pending = context.state.get("_dsh_question")
    if not pending or pending[0] != payload.question_id or pending[1].done():
        raise HTTPException(409, detail={"code": "question_not_current"})
    pending[1].set_result(payload.answer)
    return Result.ok(info="已提交本轮回答。")


def management_subject(request: Request) -> str:
    """读取已经验证的管理账号主体，不转换为 QQ 身份。"""
    from ....security import decode_access_token_status

    principal, status = decode_access_token_status(
        request.headers.get("Authorization", "")
    )
    if status != "valid" or not principal:
        raise HTTPException(401, detail={"code": "invalid_auth"})
    return str(principal["sub"])


@router.get("/console/session", dependencies=[authentication()], response_model=Result)
async def resolve_console_session(
    request: Request, persona_id: int | None = None, agent: bool = False
) -> Result:
    from zhenxun.builtin_plugins.ai_chat._agent import AGENT_ACTIVE_KEY, bind_source
    from zhenxun.builtin_plugins.ai_chat._runner import agent_runner, dsh_mode
    from zhenxun.models.ai_persona import AIPersona
    from zhenxun.services.ai.run.context import RunContext, ScheduledDeps
    from zhenxun.services.ai.run.dsh import DshSessions

    from ....security import validate_access_token

    subject = management_subject(request)
    if agent_runner(None) != "dsh":
        return Result.ok(
            {"session_id": console_session(persona_id), "executor": "local"}
        )
    context = RunContext(
        deps=ScheduledDeps(user_id=f"webui:{subject}", platform="webui")
    )
    credential = request.headers.get("Authorization", "")
    bind_source(
        context, "webui", management_check=lambda: validate_access_token(credential)
    )
    context.state[AGENT_ACTIVE_KEY] = agent
    persona = await AIPersona.get_or_none(id=persona_id) if persona_id else None
    mode = await dsh_mode(context, persona)
    item = DshSessions._read()["owners"].get(DshSessions.owner(context, mode))
    return Result.ok(
        {
            "session_id": item["local_id"] if item else None,
            "executor": "dsh",
            "mode": mode,
            "revision": item["revision"] if item else None,
        }
    )


def console_session(persona_id: int | None) -> str:
    """控制台会话id

    参数:
        persona_id: 人设id，为空时为默认人设

    返回:
        str: 会话id
    """
    return f"{CONSOLE_PREFIX}{persona_id if persona_id is not None else 'default'}"


def _shorten(value: Any, limit: int = 2000) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    return text if len(text) <= limit else f"{text[:limit]}…"


def _event_item(event: Any) -> dict[str, Any] | None:
    """把 Agent 工具事件转为推送给前端的数据，回复正文只随结束事件推送

    参数:
        event: Agent 流事件

    返回:
        dict[str, Any] | None: 推送数据，不需要推送的事件为空
    """
    from zhenxun.services.ai.core.stream_events import (
        ToolCallEndEvent,
        ToolCallStartEvent,
    )

    if isinstance(event, ToolCallStartEvent):
        return {
            "type": "tool_start",
            "name": event.tool_name,
            "run_id": event.run_id,
            "parent_run_id": event.parent_run_id,
            "tool_call_id": event.tool_call_id,
            "arguments": _shorten(event.arguments),
        }
    if isinstance(event, ToolCallEndEvent):
        result = getattr(event.result, "output", event.result)
        return {
            "type": "tool_end",
            "name": event.tool_name,
            "run_id": event.run_id,
            "parent_run_id": event.parent_run_id,
            "tool_call_id": event.tool_call_id,
            "result": _shorten(result),
            "is_error": event.is_error,
            "uncertain": bool(getattr(event.result, "uncertain", False)),
            "reused": bool(getattr(event.result, "reused", False)),
        }
    return None


async def _run(
    payload: ConsoleRequest,
    persona: Any,
    queue: asyncio.Queue[dict | None],
    receipt: dict | None = None,
    management_check=None,
    subject: str = "",
) -> None:
    """在会话锁内运行一次控制台对话：工具事件实时推送，回复正文随结束事件推送，
    等待前端收到结束事件后按实际送达的内容结算历史与运行记录，最后写入 None

    参数:
        payload: 请求
        persona: 人设
        queue: 推送队列
        receipt: 推送确认，accepted 为前端已收到结束事件
    """
    from zhenxun.builtin_plugins.ai_chat._agent import (
        AGENT_ACTIVE_KEY,
        USER_KEY,
        bind_source,
    )
    from zhenxun.builtin_plugins.ai_chat._data_source import AIChatManager
    from zhenxun.builtin_plugins.ai_chat._reply import parse_silent
    from zhenxun.builtin_plugins.ai_chat._runner import (
        agent_runner,
        run_dsh,
        select_dsh_session,
    )
    from zhenxun.builtin_plugins.ai_chat._safety import Safety
    from zhenxun.builtin_plugins.ai_chat._turn import FAILED_NOTICE, ChatTurn
    from zhenxun.services.ai.core.exceptions import ControlFlowExit
    from zhenxun.services.ai.run.context import RunContext, ScheduledDeps
    from zhenxun.services.ai.run.session import session_manager

    context = RunContext(
        session_id=console_session(payload.persona_id),
        deps=ScheduledDeps(user_id=f"webui:{subject}", platform="webui"),
    )
    if receipt and receipt.get("run_id"):
        context.run.run_id = receipt["run_id"]
    bind_source(context, "webui", management_check=management_check)
    context.state[AGENT_ACTIVE_KEY] = payload.agent
    context.state[USER_KEY] = "WebUI"
    context.state["_dsh_event"] = queue.put_nowait

    async def ask_dsh(question):
        question_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        context.state["_dsh_question"] = (question_id, future)
        try:
            await queue.put(
                {
                    "type": "question",
                    "run_id": context.run.run_id,
                    "question_id": question_id,
                    "question": question,
                }
            )
            return await asyncio.wait_for(future, 120)
        finally:
            if context.state.get("_dsh_question", (None,))[0] == question_id:
                context.state.pop("_dsh_question", None)
            if not future.done():
                future.cancel()

    context.state["_dsh_ask"] = ask_dsh
    runner = agent_runner(None)
    if runner == "dsh":
        try:
            await select_dsh_session(context, persona)
            if payload.session_id and payload.session_id != context.session_id:
                raise ValueError("DSH 会话已变化，请刷新后重新发送")
        except Exception:
            await queue.put(
                {
                    "type": "done",
                    "error": "DSH 会话未能确认，请刷新状态后重试。",
                    "outcome": "failed",
                }
            )
            await queue.put(None)
            return
    AIChatManager.prepare_input(context, payload.message)
    agent = AIChatManager.get_agent(persona)
    turn = ChatTurn(
        context,
        agent,
        "webui",
        persona,
        payload.message,
        "WebUI",
        usage_id=f"webui:{subject}" if runner == "dsh" else None,
    )
    AIChatManager.configuration(context, None, persona, runner)
    await queue.put(
        {
            "type": "accepted",
            "run_id": turn.run_id,
            "session_id": context.session_id,
            "executor": runner,
            "phase": "queued",
        }
    )
    receipt = receipt if receipt is not None else {"event": asyncio.Event()}
    finalized = False

    async def prepare_run():
        nonlocal persona, agent
        from zhenxun.models.ai_persona import AIPersona
        from zhenxun.services.ai.core.exceptions import LLMException

        if agent_runner(None) != runner:
            raise LLMException("排队期间执行器已切换，请重新发送。")

        if payload.persona_id is not None:
            persona = await AIPersona.get_or_none(id=payload.persona_id)
            if persona is None or not persona.enabled:
                context.run.user_notice = "所选人设已删除或停用，请重新选择。"
                raise LLMException(context.run.user_notice)
        agent = AIChatManager.get_agent(persona)
        turn.select_agent(agent, persona)
        AIChatManager.configuration(context, None, persona, runner)
        context.run.current_model = context.state["_configuration"]["model"]
        return agent, payload.message

    async def before_run():
        if Safety.check_input(payload.message):
            raise ValueError("输入未通过安全检查")

    async def finalize(result, error=None):
        nonlocal finalized
        if finalized:
            return
        finalized = True
        turn.result = result
        output = parse_silent(str(result.output)) if result else None
        if not output and not isinstance(
            error, asyncio.CancelledError | ControlFlowExit
        ):
            output = (
                context.run.user_notice
                or (persona and persona.error_message)
                or FAILED_NOTICE
            )
        output = Safety.filter_output(output or "")[0]
        from zhenxun.builtin_plugins.ai_chat._agent import (
            AGENT_TOOLS_KEY,
            agent_enabled,
        )
        from zhenxun.services.ai.utils.runtime import PermissionUtils

        allowed = (
            payload.agent
            and agent_enabled()
            and await PermissionUtils.check_superuser(context)
        )
        actual = context.state.get("_available_tool_names", set()) & context.state.get(
            AGENT_TOOLS_KEY, set()
        )
        available = bool(allowed and actual)
        if runner == "dsh":
            available = bool(
                allowed and context.run.operations.get("dsh", {}).get("mode") == "agent"
            )
        reason = ""
        if payload.agent and not available:
            reason = (
                "Agent 开关未开启"
                if not agent_enabled()
                else context.state.get(
                    "_permission_reason", "当前人设没有可用的敏感工具"
                )
            )
        done = {
            "type": "done",
            "run_id": turn.run_id,
            "outcome": context.run.outcome,
            "phase": context.run.phase,
            "duration": int((time.monotonic() - turn.started) * 1000),
            "output": output,
            "tokens": turn.tokens,
            "agent_requested": payload.agent,
            "agent_available": available,
            "agent_unavailable_reason": reason,
            "diagnostics": {
                "dsh": context.run.operations.get("dsh"),
                "tool_conflicts": context.state.get("_tool_conflicts", []),
                "operation_uncertain": bool(
                    context.run.operations.get("uncertain_tools")
                ),
            },
        }
        if error is not None:
            done[
                "stopped"
                if isinstance(error, asyncio.CancelledError | ControlFlowExit)
                else "error"
            ] = (
                "已停止"
                if isinstance(error, asyncio.CancelledError | ControlFlowExit)
                else "本轮未完成，请查看运行记录或稍后重试。"
            )
        try:
            await queue.put(done)
            if not isinstance(error, asyncio.CancelledError):
                await asyncio.wait_for(receipt["event"].wait(), 5)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError as exc:
            error = exc
            raise
        finally:
            if not output:
                turn.delivery.status = "skipped"
            elif receipt.get("accepted"):
                turn.delivery.status, turn.delivery.parts = "sent", [output]
            else:
                turn.delivery.status, turn.delivery.uncertain = "unknown", True
            await turn.settle(error)

    try:
        if runner == "dsh":
            from zhenxun.services.ai.core.messages import TextPart

            async def prepare_dsh():
                await prepare_run()
                return [TextPart(text=payload.message)], persona

            await run_dsh(
                context,
                [TextPart(text=payload.message)],
                persona,
                finalize=finalize,
                before_run=before_run,
                prepare_run=prepare_dsh,
            )
            return
        async with agent.run_stream(
            payload.message,
            context=context,
            finalize=finalize,
            before_run=before_run,
            prepare_run=prepare_run,
        ) as stream:
            async for event in stream.stream_events():
                if item := _event_item(event):
                    await queue.put(item)
            await stream.get_run_result()
    except asyncio.CancelledError as error:
        if not finalized:
            await finalize(None, error)
        raise
    except (ControlFlowExit, Exception) as error:
        if not finalized:
            await finalize(None, error)
        logger.warning(f"控制台对话结束: {type(error).__name__}", "WebUi")
    finally:
        await queue.put(
            {
                "type": "settled",
                "run_id": turn.run_id,
                "outcome": context.run.outcome,
                "phase": "finished",
                "details": session_manager.details(context),
            }
        )
        await queue.put(None)


@router.post("/console/run", dependencies=[authentication()])
async def console_run(payload: ConsoleRequest, request: Request) -> StreamingResponse:
    from zhenxun.models.ai_persona import AIPersona

    persona = None
    if payload.persona_id is not None:
        persona = await AIPersona.get_or_none(id=payload.persona_id)
        if persona is None:
            raise HTTPException(404, detail={"code": "persona_not_found"})
        if not persona.enabled:
            raise HTTPException(409, detail={"code": "persona_disabled"})
    subject = management_subject(request)
    session = (
        console_session(payload.persona_id)
        + ":"
        + hashlib.sha256(subject.encode()).hexdigest()
    )
    if (task := _tasks.get(session)) and not task.done():
        raise HTTPException(409, detail={"code": "console_busy"})
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    receipt: dict[str, Any] = {"event": asyncio.Event(), "run_id": uuid.uuid4().hex}
    from ....security import decode_access_token_status, validate_access_token

    credential = request.headers.get("Authorization", "")
    principal, status = decode_access_token_status(credential)
    if status != "valid" or not principal:
        raise HTTPException(401, detail={"code": "invalid_auth"})
    expires_at = float(principal.get("exp", time.time()))
    task = asyncio.create_task(
        _run(
            payload,
            persona,
            queue,
            receipt,
            lambda: validate_access_token(credential),
            str(principal["sub"]),
        )
    )
    _tasks[session] = task

    async def events() -> AsyncIterator[str]:
        try:
            while validate_access_token(credential):
                remaining = expires_at - time.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), min(remaining, 5))
                except asyncio.TimeoutError:
                    continue
                if item is None or not validate_access_token(credential):
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") == "done":
                    receipt["accepted"] = True
                    receipt["event"].set()
        finally:
            with anyio.CancelScope(shield=True):
                if not task.done():
                    task.cancel()
                await asyncio.wait({task}, timeout=10)
            if _tasks.get(session) is task:
                _tasks.pop(session, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/console/stop", dependencies=[authentication()], response_model=Result)
async def console_stop(payload: ConsoleStopRequest, request: Request) -> Result:
    from zhenxun.builtin_plugins.ai_chat._data_source import AIChatManager
    from zhenxun.services.ai.run.session import session_manager

    subject = management_subject(request)
    session = console_session(payload.persona_id)
    if payload.run_id:
        entry = session_manager.runs.get(payload.run_id)
        if entry and (
            entry[0].run.caller is None
            or entry[0].run.caller.source != "webui"
            or entry[0].run.caller.user_id != f"webui:{subject}"
        ):
            raise HTTPException(409, detail={"message": "运行不属于当前会话"})
        stopped = await session_manager.stop(payload.run_id)
        confirmed = (
            not entry or entry[0].run.operations.get("dsh", {}).get("stop") != "unknown"
        )
        return Result.ok(
            info="已停止。" if stopped and confirmed else "正在停止，结果待核验。"
        )
    task = _tasks.get(session + ":" + hashlib.sha256(subject.encode()).hexdigest())
    count = await AIChatManager.stop(session)
    if task and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        count += 1
    return Result.ok(info="已停止。" if count else "当前没有进行中的对话。")
