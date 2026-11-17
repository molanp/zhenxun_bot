"""内置 AI 聊天的主动对话：主动插话的按群设置、运行状态与主动任务"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
import nonebot
from pydantic import BaseModel, Field

from zhenxun.services.log import logger

from ....base_model import Result
from ....utils import authentication

router = APIRouter()


class TaskPayload(BaseModel):
    name: str = Field(default="", max_length=64)
    prompt: str = Field(min_length=1, max_length=4000)
    kind: Literal["cron", "interval", "date"]
    value: str = Field(min_length=1, max_length=64)
    bot_id: str | None = Field(default=None, max_length=255)
    group_id: str | None = Field(default=None, max_length=255)
    user_id: str | None = Field(default=None, max_length=255)


def _state_item(bot_id: str, group_id: str, st: Any, cooldown: int) -> dict[str, Any]:
    now = time.time()
    left = st.last_proactive_at + cooldown - now if st.last_proactive_at else 0
    return {
        "bot_id": bot_id,
        "group_id": group_id,
        "last_message_at": st.last_human_at or None,
        "last_proactive_at": st.last_proactive_at or None,
        "last_judge_at": st.last_judge_at or None,
        "pending": st.pending,
        "unanswered": st.unanswered,
        "cooldown_left": max(0, int(left)),
    }


@router.get("/chat/proactive", dependencies=[authentication()], response_model=Result)
async def proactive_state(
    group_ids: list[str] | None = Query(default=None, max_length=50),
) -> Result:
    from zhenxun.builtin_plugins.ai_chat._proactive import Proactive
    from zhenxun.builtin_plugins.ai_chat._steal import Steal

    selected = set(group_ids) if group_ids is not None else None
    _, total = await Proactive.used_tokens("")
    _, judges = await Steal.today("")
    states = []
    summaries = {}
    for (bot_id, group_id), st in list(Proactive._states.items()):
        if selected is not None and group_id not in selected:
            continue
        if group_id not in summaries:
            cfg = await Proactive.settings(group_id)
            group_used, _ = await Proactive.used_tokens(group_id)
            stolen, _ = await Steal.today(group_id)
            count = await Proactive.count_today(group_id)
            summaries[group_id] = cfg, group_used, stolen, count
        cfg, group_used, stolen, count = summaries[group_id]
        item = _state_item(bot_id, group_id, st, cfg.cooldown if cfg else 0)
        states.append(
            {
                **item,
                "mode": cfg.mode if cfg else "off",
                "used_tokens": group_used,
                "count": count,
                "stolen": stolen,
            }
        )
    return Result.ok({"used_tokens": total, "steal_judges": judges, "groups": states})


@router.put(
    "/chat/groups/{group_id}/proactive",
    dependencies=[authentication()],
    response_model=Result,
)
async def update_group_proactive(group_id: str, payload: dict[str, Any]) -> Result:
    from zhenxun.builtin_plugins.ai_chat._proactive import ProactiveConfig
    from zhenxun.models.ai_chat_group import AIChatGroup

    try:
        config = ProactiveConfig.model_validate(payload)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e
    data = config.model_dump(exclude_none=True)
    await AIChatGroup.save_setting(group_id, proactive=data or None)
    logger.info(f"WebUI 修改了群 {group_id} 的主动对话设置", "WebUi")
    return Result.ok(info="主动对话设置已保存。")


def _task_error(error: ValueError) -> HTTPException:
    return HTTPException(422, detail={"code": "invalid_task", "message": str(error)})


async def _last_runs() -> dict[str, Any]:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks
    from zhenxun.models.ai_agent_run import AIAgentRun

    runs: dict[str, Any] = {}
    for job in await AITasks.all():
        run = (
            await AIAgentRun.filter(
                source="task", session_id__endswith=f":task:{job.id}"
            )
            .order_by("-id")
            .first()
        )
        if run:
            runs[str(job.id)] = {
                "status": run.status,
                "output": run.output or run.error,
                "at": run.created_at.isoformat(),
            }
    return runs


@router.get("/chat/tasks", dependencies=[authentication()], response_model=Result)
async def list_tasks() -> Result:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks
    from zhenxun.services.ai.run.session import session_manager

    active = {
        str(getattr(context.deps, "schedule_id", "")): {
            "run_id": run_id,
            "phase": context.run.phase,
            "parent_run_id": context.run.parent_run_id,
        }
        for run_id, (context, _) in session_manager.runs.items()
        if not context.run.parent_run_id
    }

    runs = await _last_runs()
    return Result.ok(
        {
            "tasks": [
                {
                    **AITasks.item(job),
                    "last_run": runs.get(str(job.id)),
                    "active_run": active.get(str(job.id)),
                }
                for job in await AITasks.all()
            ],
            "bots": list(nonebot.get_bots()),
        }
    )


@router.post("/chat/tasks", dependencies=[authentication()], response_model=Result)
async def create_task(payload: TaskPayload, request: Request) -> Result:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks, TaskParams

    bot_id = payload.bot_id or next(iter(nonebot.get_bots()), None)
    if not bot_id:
        raise HTTPException(
            422, detail={"code": "no_bot", "message": "没有在线的 Bot。"}
        )
    from ....security import decode_access_token_status

    principal, status = decode_access_token_status(
        request.headers.get("Authorization", "")
    )
    if status != "valid" or not principal:
        raise HTTPException(401, detail={"code": "invalid_auth"})
    subject = str(principal["sub"])
    params = TaskParams(
        prompt=payload.prompt.strip(),
        creator=f"webui:{subject}",
        source="management",
        management_subject=subject,
        group_id=(payload.group_id or "").strip() or None,
        user_id=None if payload.group_id else (payload.user_id or "").strip() or None,
    )
    try:
        job = await AITasks.create(
            bot_id, params, payload.name, payload.kind, payload.value
        )
    except ValueError as e:
        raise _task_error(e) from e
    logger.info(f"WebUI 创建了主动任务 #{job.id}", "WebUi")
    return Result.ok(AITasks.item(job), info="主动任务已创建。")


async def _require_task(task_id: int) -> Any:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks

    job = await AITasks.get(task_id)
    if job is None:
        raise HTTPException(
            404, detail={"code": "task_not_found", "message": "任务不存在。"}
        )
    return job


@router.put(
    "/chat/tasks/{task_id}", dependencies=[authentication()], response_model=Result
)
async def update_task(task_id: int, payload: TaskPayload) -> Result:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks

    job = await _require_task(task_id)
    try:
        await AITasks.update(
            job, payload.name, payload.prompt, payload.kind, payload.value
        )
    except ValueError as e:
        raise _task_error(e) from e
    return Result.ok(AITasks.item(job), info="主动任务已保存。")


@router.delete(
    "/chat/tasks/{task_id}", dependencies=[authentication()], response_model=Result
)
async def delete_task(task_id: int) -> Result:
    from zhenxun.builtin_plugins.ai_chat._tasks import AITasks

    await _require_task(task_id)
    try:
        await AITasks.remove(task_id)
    except ValueError as error:
        raise _task_error(error) from error
    logger.info(f"WebUI 删除了主动任务 #{task_id}", "WebUi")
    return Result.ok(info="主动任务已删除。")


@router.post(
    "/chat/tasks/{task_id}/{action}",
    dependencies=[authentication()],
    response_model=Result,
)
async def task_action(
    task_id: int,
    action: Literal["run", "pause", "resume", "stop"],
    run_id: str | None = None,
) -> Result:
    from zhenxun.services.scheduler import scheduler_manager

    await _require_task(task_id)
    if action == "stop":
        from zhenxun.services.ai.run.session import session_manager

        entry = session_manager.runs.get(run_id or "")
        if not run_id:
            raise HTTPException(422, detail={"message": "需要指定本次运行标识"})
        if entry and str(getattr(entry[0].deps, "schedule_id", "")) != str(task_id):
            raise HTTPException(409, detail={"message": "运行不属于此计划"})
        stopped = await session_manager.stop(run_id)
        return Result.ok(
            info="本次运行已停止。" if stopped else "正在停止，执行结果待核验。"
        )
    handler = {
        "run": scheduler_manager.trigger_now,
        "pause": scheduler_manager.pause_schedule,
        "resume": scheduler_manager.resume_schedule,
    }[action]
    ok, message = await handler(task_id)
    if not ok:
        raise HTTPException(
            409, detail={"code": "task_action_failed", "message": message}
        )
    return Result.ok(info=message)
