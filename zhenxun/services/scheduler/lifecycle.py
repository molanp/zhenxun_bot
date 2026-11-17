"""
定时任务的生命周期管理

包含在机器人启动时加载和调度数据库中保存的任务的逻辑。
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import time
from uuid import uuid4

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_MAX_INSTANCES,
    EVENT_JOB_MISSED,
)
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from nonebot_plugin_apscheduler import scheduler

from zhenxun.models.scheduled_job import ScheduledJob
from zhenxun.services.log import logger
from zhenxun.utils.manager.priority_manager import PriorityLifecycle
from zhenxun.utils.pydantic_compat import model_dump

from .engine import APSchedulerAdapter
from .manager import scheduler_manager
from .registry import scheduler_registry
from .repository import ScheduleRepository
from .types import JobConfig, ScheduleContext

_CORE_OBSERVER_REGISTERED = False
_CORE_SERVICE_MODULES = {
    "zhenxun.services.avatar_service",
    "zhenxun.services.ai.llm.manager",
}


def _worker_identity() -> str:
    """读取本轮 worker 的启动身份。"""
    from zhenxun.services.startup import startup_coordinator

    return startup_coordinator.boot_id


@asynccontextmanager
async def core_task_run(task: str):
    """记录本体任务执行摘要，调用者填入处理数与跳过原因。"""
    from zhenxun.utils.time_utils import TimeUtils

    started = time.monotonic()
    identity = f"task={task} boot={_worker_identity()} run={uuid4().hex[:12]}"
    result = {"status": "completed", "processed": 0}
    logger.info(
        f"{identity} phase=start at={TimeUtils.now().isoformat()}", "本体定时任务"
    )
    try:
        yield result
    except asyncio.CancelledError:
        result.update(status="cancelled", reason="shutdown_or_cancelled")
        raise
    except Exception as error:
        result.update(status="failed", reason=type(error).__name__)
        raise
    finally:
        failures = result.get("failed", result.get("errors", 0))
        if failures and result["status"] == "completed":
            result["status"] = "partial" if result["processed"] else "failed"
        if result["status"] == "completed" and not result["processed"]:
            if result.get("reason_counts", {}).get("no_bot"):
                result.update(status="skipped", reason="no_connected_bot")
            else:
                result.setdefault("reason", "no_work")
        logger.info(
            f"{identity} phase=end at={TimeUtils.now().isoformat()} "
            f"duration={time.monotonic() - started:.3f}s "
            f"result={json.dumps(result, ensure_ascii=False, sort_keys=True)}",
            "本体定时任务",
        )


def _is_core_job(job) -> bool:
    if not isinstance(getattr(job, "trigger", None), CronTrigger | IntervalTrigger):
        return False
    module = getattr(getattr(job, "func", None), "__module__", "")
    return module.startswith("zhenxun.builtin_plugins.") or (
        module in _CORE_SERVICE_MODULES
    )


def core_job_snapshot() -> list[dict]:
    """读取实际注册的本体任务，不把源码声明当作运行证据。"""
    return [
        {
            "id": str(job.id),
            "module": job.func.__module__,
            "function": job.func.__name__,
            "trigger": str(job.trigger),
            "timezone": str(getattr(job.trigger, "timezone", scheduler.timezone)),
            "max_instances": getattr(job, "max_instances", None),
            "coalesce": getattr(job, "coalesce", None),
            "misfire_grace_time": getattr(job, "misfire_grace_time", None),
            "next_run_time": (
                job.next_run_time.isoformat()
                if getattr(job, "next_run_time", None) is not None
                else None
            ),
            "pending_registration": job.pending,
        }
        for job in scheduler.get_jobs()
        if _is_core_job(job)
    ]


def _observe_core_job(event) -> None:
    job = scheduler.get_job(event.job_id)
    if job is None or not _is_core_job(job):
        return
    reason = {
        EVENT_JOB_ERROR: "execution_failed",
        EVENT_JOB_MISSED: "misfire",
        EVENT_JOB_MAX_INSTANCES: "max_instances",
    }.get(event.code)
    if reason is None:
        return
    scheduled = getattr(event, "scheduled_run_times", None) or [
        getattr(event, "scheduled_run_time", None)
    ]
    error = getattr(event, "exception", None)
    logger.warning(
        f"task={job.id} module={job.func.__module__} boot={_worker_identity()} "
        f"event={reason} scheduled={[str(value) for value in scheduled]} "
        f"error={type(error).__name__ if error else ''}",
        "本体定时任务",
    )


def _stop_core_observer() -> None:
    global _CORE_OBSERVER_REGISTERED
    if _CORE_OBSERVER_REGISTERED:
        scheduler.remove_listener(_observe_core_job)
        _CORE_OBSERVER_REGISTERED = False


def _start_core_observer(context) -> None:
    global _CORE_OBSERVER_REGISTERED
    if _CORE_OBSERVER_REGISTERED:
        return
    scheduler.add_listener(
        _observe_core_job, EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES
    )
    _CORE_OBSERVER_REGISTERED = True
    context.add_finalizer(_stop_core_observer)
    logger.info(
        f"boot={_worker_identity()} registered_core_jobs="
        + json.dumps(core_job_snapshot(), ensure_ascii=False, sort_keys=True),
        "本体定时任务",
    )


def _managed_jobs():
    return [
        job
        for job in scheduler.get_jobs()
        if str(job.id).startswith("zhenxun_schedule_")
        or str(job.id).startswith("runtime::")
    ]


async def _drain_scheduler_jobs() -> None:
    """暂停托管任务并等待正在执行的任务结束，最多 10 秒"""
    for job in _managed_jobs():
        with contextlib.suppress(Exception):
            job.pause()
    deadline = asyncio.get_running_loop().time() + 10
    for schedule_id, handles in list(scheduler_registry.running_handles.items()):
        schedule = await ScheduleRepository.get_by_id(schedule_id)
        if schedule and scheduler_registry.tasks.get(schedule.plugin_name, {}).get(
            "retain_finished"
        ):
            for handle in handles:
                handle.cancel()
    while scheduler_registry.running_tasks:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("scheduler_drain_timeout")
        await asyncio.sleep(0.05)


def _remove_scheduler_jobs() -> None:
    """移除全部托管任务"""
    for job in _managed_jobs():
        with contextlib.suppress(Exception):
            scheduler.remove_job(job.id)


@PriorityLifecycle.on_startup(
    priority=90,
    component_id="runtime:scheduler",
    depends_on=("runtime:runtime_cache",),
    pass_context=True,
)
async def _load_schedules_from_db(context):
    """在服务启动时从数据库加载并调度所有任务。"""
    _start_core_observer(context)
    logger.info("正在从数据库加载并调度所有定时任务...")
    context.add_finalizer(_remove_scheduler_jobs)
    context.add_finalizer(_drain_scheduler_jobs)
    all_schedules = await ScheduleRepository.get_all()
    schedules = [schedule for schedule in all_schedules if schedule.is_enabled]
    count = 0
    for schedule in schedules:
        if schedule.plugin_name in scheduler_registry.tasks:
            policy = scheduler_registry.tasks[schedule.plugin_name]
            if policy.get("retain_finished") and schedule.last_run_status == "RUNNING":
                await ScheduleRepository.update_run_status(
                    schedule, status="INTERRUPTED", disable=schedule.is_one_off
                )
                if schedule.is_one_off:
                    continue
            if APSchedulerAdapter.add_or_reschedule_job(schedule, catch_up=True):
                count += 1
            elif schedule.is_one_off:
                if policy.get("retain_finished"):
                    await ScheduleRepository.update_run_status(
                        schedule,
                        status=schedule.last_run_status or "MISSED",
                        disable=True,
                    )
                else:
                    await ScheduledJob.filter(id=schedule.id).delete()
                    logger.info(f"一次性任务 {schedule.id} 已错过执行时间，已删除")
        else:
            logger.warning(f"跳过加载定时任务：插件 '{schedule.plugin_name}' 未注册。")
    logger.info(f"数据库定时任务加载完成，共成功加载 {count} 个任务。")

    logger.info("正在检查并注册声明式默认任务...")
    declared_count = 0
    existing_declarations: set[tuple[str, str, str | None]] = {
        (
            schedule.plugin_name,
            schedule.target_identifier,
            schedule.bot_id,
        )
        for schedule in all_schedules
    }
    for task_info in scheduler_registry.persistent_declarations:
        plugin_name = task_info.plugin_name
        group_id = task_info.group_id
        bot_id = task_info.bot_id

        declaration_key = (plugin_name, group_id or "", bot_id)
        if declaration_key not in existing_declarations:
            logger.info(f"为插件 '{plugin_name}' 注册新的默认定时任务...")

            target_type = "GROUP" if group_id else "GLOBAL"
            target_identifier = group_id or ""

            config = JobConfig(
                trigger=task_info.trigger,
                job_kwargs=task_info.job_kwargs,
                bot_id=bot_id,
                source="PLUGIN_DEFAULT",
            )

            schedule = await scheduler_manager.add_schedule(
                plugin_name=plugin_name,
                target_type=target_type,
                target_identifier=target_identifier,
                config=config,
            )
            if schedule:
                declared_count += 1
                existing_declarations.add(declaration_key)
                logger.debug(f"默认任务 '{plugin_name}' 注册成功 (ID: {schedule.id})")
            else:
                logger.error(f"默认任务 '{plugin_name}' 注册失败")
        else:
            logger.debug(f"插件 '{plugin_name}' 的默认任务已存在于数据库中，跳过注册。")

    if declared_count > 0:
        logger.info(f"声明式任务检查完成，新注册了 {declared_count} 个默认任务。")

    logger.info("正在调度声明式临时任务...")
    ephemeral_count = 0
    for declaration in scheduler_registry.ephemeral_declarations:
        try:
            job_id = f"runtime::{declaration.plugin_name}::{declaration.func.__name__}"

            context = ScheduleContext(
                schedule_id=0,
                plugin_name=job_id,
                bot_id=None,
                platform_scope=None,
                group_id=None,
                job_kwargs={},
            )

            trigger_config_dict = model_dump(
                declaration.trigger, exclude={"trigger_type"}
            )

            APSchedulerAdapter.add_ephemeral_job(
                job_id=job_id,
                func=declaration.func,
                trigger_type=declaration.trigger.trigger_type,
                trigger_config=trigger_config_dict,
                context=context,
            )
            ephemeral_count += 1
        except Exception as e:
            logger.error(f"调度临时任务 '{declaration.plugin_name}' 失败", e=e)

    if ephemeral_count > 0:
        logger.info(f"临时任务调度完成，共成功加载 {ephemeral_count} 个任务。")
