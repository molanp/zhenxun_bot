import asyncio
from collections.abc import Awaitable, Callable

from arclet.alconna import Alconna, Option
from nonebot.adapters import Bot
from pydantic import BaseModel

from zhenxun.services.log import logger

from .types import EphemeralJobDeclaration, ScheduledJobDeclaration


class SchedulerRegistry:
    """定时任务注册中心，统一管理所有任务的元数据和解析器"""

    ALL_GROUPS = "__ALL_GROUPS__"

    def __init__(self):
        """初始化调度任务注册中心容器。"""
        self.tasks: dict[
            str,
            dict[str, Callable | type[BaseModel] | int | list[Option] | Alconna | None],
        ] = {}
        self.persistent_declarations: list[ScheduledJobDeclaration] = []
        self.ephemeral_declarations: list[EphemeralJobDeclaration] = []
        self.target_resolvers: dict[
            str, Callable[[str, Bot], Awaitable[list[str | None]]]
        ] = {}
        self.running_tasks: set[int] = set()
        self.running_handles: dict[int, set[asyncio.Task]] = {}

    async def stop_running(self, schedule_id: int) -> bool:
        """取消计划的当前执行并等待清理，不修改未来触发定义。"""
        tasks = set(self.running_handles.get(schedule_id, ()))
        if asyncio.current_task() in tasks:
            return False
        for task in tasks:
            task.cancel()
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=10)
        return not pending

    def register_target_resolver(
        self,
        target_type: str,
        resolver_func: Callable[[str, Bot], Awaitable[list[str | None]]],
    ):
        """注册指定执行目标的解析策略。"""
        if target_type in self.target_resolvers:
            logger.warning(f"目标解析器 '{target_type}' 已存在，将被覆盖。")
        self.target_resolvers[target_type.upper()] = resolver_func
        logger.info(f"已注册新的定时任务目标解析器: '{target_type}'")


scheduler_registry = SchedulerRegistry()
