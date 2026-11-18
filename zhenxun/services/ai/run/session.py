import asyncio
from contextlib import asynccontextmanager
import time
from typing import Any
import uuid

from zhenxun.services.ai.core.exceptions import (
    InterventionHandledException,
    LLMException,
)
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.utils.logger import log_flow as logger


class AgentSessionManager:
    """
    Agent 会话运行管理器。
    登记会话中运行或排队的任务，维护同一会话的执行锁与追加消息队列；
    会话没有运行中的任务时立即释放这些记录。
    """

    def __init__(self):
        self.live_tasks: dict[
            str, list[tuple[CancellationToken, asyncio.Task | None]]
        ] = {}
        self._revisions: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._follow_ups: dict[str, list[Any]] = {}
        self._owners: dict[str, str | None] = {}
        self._admissions: dict[str, Any] = {}
        self._in_tools: set[str] = set()
        self.runs: dict[str, tuple[Any, asyncio.Task | None]] = {}
        self.preparations: dict[str, set[asyncio.Task]] = {}
        self._preparation_cancels: set[asyncio.Task] = set()
        self.boot_id = uuid.uuid4().hex
        self.closing = False

    def own_preparations(self, context, tasks: list[asyncio.Task]) -> None:
        """登记本轮准备协程，完成后回收引用并取走异常。"""
        if not tasks:
            return
        run_id = context.run.run_id
        owned = self.preparations.setdefault(run_id, set())
        owned.update(tasks)

        def completed(task):
            owned.discard(task)
            self._preparation_cancels.discard(task)
            if not task.cancelled():
                task.exception()
            if not owned:
                self.preparations.pop(run_id, None)
                entry = self.runs.get(run_id)
                if entry and entry[1] and entry[1].done():
                    self.runs.pop(run_id, None)
                    entry[0].run.cancellation_token.detach()

        for task in tasks:
            task.add_done_callback(completed)

    def cleanup_remaining(self, context) -> float:
        """同根运行的所有收尾共享一个十秒窗口。"""
        deadline = context.run.operations.setdefault(
            "cleanup_deadline", time.monotonic() + 10
        )
        return max(0.0, deadline - time.monotonic())

    async def drain_preparations(self, context, tasks=None) -> bool:
        """取消并等待拥有的准备协程；未退出的任务保留登记。"""
        owned = set(
            tasks
            if tasks is not None
            else self.preparations.get(context.run.run_id, ())
        )
        owned.discard(asyncio.current_task())
        owned = {task for task in owned if not task.done()}
        if not owned:
            return True
        for task in owned:
            if task not in self._preparation_cancels:
                self._preparation_cancels.add(task)
                task.cancel()
        waiter = asyncio.create_task(
            asyncio.wait(owned, timeout=self.cleanup_remaining(context))
        )
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                continue
        _, pending = waiter.result()
        if pending:
            context.run.outcome = "unknown"
            context.state["_preparation"] = {
                "status": "unknown",
                "pending": len(pending),
                "reason": "cleanup_unconfirmed",
            }
        return not pending

    def register(self, context, task: asyncio.Task | None = None) -> str:
        """登记已接纳的运行及来源，排队阶段也属于可取消范围。"""
        if self.closing:
            raise asyncio.CancelledError("Bot 正在关闭，不再接纳运行")
        run = context.run
        if run.cancellation_token is None:
            run.cancellation_token = CancellationToken()
        previous = self.runs.get(run.run_id)
        owner = task or (previous[1] if previous else asyncio.current_task())
        if previous and previous[1] is not owner and previous[1] is not None:
            run.cancellation_token._unlink(previous[1])
        self.runs[run.run_id] = (context, owner)
        if owner is not None:
            run.cancellation_token.link_future(owner)
        return run.run_id

    async def accept(self, context, text: str = "") -> None:
        """耗时执行前保存运行占位，不把记录作为重启恢复队列。"""
        from zhenxun.models.ai_agent_run import AIAgentRun

        self.register(context)
        if context.run.source:
            await AIAgentRun.get_or_create(
                run_id=context.run.run_id,
                defaults={
                    "source": context.run.source,
                    "session_id": context.session_id or "",
                    "user_id": context.get_user_id() or "",
                    "group_id": context.get_group_id(),
                    "status": "queued",
                    "details": self.details(context),
                },
            )

    def details(self, context) -> dict:
        """返回不含凭据的生命周期证据。"""
        return {
            "boot_id": self.boot_id,
            "phase": context.run.phase,
            "parent_run_id": context.run.parent_run_id,
            "connection_id": context.run.connection_id,
            "outcome": context.run.outcome,
            "interruption": context.run.operations.get("interruption"),
            "stop_reason": context.run.cancellation_token.reason
            if context.run.cancellation_token
            else None,
            "children": list(context.run.operations.get("children", {}).values()),
            "external_stop_guarantee": context.run.operations.get(
                "external_stop_guarantee"
            ),
            "dsh": context.run.operations.get("dsh"),
            "dsh_tools": context.run.operations.get("dsh_tools"),
            "preparation": context.state.get("_preparation"),
            "configuration": context.state.get("_configuration"),
            "injections": context.state.get("_injections"),
            "reported_cache_read_tokens": context.run.budget.get("reported_cache_read"),
            "background": context.state.get("_background_receipt"),
        }

    async def phase(self, context, phase: str) -> None:
        """更新执行阶段，终态由收尾入口统一结算。"""
        from zhenxun.models.ai_agent_run import AIAgentRun

        context.run.phase = phase
        if context.run.source:
            await AIAgentRun.filter(
                run_id=context.run.run_id,
                status__in=["queued", "running", "delivering", "settling", "stopping"],
            ).update(status=phase, details=self.details(context))

    def interrupt_siblings(self, context, reason: str) -> None:
        """通信失败取消同根兄弟；异常沿当前父调用栈向上传递。"""
        ancestors = {context.run.run_id}
        parent = context.run.parent_run_id
        while parent and parent in self.runs:
            ancestors.add(parent)
            parent = self.runs[parent][0].run.parent_run_id
        for run_id, (other, _) in list(self.runs.items()):
            if (
                other.run.operations is context.run.operations
                and run_id not in ancestors
            ):
                other.run.cancellation_token.cancel(reason)

    async def stop(
        self, run_id: str, reason: str = "cancelled", timeout: float = 10
    ) -> bool:
        """停止指定运行并等待收尾；超时只报告停止尚未确认。"""
        entry = self.runs.get(run_id)
        if entry is None:
            return True
        context, task = entry
        context.run.phase = "stopping"
        remaining = min(timeout, self.cleanup_remaining(context))
        context.run.cancellation_token.cancel(reason)
        if task is None or task is asyncio.current_task():
            return False
        _, pending = await asyncio.wait({task}, timeout=remaining)
        cleaned = await self.drain_preparations(context)
        return not pending and cleaned

    async def stop_source(self, bot, connection_id: str | None = None) -> None:
        """仅中断失效连接所属账号与代际的运行。"""
        ids = [
            run_id
            for run_id, (context, _) in list(self.runs.items())
            if context.get_bot() is bot
            and (connection_id is None or context.run.connection_id == connection_id)
        ]
        await asyncio.gather(*(self.stop(run_id, "disconnected") for run_id in ids))

    async def finish(self, context) -> None:
        """根运行退出前等待所有登记子运行停止，再释放登记。"""
        await self.drain_preparations(context)
        children = [
            run_id
            for run_id, (child, _) in list(self.runs.items())
            if child.run.parent_run_id == context.run.run_id
        ]
        if children:
            stopped = await asyncio.gather(
                *(self.stop(run_id, "parent_finished") for run_id in children)
            )
            if not all(stopped):
                context.run.outcome = "unknown"
        if context.run.source:
            from zhenxun.models.ai_agent_run import AIAgentRun

            active = AIAgentRun.filter(
                run_id=context.run.run_id,
                status__in=["queued", "running", "delivering", "settling", "stopping"],
            )
            record = await active.first()
            if record:
                details = record.details or {}
                status = (
                    details.get("execution_status")
                    or context.run.outcome
                    or "interrupted"
                )
                if context.run.outcome == "unknown":
                    status = "unknown"
                await active.update(
                    status=status,
                    details={**details, **self.details(context), "phase": "finished"},
                )
        if context.run.parent_run_id:
            context.run.operations.setdefault("children", {})[context.run.run_id] = {
                "run_id": context.run.run_id,
                "parent_run_id": context.run.parent_run_id,
                "outcome": context.run.outcome or "interrupted",
            }
        if not self.preparations.get(context.run.run_id):
            self.runs.pop(context.run.run_id, None)
            if context.run.cancellation_token:
                context.run.cancellation_token.detach()

    async def start(self) -> None:
        """旧启动身份的未结束记录只标记中断，不执行任何工具或消息。"""
        from zhenxun.models.ai_agent_run import AIAgentRun

        self.closing = False
        active = ["queued", "running", "delivering", "settling", "stopping"]
        for record in await AIAgentRun.filter(status__in=active):
            details = record.details or {}
            if details.get("boot_id") != self.boot_id:
                await AIAgentRun.filter(id=record.id, status__in=active).update(
                    status="interrupted",
                    details={
                        **details,
                        "outcome": "interrupted",
                        "interruption": "previous_boot",
                        "external_result": "unknown",
                    },
                )

    async def shutdown(self) -> list[str]:
        """关闭准入，给所有当前运行最多十秒清理窗口。"""
        self.closing = True
        entries = list(self.runs.items())
        results = await asyncio.gather(
            *(self.stop(run_id, "shutdown") for run_id, _ in entries)
        )
        unknown = [
            run_id for (run_id, _), stopped in zip(entries, results) if not stopped
        ]
        if unknown:
            from zhenxun.models.ai_agent_run import AIAgentRun

            for run_id in unknown:
                context = self.runs[run_id][0]
                context.run.outcome = "unknown"
                await AIAgentRun.filter(run_id=run_id).update(
                    status="unknown",
                    details={**self.details(context), "cleanup": "unconfirmed"},
                )
        return unknown

    def revision(self, session_id: str) -> int:
        """会话当前的修订号，会话被重置时递增

        参数:
            session_id: 会话id

        返回:
            int: 修订号
        """
        return self._revisions.get(session_id, 0)

    def invalidate(self, session_id: str) -> int:
        """递增会话修订号并丢弃待并入的输入，此前开始的运行不再写回历史

        参数:
            session_id: 会话id

        返回:
            int: 新的修订号
        """
        revision = self.revision(session_id) + 1
        self._revisions[session_id] = revision
        self._release_follow_ups(session_id)
        return revision

    def drain_follow_ups(self, session_id: str) -> list[tuple[Any, asyncio.Future]]:
        """交接待并入的输入及消费回执，消费方负责结算每张回执。"""
        return self._follow_ups.pop(session_id, [])

    def _release_follow_ups(self, session_id: str) -> None:
        for _, receipt in self.drain_follow_ups(session_id):
            if not receipt.done():
                receipt.set_result(False)

    def update_admission(self, session_id: str, admission: Any) -> None:
        """在已取得执行权的窗口内更新追加输入的兼容条件。"""
        if session_id in self._admissions:
            self._admissions[session_id] = admission

    def set_tools_running(self, session_id: str, running: bool) -> None:
        """标记会话当前运行是否正在执行工具，执行期间同一发送者的新消息可以并入

        参数:
            session_id: 会话 ID
            running: 是否正在执行工具
        """
        if running:
            self._in_tools.add(session_id)
        else:
            self._in_tools.discard(session_id)

    @asynccontextmanager
    async def run_scope(
        self,
        session_id: str,
        cancel_token: CancellationToken,
        exclusive: bool,
        follow_up: Any = None,
        sender: str | None = None,
        revision: int | None = None,
        admission: Any = None,
    ):
        """登记一次运行；独占运行时同一会话依次执行。会话正在执行工具，且新输入
        与当前运行来自同一发送者时，把输入并入当前运行，否则排队等待

        参数:
            session_id: 会话 ID
            cancel_token: 本次运行的取消令牌
            exclusive: 是否与同一会话的其他独占运行互斥
            follow_up: 可以并入当前运行的用户输入，为 None 时总是排队
            sender: 发送者 ID
            revision: 发起时的会话修订号，取得锁时会话已被重置则不再运行

        异常:
            InterventionHandledException: 输入已并入正在运行的任务
        """
        entry = (cancel_token, asyncio.current_task())
        tasks = self.live_tasks.setdefault(session_id, [])
        tasks.append(entry)
        try:
            if not exclusive:
                yield
                return
            lock = self._locks.setdefault(session_id, asyncio.Lock())
            if lock.locked():
                if (
                    follow_up is not None
                    and sender
                    and session_id in self._in_tools
                    and self._owners.get(session_id) == sender
                    and admission is not None
                    and self._admissions.get(session_id) == admission
                    and (revision is None or revision == self.revision(session_id))
                ):
                    receipt = asyncio.get_running_loop().create_future()
                    self._follow_ups.setdefault(session_id, []).append(
                        (follow_up, receipt)
                    )
                    if await receipt:
                        raise InterventionHandledException("Follow-up successful")
                    cancel_token.raise_if_cancelled()
                    raise LLMException(
                        "上一轮已结束，追加消息未被处理，请重新发送。",
                        details={"code": "follow_up_unconsumed"},
                    )
                logger.info(f"⏳ [并发控制] 会话 {session_id} 正忙，新请求排队等待...")
            async with lock:
                cancel_token.raise_if_cancelled()
                if revision is not None and revision != self.revision(session_id):
                    raise asyncio.CancelledError("会话已重置")
                self._owners[session_id] = sender
                self._admissions[session_id] = admission
                try:
                    yield
                finally:
                    self._owners.pop(session_id, None)
                    self._admissions.pop(session_id, None)
                    self._in_tools.discard(session_id)
                    self._release_follow_ups(session_id)
        finally:
            tasks.remove(entry)
            if not tasks:
                self.live_tasks.pop(session_id, None)
                self._locks.pop(session_id, None)


session_manager = AgentSessionManager()
