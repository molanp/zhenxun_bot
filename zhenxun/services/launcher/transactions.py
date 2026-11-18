"""launcher 侧的插件事务：重启前应用、worker 启动后验证或回滚"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from zhenxun.services.nonebot_store import runtime as nonebot_runtime
from zhenxun.services.nonebot_store import storage as nonebot_storage
from zhenxun.services.plugin_store import plugin_store_transaction as source_store
from zhenxun.utils.restart_state import (
    clear_pending_restart_state,
    read_restart_state,
)

from .maintenance import run as run_maintenance
from .supervisor import launcher_log


class PluginTransactions:
    """NoneBot 依赖层与真寻插件源码事务"""

    def __init__(self) -> None:
        self.verify_nonebot = False
        self.verify_source = False
        self.source_pending: dict[str, Any] | None = None
        self.restart_markers: dict[str, Any] = {}

    def settle_restart_markers(self, sources: list[str]) -> None:
        """仅核销本轮已确认事务对应且没有被后续写入替换的提示。"""
        for source in sources:
            expected = self.restart_markers.get(source)
            if expected is not None:
                try:
                    clear_pending_restart_state(source, expected=expected)
                except Exception as error:
                    launcher_log(
                        f"restart marker settlement pending: {type(error).__name__}"
                    )

    def settle_source_restart_markers(self) -> None:
        """依据原事务的操作收据核销已提交或已回滚的源码提示。"""
        for operation in (self.source_pending or {}).get("operations", []):
            value = {"operation_id": operation.get("operation_id")}
            try:
                if source_store.reconcile_operation(value):
                    self.settle_restart_markers(
                        [f"webui.plugin:{operation.get('store_key', '')}"]
                    )
            except Exception as error:
                launcher_log(
                    f"source restart receipt unavailable: {type(error).__name__}"
                )

    @property
    def needs_verification(self) -> bool:
        return self.verify_nonebot or self.verify_source

    async def apply(self) -> None:
        """在启动 worker 前应用待处理事务，并记录需要验证的部分

        解释器或核心虚拟环境变化后先按当前环境重新同步依赖层；已确认回滚的失败
        依赖层事务随后归档，不再阻挡新的依赖操作。
        """
        from zhenxun.services.runtime_mutation import mutation_owner

        try:
            self.restart_markers = read_restart_state().get("pending_restarts", {})
        except Exception:
            self.restart_markers = {}

        parent = mutation_owner()
        if (
            not parent
            and nonebot_runtime.environment_resync_required()
            and await run_maintenance("nonebot-resync")
        ):
            launcher_log("解释器或核心依赖已变化，依赖层将按当前环境重新同步")
        source_before = source_store.pending_transaction()
        dependencies_ready = source_store.prepare_dependency_transaction()
        nonebot_before = nonebot_storage.pending_transaction()
        nonebot_applied = False
        if dependencies_ready and nonebot_storage.pending_transaction():
            nonebot_applied = await run_maintenance("nonebot-apply")
        nonebot_after = nonebot_storage.pending_transaction()
        nonebot_failed = bool(
            nonebot_before
            and nonebot_before.get("state") == "pending_restart"
            and nonebot_after
            and nonebot_after.get("state")
            in {"failed", "migration_blocked", "recovery_required"}
        )
        if nonebot_failed and source_before:
            source_store.fail_pending_transaction(
                (nonebot_after or {}).get("error_code")
                or "dependency_layer_build_failed",
                (nonebot_after or {}).get("error_details"),
            )
        source_applied = (
            await run_maintenance("source-apply")
            if dependencies_ready
            and not nonebot_failed
            and not nonebot_runtime.recovery_required()
            and source_before
            else False
        )
        source_after = source_store.pending_transaction()
        source_failed = bool(
            source_before
            and source_before.get("state") == "pending_restart"
            and source_after
            and source_after.get("state") == "failed"
        )
        if source_applied:
            launcher_log("真寻插件源码事务已应用，等待 worker 启动验证")
        if source_failed and nonebot_applied:
            rolled_back = nonebot_runtime.rollback_for_launcher()
            nonebot_applied = False
            launcher_log(
                "真寻插件源码事务应用失败；依赖回滚"
                + ("已确认" if rolled_back else "待处理")
            )
        if nonebot_applied:
            launcher_log("NoneBot 插件依赖层已构建，等待 worker 启动验证")
        recovery = nonebot_runtime.recovery_required()
        self.source_pending = source_store.pending_transaction()
        self.verify_source = bool(
            self.source_pending
            and self.source_pending.get("state") == "verification_pending"
            and not recovery
        )
        self.verify_nonebot = bool(
            nonebot_storage.load_manifest().get("pending_verification") and not recovery
        )
        archived = nonebot_storage.archive_settled()
        if archived:
            self.settle_restart_markers(["webui.nonebot-store"])
        if source_after and source_after.get("rollback_confirmed"):
            self.settle_restart_markers(
                [
                    f"webui.plugin:{item.get('store_key', '')}"
                    for item in source_after.get("operations", [])
                ]
            )

    def failures(self, ready: bool) -> list[dict[str, Any]] | None:
        """检查 worker 启动后的插件加载结果，依赖层只有告警时视为通过

        参数:
            ready: worker 是否在限定时间内完成 warmup

        返回:
            list[dict[str, Any]] | None: 失败项，验证通过时为 None
        """
        if not ready:
            launcher_log("managed plugin worker failed, rolling back transaction")
            return [{"code": "worker_health_check_failed"}]
        nonebot_ok, nonebot_result = (
            nonebot_runtime.startup_verification()
            if self.verify_nonebot
            else (True, {})
        )
        source_ok, source_result = (
            source_store.startup_verification() if self.verify_source else (True, {})
        )
        if nonebot_ok and source_ok:
            child = nonebot_storage.pending_transaction() or {}
            if child.get("parent_operation_id") and nonebot_result.get("failed"):
                return list(nonebot_result["failed"])
            return None
        failed = [
            *(nonebot_result.get("failures") or []),
            *(source_result.get("failed") or []),
        ]
        launcher_log(
            "managed plugin verification failed, rolling back transaction: "
            + ", ".join(
                str(item.get("store_key") or item.get("code") or "unknown")
                for item in failed
            )
        )
        return failed

    def rollback(self, failed: list[dict[str, Any]]) -> None:
        """回滚未通过验证的事务并归档已确认回滚的依赖层事务（worker 必须已停止）

        参数:
            failed: 失败项
        """
        rolled_back = not self.verify_nonebot or nonebot_runtime.rollback_for_launcher(
            failed
        )
        if self.verify_source and rolled_back:
            source_store.rollback_pending_transaction(failed)
            self.settle_source_restart_markers()
        if nonebot_storage.archive_settled():
            self.settle_restart_markers(["webui.nonebot-store"])

    async def settle(
        self, action: Callable[[], None], label: str, *, attempts: int = 60
    ) -> bool:
        """执行事务收尾，其他变更占用时等待重试；仍未完成时保留事务并继续运行

        参数:
            action: 确认或回滚
            label: 日志中的动作名称
            attempts: 遇到占用时的尝试次数，每次间隔一秒

        返回:
            bool: 是否完成
        """
        from zhenxun.services.runtime_mutation import (
            RuntimeMutationBusyError,
            clear_stale_runtime_owner,
        )

        error: Exception | None = None
        for index in range(attempts):
            try:
                clear_stale_runtime_owner()
                action()
                return True
            except RuntimeMutationBusyError as busy:
                error = busy
                if index + 1 < attempts:
                    await asyncio.sleep(1)
            except RuntimeError as failure:
                error = failure
                break
        launcher_log(
            f"插件事务{label}未完成（{error}），事务已保留，"
            "可在 WebUI 插件管理中查看并处理"
        )
        return False

    def finalize(self) -> None:
        """确认已通过验证的事务并清除对应的待重启项"""
        if self.verify_nonebot:
            original = nonebot_storage.pending_transaction() or {}
            nonebot_runtime.finalize_pending_transaction()
            receipts = [
                nonebot_storage.pending_transaction(),
                *nonebot_storage.settled_transactions(),
            ]
            if any(
                receipt
                and receipt.get("revision") == original.get("revision")
                and nonebot_storage.is_settled(receipt)
                for receipt in receipts
            ):
                self.settle_restart_markers(["webui.nonebot-store"])
        if self.verify_source:
            source_store.finalize_pending_transaction()
            self.settle_source_restart_markers()
        launcher_log("managed plugin transaction passed startup verification")
