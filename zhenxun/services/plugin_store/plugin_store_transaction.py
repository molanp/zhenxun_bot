from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import threading
from typing import Any
import uuid

from filelock import FileLock, Timeout
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from zhenxun.services.plugin_store.operation_journal import settle_operations
from zhenxun.services.plugin_store.plugin_store_receipts import source_digest
from zhenxun.utils.atomic_json import (
    AtomicJsonLockTimeout,
    AtomicJsonRecordError,
    mutate_json_locked,
    read_json_locked,
    write_json_locked,
)
from zhenxun.utils.bytecode import precompile_path
from zhenxun.utils.pythonpath import INSTANCE_ROOT

ROOT = INSTANCE_ROOT / "data/runtime/plugin-store-transaction"
PENDING_FILE = ROOT / "pending-v1.json"
LOADED_PLUGINS_INDEX = INSTANCE_ROOT / "data/runtime/loaded-plugins-v1.json"
_TRANSACTION_LOCK = ROOT / ".transaction.lock"
_LOCKS: dict[Path, FileLock] = {}
_LOCK_GUARD = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked_transaction(timeout: float = 30.0) -> Iterator[None]:
    from zhenxun.services.runtime_mutation import mutation_guard

    ROOT.mkdir(parents=True, exist_ok=True)
    path = _TRANSACTION_LOCK.resolve()
    with _LOCK_GUARD:
        lock = _LOCKS.setdefault(path, FileLock(path, thread_local=True))
    try:
        if lock.is_locked:
            lock.acquire(timeout=timeout)
        else:
            with mutation_guard():
                lock.acquire(timeout=min(timeout, 2))
    except Timeout as error:
        raise AtomicJsonLockTimeout(str(path)) from error
    try:
        yield
    finally:
        lock.release()


def _digest(path: Path, *, python_only: bool = False) -> str:
    """计算插件源码摘要，路径不存在时为 missing

    参数:
        path: 插件目录或文件
        python_only: 只计入 .py/.pyi 文件；待重启卸载期间插件仍会在目录内写数据，
            只核对源码

    返回:
        str: 摘要
    """
    if not path.exists():
        return "missing"
    digest = source_digest(path, python_only=python_only)
    if digest is None:
        raise RuntimeError("plugin_transaction_source_unreadable")
    return digest


def _copy(source: Path, target: Path) -> None:
    if target.exists():
        _remove(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _content_digest(path: Path) -> str:
    """核对文件内容，不把暂存文件名当作插件内容的一部分"""
    if path.is_file():
        return sha256(path.read_bytes()).hexdigest()
    return _digest(path)


def _win_on_rm_error(func: Callable[[str], Any], path: str, _exc_info: object) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, onerror=_win_on_rm_error)
    else:
        path.unlink(missing_ok=True)


def _read_pending() -> dict[str, Any] | None:
    value = read_json_locked(PENDING_FILE, None, strict=True)
    if value is not None and (not isinstance(value, dict) or not value.get("state")):
        raise AtomicJsonRecordError("source_transaction_invalid")
    return value


@_locked_transaction()
def _write_pending(
    value: dict[str, Any] | None, *, expected: dict[str, Any] | None = None
) -> None:
    """比较记录修订后写入或清除源码事务，拒绝迟到的执行结果"""

    def update(current):
        if not isinstance(current, dict):
            raise AtomicJsonRecordError("source_transaction_invalid")
        candidate = value if value is not None else expected
        if candidate is None:
            raise RuntimeError("source_clear_revision_required")
        same = current.get("revision") == candidate.get("revision")
        supersedes = current.get("revision") == candidate.get("supersedes_revision")
        if current and (
            not (same or (value is not None and supersedes))
            or current.get("record_revision", 0) != candidate.get("record_revision", 0)
        ):
            raise RuntimeError("source_transaction_changed")
        if not current and value and candidate.get("record_revision", 0):
            raise RuntimeError("source_transaction_changed")
        revision = int(current.get("record_revision", 0)) + 1
        current.clear()
        if value is not None:
            value["record_revision"] = revision
            current.update(deepcopy(value))

    mutate_json_locked(PENDING_FILE, {}, update, strict=True)


def pending_transaction() -> dict[str, Any] | None:
    return _read_pending()


def _public_operation(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: operation.get(key)
        for key in (
            "operation_id",
            "action",
            "store_key",
            "module",
            "runtime_module",
            "reason",
            "created_at",
        )
    }


def public_transaction() -> dict[str, Any] | None:
    transaction = pending_transaction()
    if not transaction:
        return None
    return {
        "revision": transaction.get("revision"),
        "state": transaction.get("state"),
        "rollback_confirmed": bool(transaction.get("rollback_confirmed")),
        "verification_confirmed": bool(transaction.get("verification_confirmed")),
        "cleanup_error": transaction.get("cleanup_error"),
        "operations": [
            _public_operation(item) for item in transaction.get("operations", [])
        ],
        "failure_reasons": deepcopy(transaction.get("failure_reasons", [])),
        "created_at": transaction.get("created_at"),
        "updated_at": transaction.get("updated_at"),
    }


def _workspace_relative(path: Path) -> str:
    try:
        return _live_path(path).relative_to(INSTANCE_ROOT).as_posix()
    except ValueError as error:
        raise RuntimeError("plugin_transaction_path_outside_workspace") from error


def _live_path(value: str | Path) -> Path:
    """以实例根目录解析插件路径，拒绝符号链接越界"""
    path = Path(value)
    resolved = (path if path.is_absolute() else INSTANCE_ROOT / path).resolve()
    if not resolved.is_relative_to(INSTANCE_ROOT) or resolved == INSTANCE_ROOT:
        raise RuntimeError("plugin_transaction_path_outside_workspace")
    return resolved


def stage_operation(
    *,
    action: str,
    store_key: str,
    module: str,
    runtime_module: str,
    live_path: Path,
    candidate_path: Path | None,
    base_source_path: Path | None = None,
    base_digest: str | None = None,
    receipt: dict[str, Any] | None,
    reason: str,
    dependency_inputs: list[str] | None = None,
    dependency_packages: dict[str, str] | None = None,
    source_build_confirmed: bool = False,
    operation_id: str | None = None,
) -> dict[str, Any]:
    if action not in {"install", "update", "uninstall"}:
        raise RuntimeError("plugin_transaction_action_invalid")
    if operation_id and (
        Path(operation_id).name != operation_id or operation_id in {".", ".."}
    ):
        raise RuntimeError("plugin_operation_id_invalid")
    live_path = _live_path(live_path)
    inputs = {
        "action": action,
        "store_key": store_key,
        "module": module,
        "runtime_module": runtime_module,
        "live_path": _workspace_relative(live_path),
        "candidate": _digest(candidate_path) if candidate_path else None,
        "dependencies": sorted(set(dependency_inputs or [])),
        "packages": dependency_packages or {},
        "source_build_confirmed": bool(source_build_confirmed),
        "receipt": receipt,
    }
    input_digest = sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    with _locked_transaction():
        transaction = _read_pending()
        if transaction and transaction.get("rollback_confirmed"):
            _archive_result(transaction)
            _write_pending(None, expected=transaction)
            transaction = None
        if transaction and transaction.get("state") != "pending_restart":
            raise RuntimeError("plugin_transaction_not_mutable")
        if transaction is None:
            transaction = {
                "version": 1,
                "transaction_id": uuid.uuid4().hex,
                "created_at": _now(),
                "operations": [],
                "state": "pending_restart",
            }
        transaction.setdefault(
            "transaction_id", transaction.get("revision") or uuid.uuid4().hex
        )
        operations = list(transaction.get("operations", []))
        if operation_id:
            existing_id = next(
                (
                    item
                    for item in operations
                    if item.get("operation_id") == operation_id
                ),
                None,
            )
            if existing_id:
                if existing_id.get("input_digest") != input_digest:
                    raise RuntimeError("plugin_operation_id_conflict")
                return {"operation_id": operation_id, "apply_mode": "restart_pending"}
        previous_index = next(
            (
                index
                for index, item in enumerate(operations)
                if item.get("store_key") == store_key
            ),
            None,
        )
        previous = operations[previous_index] if previous_index is not None else None
        if previous and previous.get("action") == "install" and action == "uninstall":
            _remove(ROOT / str(previous["operation_id"]))
            operations.pop(previous_index)
            if not operations:
                _write_pending(None, expected=transaction)
            else:
                transaction["operations"] = operations
                transaction["supersedes_revision"] = transaction.get("revision")
                transaction["revision"] = uuid.uuid4().hex
                transaction["updated_at"] = _now()
                _write_pending(transaction)
            return {
                "operation_id": previous["operation_id"],
                "apply_mode": "rolled_back",
            }
        else:
            operation_id = operation_id or uuid.uuid4().hex
            bundle = ROOT / operation_id
            old_path = bundle / "old"
            new_path = bundle / "new"
            old_source = base_source_path if base_source_path is not None else live_path
            try:
                if old_source.exists():
                    _copy(old_source, old_path)
                if candidate_path is not None and candidate_path.exists():
                    _copy(candidate_path, new_path)
                operation = {
                    "operation_id": operation_id,
                    "input_digest": input_digest,
                    "action": action,
                    "store_key": store_key,
                    "module": module,
                    "runtime_module": runtime_module,
                    "live_path": _workspace_relative(live_path),
                    "base_digest": base_digest or _digest(old_source),
                    "base_source_digest": _digest(old_source, python_only=True),
                    "receipt": receipt,
                    "reason": reason,
                    "dependency_inputs": sorted(set(dependency_inputs or [])),
                    "dependency_packages": dict(
                        sorted((dependency_packages or {}).items())
                    ),
                    "source_build_confirmed": bool(source_build_confirmed),
                    "created_at": _now(),
                }
            except Exception:
                _remove(bundle)
                raise
            if previous_index is None:
                operations.append(operation)
            else:
                _remove(ROOT / str(previous["operation_id"]))
                operations[previous_index] = operation
        transaction["operations"] = operations
        transaction["supersedes_revision"] = transaction.get("revision")
        transaction["revision"] = uuid.uuid4().hex
        transaction["updated_at"] = _now()
        transaction["state"] = "pending_restart"
        _write_pending(transaction)
        return {
            "operation_id": operation["operation_id"],
            "apply_mode": "restart_pending",
        }


def cancel_operation(operation_id: str | None = None) -> dict[str, Any]:
    with _locked_transaction():
        transaction = _read_pending()
        if not transaction or transaction.get("state") not in {
            "pending_restart",
            "failed",
        }:
            raise RuntimeError("plugin_transaction_not_cancelable")
        operations = list(transaction.get("operations", []))
        if operation_id is None:
            removed = operations
            operations = []
        else:
            removed = [
                item for item in operations if item.get("operation_id") == operation_id
            ]
            operations = [
                item for item in operations if item.get("operation_id") != operation_id
            ]
            if not removed:
                raise RuntimeError("plugin_transaction_operation_not_found")
        for item in removed:
            _remove(ROOT / str(item["operation_id"]))
        if not operations:
            _write_pending(None, expected=transaction)
        else:
            transaction["operations"] = operations
            transaction["supersedes_revision"] = transaction.get("revision")
            transaction["revision"] = uuid.uuid4().hex
            transaction["updated_at"] = _now()
            transaction["state"] = "pending_restart"
            transaction.pop("failure_reasons", None)
            _write_pending(transaction)
        settle_operations(
            [str(item["operation_id"]) for item in removed], "canceled", "rolled_back"
        )
        return {
            "apply_mode": "rolled_back",
            "removed_operation_ids": [item["operation_id"] for item in removed],
        }


def dependency_packages() -> dict[str, str]:
    transaction = pending_transaction() or {}
    packages: dict[str, str] = {}
    for operation in transaction.get("operations", []):
        for name, version in operation.get("dependency_packages", {}).items():
            existing = packages.get(str(name))
            if existing is not None and existing != str(version):
                raise RuntimeError("cross_store_dependency_conflict")
            packages[str(name)] = str(version)
    return packages


def _conflict_diagnostic(details: Any) -> str:
    """提取可展示的包名、版本及使用者，不展示来源地址或原始异常。"""
    if not isinstance(details, list):
        return ""
    lines = []
    for issue in details[:8]:
        if not isinstance(issue, dict):
            continue

        def label(key):
            value = issue.get(key)
            return (
                value
                if isinstance(value, str) and re.fullmatch(r"[\w.+:-]{1,160}", value)
                else ""
            )

        if not (name := label("name")):
            continue
        try:
            required = str(Requirement(str(issue.get("requirement"))).specifier)
        except InvalidRequirement:
            required = ""
        version = label("actual") or (
            "未安装" if "actual" in issue and issue["actual"] is None else "未知"
        )
        expected = required or label("expected")
        lines.append(
            f"{label('owner') or '依赖检查'}：{name}{expected}，实际 {version}"
            + (
                "（历史兼容记录需要重新核验）"
                if issue.get("reason") == "compatibility_trial_recheck_required"
                else ""
            )
        )
    return "；".join(lines)


def fail_pending_transaction(code: str, details: Any = None) -> None:
    """将待重启的源码事务标记为失败

    参数:
        code: 失败原因
    """
    with _locked_transaction():
        transaction = _read_pending()
        if not transaction or transaction.get("state") != "pending_restart":
            return
        transaction["state"] = "failed"
        transaction["rollback_confirmed"] = True
        transaction["failure_reasons"] = [{"code": code}]
        diagnostic = (
            _conflict_diagnostic(details)
            if code == "current_environment_dependency_conflict"
            else ""
        )
        if diagnostic:
            transaction["failure_reasons"][0]["diagnostic"] = diagnostic
        transaction["updated_at"] = _now()
        _write_pending(transaction)
        settle_operations(
            [str(item["operation_id"]) for item in transaction.get("operations", [])],
            "failed",
            "failed",
            code,
            diagnostic=diagnostic,
        )


def prepare_dependency_transaction() -> bool:
    """Merge source-store dependency changes into the managed generation."""
    retry_settled_cleanup()
    source = pending_transaction()
    if source and source.get("rollback_confirmed"):
        with _locked_transaction():
            _archive_result(source)
            _write_pending(None, expected=source)
        source = None
    if source and source.get("state") in {"committing", "committed", "cleanup_pending"}:
        finalize_pending_transaction()
        source = pending_transaction()
    if not source or source.get("state") != "pending_restart":
        return True
    packages = dependency_packages()
    layer_packages: dict[str, str] = {}
    if packages:
        from zhenxun.services.nonebot_store.dependencies import protected_core

        core = protected_core()
        for raw_name, version in packages.items():
            name = canonicalize_name(raw_name)
            if name in core:
                if core[name] != version:
                    fail_pending_transaction("core_dependency_conflict")
                    return False
                continue
            layer_packages[name] = version
    from zhenxun.services.nonebot_store.storage import merge_source_dependencies

    result = merge_source_dependencies(layer_packages, source=source)
    if isinstance(result, str):
        fail_pending_transaction(result)
        return False
    return True


def apply_pending_transaction() -> bool:
    with _locked_transaction():
        transaction = _read_pending()
        if transaction and transaction.get("state") == "applying":
            transaction["state"] = "recovery_required"
            transaction["failure_reasons"] = [{"code": "plugin_apply_interrupted"}]
            transaction["updated_at"] = _now()
            _write_pending(transaction)
            return False
        if not transaction or transaction.get("state") != "pending_restart":
            return False
        switched: list[dict[str, Any]] = []
        try:
            for operation in transaction.get("operations", []):
                live = _live_path(operation["live_path"])
                if operation.get("action") != "uninstall":
                    if _digest(live) != operation.get("base_digest"):
                        raise RuntimeError("plugin_transaction_stale")
                    continue
                # A staged uninstall cannot demand a whole-tree match: the plugin
                # is still running and writes its own state inside ``live``. It is
                # checked against its source digest instead, which ignores that
                # churn but still refuses to delete code that was replaced after
                # the operation was reviewed. A live path that is already gone has
                # nothing left to delete, and failing it here would roll the
                # plugin back into place -- the opposite of what was asked.
                expected_source = operation.get("base_source_digest")
                if expected_source is None:
                    continue
                current_source = _digest(live, python_only=True)
                if current_source not in {expected_source, "missing"}:
                    raise RuntimeError("plugin_transaction_stale")
            transaction["state"] = "applying"
            _write_pending(transaction)
            for operation in transaction.get("operations", []):
                live = _live_path(operation["live_path"])
                new_path = ROOT / str(operation["operation_id"]) / "new"
                old_path = ROOT / str(operation["operation_id"]) / "old"
                if live.exists():
                    _copy(live, old_path)
                else:
                    _remove(old_path)
                operation["rollback_digest"] = _content_digest(live)
                _write_pending(transaction)
                switched.append(operation)
                _remove(live)
                if operation.get("action") != "uninstall":
                    _copy(new_path, live)
                operation["applied_digest"] = (
                    "missing"
                    if operation.get("action") == "uninstall"
                    else _content_digest(new_path)
                )
                if _content_digest(live) != operation["applied_digest"]:
                    raise RuntimeError("plugin_apply_content_changed")
                _write_pending(transaction)
        except BaseException as error:
            try:
                _restore_operations({**transaction, "operations": switched})
            except Exception as rollback_error:
                transaction.update(
                    state="recovery_required",
                    rollback_error=str(rollback_error),
                    failure_reasons=[{"code": str(error)}],
                )
                _write_pending(transaction)
                raise
            transaction["state"] = "failed"
            transaction["rollback_confirmed"] = True
            transaction["failure_reasons"] = [{"code": str(error)}]
            transaction["updated_at"] = _now()
            _write_pending(transaction)
            if not isinstance(error, Exception):
                raise
            return False
        transaction["state"] = "verification_pending"
        for operation in switched:
            if operation.get("action") != "uninstall":
                precompile_path(_live_path(operation["live_path"]))
        transaction["updated_at"] = _now()
        _write_pending(transaction)
        return True


def startup_verification() -> tuple[bool, dict[str, Any]]:
    transaction = pending_transaction()
    if not transaction or transaction.get("state") != "verification_pending":
        return True, {"failed": []}
    index = read_json_locked(LOADED_PLUGINS_INDEX, {})
    modules = {str(module) for module in index.get("modules", [])}
    failures = []
    for operation in transaction.get("operations", []):
        runtime_module = str(operation.get("runtime_module") or "")
        loaded = runtime_module in modules
        expected = operation.get("action") != "uninstall"
        if loaded != expected:
            failures.append(
                {
                    "store_key": operation.get("store_key"),
                    "code": "plugin_startup_verification_failed",
                }
            )
    return not failures, {"failed": failures}


def finalize_pending_transaction() -> None:
    """持久化提交决定和收据，操作日志与清理可在重启后继续完成"""
    with _locked_transaction():
        transaction = _read_pending()
        if not transaction or transaction.get("state") not in {
            "verification_pending",
            "committing",
            "committed",
            "cleanup_pending",
        }:
            return
        from zhenxun.services.plugin_store.plugin_store_receipts import (
            StoreReceiptStore,
        )

        transaction.update(state="committing", verification_confirmed=True)
        _write_pending(transaction)
        for operation in transaction.get("operations", []):
            if operation.get("action") == "uninstall":
                StoreReceiptStore.delete(str(operation["store_key"]))
            elif isinstance(operation.get("receipt"), dict):
                StoreReceiptStore.set(str(operation["store_key"]), operation["receipt"])
        transaction.update(state="committed", committed_at=_now())
        _write_pending(transaction)
        _archive_result(transaction)
        _write_pending(None, expected=transaction)


def _settled_path(transaction: dict[str, Any]) -> Path:
    revision = str(transaction.get("revision") or "")
    if not revision:
        raise AtomicJsonRecordError("source_transaction_revision_missing")
    return ROOT / "settled" / (sha256(revision.encode()).hexdigest() + ".json")


def _archive_result(transaction: dict[str, Any]) -> None:
    """先保存可查询终态，再独立结算日志和清理本事务文件"""
    path = _settled_path(transaction)
    write_json_locked(path, transaction)
    try:
        failed = bool(transaction.get("rollback_confirmed"))
        settle_operations(
            [str(item["operation_id"]) for item in transaction.get("operations", [])],
            "failed" if failed else "completed",
            "rolled_back" if failed else "applied",
        )
        transaction.pop("result_error", None)
    except Exception as error:
        transaction["result_error"] = {"code": type(error).__name__}
    try:
        active = _read_pending() or {}
        referenced = {
            item["operation_id"]
            for item in active.get("operations", [])
            if active.get("revision") != transaction.get("revision")
        }
        for operation in transaction.get("operations", []):
            identity = str(operation["operation_id"])
            if Path(identity).name != identity or identity in {".", ".."}:
                raise RuntimeError("source_cleanup_path_invalid")
            if identity not in referenced:
                _remove(ROOT / identity)
        transaction.pop("cleanup_error", None)
    except Exception as error:
        transaction["cleanup_error"] = {"code": type(error).__name__}
    write_json_locked(path, transaction)


def reconcile_operation(value: dict[str, Any]) -> bool:
    """以持久化提交或回滚收据恢复源码商店操作终态"""
    records = [_read_pending()]
    records.extend(
        read_json_locked(path, {}, strict=True)
        for path in (ROOT / "settled").glob("*.json")
    )
    for record in records:
        if not record or not any(
            item.get("operation_id") == value.get("operation_id")
            for item in record.get("operations", [])
        ):
            continue
        if record.get("rollback_confirmed"):
            status, mode = "failed", "rolled_back"
        elif record.get("state") in {"committed", "cleanup_pending"}:
            status, mode = "completed", "applied"
        else:
            return False
        value.update(status=status, apply_mode=mode)
        value.setdefault("result", {}).update(
            status=status,
            apply_mode=mode,
            restart_required=False,
            restart_available=False,
            cleanup_error=record.get("cleanup_error"),
        )
        if record.get("rollback_confirmed"):
            reasons = record.get("failure_reasons") or []
            if reasons and isinstance(reasons[0], dict):
                for key, source in (("reason", "code"), ("diagnostic", "diagnostic")):
                    if reasons[0].get(source):
                        value["result"][key] = reasons[0][source]
        return True
    return False


def _restore_operations(transaction: dict[str, Any]) -> None:
    for operation in transaction.get("operations", []):
        live = _live_path(operation["live_path"])
        old_path = ROOT / str(operation["operation_id"]) / "old"
        before = operation.get("rollback_digest")
        if before is None or _content_digest(old_path) != before:
            raise RuntimeError("plugin_rollback_evidence_missing")
        if _content_digest(live) not in {before, operation.get("applied_digest")}:
            raise RuntimeError("plugin_rollback_target_changed")
    for operation in reversed(transaction.get("operations", [])):
        live = _live_path(operation["live_path"])
        old_path = ROOT / str(operation["operation_id"]) / "old"
        if _content_digest(live) == operation["rollback_digest"]:
            continue
        _remove(live)
        if old_path.exists():
            _copy(old_path, live)
        if _content_digest(live) != operation["rollback_digest"]:
            raise RuntimeError("plugin_rollback_content_changed")


def rollback_pending_transaction(
    failures: list[dict[str, Any]] | None = None,
) -> None:
    with _locked_transaction():
        transaction = _read_pending()
        if not transaction or transaction.get("state") != "verification_pending":
            return
        try:
            _restore_operations(transaction)
        except Exception as error:
            transaction.update(
                state="recovery_required", rollback_error=type(error).__name__
            )
            _write_pending(transaction)
            raise
        transaction["state"] = "failed"
        transaction["rollback_confirmed"] = True
        transaction["failure_reasons"] = failures or [
            {"code": "plugin_startup_verification_failed"}
        ]
        transaction["updated_at"] = _now()
        _write_pending(transaction)
        _archive_result(transaction)
        _write_pending(None, expected=transaction)


__all__ = [
    "apply_pending_transaction",
    "cancel_operation",
    "dependency_packages",
    "fail_pending_transaction",
    "finalize_pending_transaction",
    "pending_transaction",
    "prepare_dependency_transaction",
    "public_transaction",
    "rollback_pending_transaction",
    "stage_operation",
    "startup_verification",
]


async def preflight_source_requirements(
    files: list[Path], *, store_key: str = "", allow_build: bool = False
) -> dict[str, Any]:
    """按依赖层规则预检商店插件的 requirements 文件

    与其他依赖来源（包括已安装和待应用的商店插件）一起解析，确认可解析、不改动
    核心依赖与已安装的 ORM 包。

    参数:
        files: 插件源码中的 requirements 文件
        store_key: 商店插件标识，该插件原有的依赖声明被本次声明替换
        allow_build: 是否允许解析时构建源码包读取依赖信息

    返回:
        dict[str, Any]: 依赖层中新增或变化的包版本、包变化与是否需要源码构建

    异常:
        DependencyAnalysisError: 依赖声明无效、环境漂移、与核心依赖冲突、需要
            安装工具或会改变已安装的 ORM 包
    """
    from zhenxun.services.nonebot_store import dependencies
    from zhenxun.services.nonebot_store.dependencies import DependencyAnalysisError
    from zhenxun.services.nonebot_store.environment import capture_environment
    from zhenxun.services.nonebot_store.policy import unresolved_conflict
    from zhenxun.services.nonebot_store.storage import (
        pending_transaction as layer_transaction,
    )
    from zhenxun.services.plugin_store.plugin_store_receipts import (
        receipt_dependency_inputs,
    )

    requirements = dependencies.read_requirement_files(files)
    if not requirements:
        return {
            "resolved_packages": {},
            "package_changes": {"added": [], "changed": [], "removed": []},
            "candidate_inputs": [],
            "source_build_required": False,
        }
    if dependencies.environment_drift():
        raise DependencyAnalysisError("environment_drift")
    tools = sorted(
        {dependencies.parse_requirement(raw).name for raw in requirements}
        & dependencies.FORBIDDEN_LAYER_PACKAGES
    )
    if tools:
        raise DependencyAnalysisError(
            "dependency_requirement_unsupported", "、".join(tools)
        )
    overrides: dict[str, list[str] | None] = {}
    for operation in (pending_transaction() or {}).get("operations", []):
        key = str(operation.get("store_key") or "")
        overrides[key] = (
            None
            if operation.get("action") == "uninstall"
            else [str(raw) for raw in operation.get("dependency_inputs") or []]
        )
    if store_key:
        overrides[store_key] = list(requirements)
    pending = layer_transaction() or {}
    manifest = (
        pending.get("target_manifest")
        if pending.get("state") == "pending_restart"
        else None
    ) or dependencies.load_manifest()
    snapshot = capture_environment()
    result = await dependencies.resolve_layer(
        requirements,
        snapshot=snapshot,
        manifest=manifest,
        sources=receipt_dependency_inputs(overrides),
        allow_build=allow_build,
    )
    changes = [
        item
        for item in result["changes"]
        if item["to"] and item["name"] in result["resolved"]
    ]
    if changes:
        affected = {item["name"] for item in result["changes"]}
        planned = replace(snapshot, manifest=manifest)
        permitted = manifest.get("dependency_warnings", [])
        if pending.get("state") == "pending_restart":
            permitted = pending.get("remaining_conflicts", permitted)
        unresolved = [
            error
            for issue in snapshot.conflicts
            if not affected.intersection((issue["name"], issue["owner"]))
            and (error := unresolved_conflict(planned, issue, permitted))
        ]
        if unresolved:
            raise DependencyAnalysisError(
                "current_environment_dependency_conflict",
                _conflict_diagnostic(unresolved),
                details=unresolved,
            )
    return {
        "resolved_packages": {item["name"]: item["to"] for item in changes},
        "package_changes": {
            "added": [
                {"name": item["name"], "version": item["to"]}
                for item in changes
                if not item["from"]
            ],
            "changed": [item for item in changes if item["from"]],
            "removed": [],
        },
        "candidate_inputs": list(requirements),
        "source_build_required": result["source_build_required"],
    }


@_locked_transaction()
def retry_settled_cleanup() -> None:
    """重试已结算事务的日志与清理，不重新应用或回滚插件"""
    for path in (ROOT / "settled").glob("*.json"):
        record = read_json_locked(path, None, strict=True)
        if record and (record.get("cleanup_error") or record.get("result_error")):
            _archive_result(record)
