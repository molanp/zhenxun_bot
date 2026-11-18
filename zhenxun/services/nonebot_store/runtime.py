from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Event
import time
from typing import Any
from uuid import uuid4

from packaging.utils import canonicalize_name

from zhenxun.utils.pythonpath import (
    INSTANCE_ROOT,
    activate_dependency_layer,
    subprocess_pythonpath,
)

from .dependencies import protected_core, safe_process_error
from .storage import (
    LAYER_ROOT,
    PENDING_FILE,
    ROLLBACK_FILE,
    STARTUP_STATUS_FILE,
    generation_path,
    load_manifest,
    next_generation,
    pending_transaction,
    prune_generations,
    read_json,
    remove_generation,
    remove_orm_backups,
    save_manifest,
    transaction_lock,
    utc_now,
    write_json,
)


class LayerBuildError(RuntimeError):
    def __init__(
        self,
        code: str,
        detail: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(detail or code)
        self.code = code
        self.details = details


def recovery_required() -> bool:
    """Keep the management surface available while rollback is unconfirmed."""
    from zhenxun.services.migration import state as restore_state
    from zhenxun.services.plugin_store.plugin_store_transaction import (
        pending_transaction as source_pending,
    )
    from zhenxun.services.runtime_mutation import mutation_owner

    try:
        if (pending_transaction() or {}).get("state") in {
            "recovery_required",
            "building",
            "rolling_back",
        }:
            return True
        if (source_pending() or {}).get("state") in {"recovery_required", "applying"}:
            return True
        owner = mutation_owner()
        if owner and owner.get("kind") == "runtime":
            return True
        restore = restore_state.load() or {}
        return bool(
            restore
            and (
                restore.get("state")
                in {"recovery_required", "applying", "awaiting_orm"}
                or (
                    restore.get("record_version") != 2
                    and restore.get("state") not in restore_state.TERMINAL
                )
            )
        )
    except (OSError, RuntimeError):
        return True


def environment_signature() -> dict[str, str]:
    """当前解释器与核心虚拟环境的特征，依赖层按它判断是否需要重新同步

    返回:
        dict[str, str]: python 为解释器的首选 wheel 标签，base 为核心虚拟环境
            site-packages 中全部包版本的摘要
    """
    from importlib import metadata
    import sysconfig

    from packaging.tags import sys_tags

    paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    packages = sorted(
        f"{canonicalize_name(dist.name)}=={dist.version}"
        for dist in metadata.distributions(path=paths)
        if dist.name
    )
    return {
        "python": str(next(iter(sys_tags()))),
        "base": sha256("\n".join(packages).encode()).hexdigest(),
    }


def environment_resync_required() -> bool:
    """依赖层记录的解释器或核心虚拟环境与当前不一致，且没有待处理的事务

    同一环境下自动同步已失败过时不再重复尝试。

    返回:
        bool: 是否需要执行 stage_environment_resync
    """
    from .storage import settled_transactions

    manifest = load_manifest()
    if not isinstance(manifest.get("active_generation"), int) or pending_transaction():
        return False
    current = environment_signature()
    if manifest.get("environment") == current:
        return False
    latest = next(iter(settled_transactions()), {})
    return not (
        latest.get("resync_environment") == current and latest.get("state") == "failed"
    )


@transaction_lock()
def stage_environment_resync() -> bool:
    """按当前解释器与核心虚拟环境重新同步依赖层

    核心虚拟环境已提供不低于依赖层版本、且满足全部依赖声明的包从依赖层移除；
    解释器变化时按原版本重新安装依赖层。需要重建时写入待应用的依赖层事务，
    由启动器构建并在 worker 启动后验证；否则只更新清单中记录的环境。

    返回:
        bool: 是否写入了待应用的依赖层事务
    """
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    from .environment import capture_environment
    from .policy import ORM_PACKAGES, retained_conflicts
    from .storage import defer_transaction, draft_transaction

    manifest = load_manifest()
    if not isinstance(manifest.get("active_generation"), int) or pending_transaction():
        return False
    current = environment_signature()
    recorded = manifest.get("environment") or {}
    if recorded == current:
        return False
    snapshot = capture_environment(manifest=manifest)
    target = deepcopy(manifest)
    dropped = []
    for name, dist in snapshot.layer.items():
        base = snapshot.base.get(name)
        if base is None or name in ORM_PACKAGES or name not in target["packages"]:
            continue
        requirements = [
            Requirement(item["requirement"])
            for item in snapshot.declarations
            if item["name"] == name
        ]
        try:
            newer = Version(base.version) >= Version(dist.version)
        except InvalidVersion:
            continue
        if newer and all(
            not requirement.url
            and requirement.specifier.contains(base.version, prereleases=True)
            for requirement in requirements
        ):
            target["packages"].pop(name)
            dropped.append(name)
    if not dropped and recorded.get("python", current["python"]) == current["python"]:
        manifest["environment"] = current
        save_manifest(manifest)
        return False
    target["dependency_warnings"] = retained_conflicts(snapshot, dropped)
    transaction = draft_transaction(
        None,
        target,
        action="environment_repair",
        resync_environment=current,
        remaining_conflicts=deepcopy(target["dependency_warnings"]),
        source_build_confirmed=True,
        operations=[
            {
                "kind": "environment_repair",
                "name": "依赖层同步：解释器或核心依赖已变化",
                "released": sorted(dropped),
                "target_manifest": deepcopy(target),
                "created_at": utc_now(),
            }
        ],
    )
    defer_transaction(transaction)
    return True


def rollback_for_launcher(failed: list[dict[str, Any]] | None = None) -> bool:
    """回滚未通过启动验证的依赖层事务，回滚失败时保留为待恢复状态

    参数:
        failed: launcher 记录的失败项

    返回:
        bool: 是否已回滚
    """
    try:
        rollback_pending_transaction(failed)
    except Exception:
        return False
    return True


def _active_path(manifest: dict[str, Any] | None = None) -> Path | None:
    manifest = manifest or load_manifest()
    generation = manifest.get("active_generation")
    if not isinstance(generation, int):
        return None
    path = generation_path(generation).resolve()
    return path if path.is_dir() else None


def activate_current_generation() -> Path | None:
    """Put exactly one managed dependency generation at the front of sys.path."""
    return activate_dependency_layer(_active_path())


def referenced_generations() -> set[int]:
    """Return generations still referenced by modules loaded in this process."""
    layer_root = LAYER_ROOT.resolve()
    generations: set[int] = set()
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        candidates: list[Any] = [getattr(module, "__file__", None)]
        module_path = getattr(module, "__path__", None)
        if module_path is not None:
            try:
                candidates.extend(tuple(module_path))
            except (KeyError, TypeError):
                pass
        for candidate in candidates:
            if not isinstance(candidate, str | os.PathLike):
                continue
            try:
                relative = Path(candidate).resolve().relative_to(layer_root)
            except (OSError, ValueError):
                continue
            if not relative.parts or not relative.parts[0].startswith("generation-"):
                continue
            try:
                generations.add(int(relative.parts[0].removeprefix("generation-")))
            except ValueError:
                continue
    return generations


def module_source_path(module_name: str, root: Path | None = None) -> Path | None:
    root = root or _active_path()
    if root is None:
        return None
    parts = module_name.split(".")
    package = root.joinpath(*parts)
    if (package / "__init__.py").is_file():
        return package
    module = package.with_suffix(".py")
    return module if module.is_file() else None


def generation_native_extensions(root: Path | None = None) -> set[str]:
    root = root or _active_path()
    if root is None:
        return set()
    metadata = read_json(root / ".zhenxun-generation.json", {})
    values = metadata.get("native_extensions", [])
    return {str(item) for item in values} if isinstance(values, list) else set()


def _scan_native_extensions(root: Path) -> list[str]:
    extensions = {".pyd", ".so", ".dylib", ".dll"}
    return [
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    ]


def layer_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix == ".pyc":
            try:
                source = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                source = path.with_suffix(".py")
            if source.is_file():
                continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _reusable_distribution(distribution) -> list[tuple[Path, Path]] | None:
    """列出当前依赖层中可以直接沿用到新依赖层的包文件

    包的 wheel 标签不适用于当前解释器、安装记录中有依赖层以外或已缺失的文件时
    不能沿用。

    参数:
        distribution: 当前依赖层中的包

    返回:
        list[tuple[Path, Path]] | None: (源文件, 相对依赖层的路径) 列表，不能沿用时
            为 None
    """
    from packaging.tags import parse_tag, sys_tags

    from .environment import _recorded_files

    tags = {
        tag
        for line in (distribution.read_text("WHEEL") or "").splitlines()
        if line.startswith("Tag:")
        for tag in parse_tag(line.partition(":")[2].strip())
    }
    if not tags or tags.isdisjoint(sys_tags()):
        return None
    root = Path(distribution.locate_file("")).resolve()
    files = []
    for file in _recorded_files(distribution):
        if "__pycache__" in file.parts:
            continue
        source = Path(distribution.locate_file(file)).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            return None
        files.append((source, source.relative_to(root)))
    return files


def _link_files(files: list[tuple[Path, Path]], staging: Path) -> None:
    """把文件硬链接到候选依赖层，无法硬链接时复制

    参数:
        files: (源文件, 相对依赖层的路径) 列表
        staging: 候选依赖层目录
    """
    for source, relative in files:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def build_generation(
    transaction: dict[str, Any], *, cancel_requested: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """按事务的目标清单构建新的依赖层

    版本未变化且适用于当前解释器的包从当前依赖层硬链接，其余包重新安装。

    参数:
        transaction: 依赖层事务
        cancel_requested: 返回是否已请求取消

    返回:
        dict[str, Any]: 依赖层编号、目录、原生扩展与摘要
    """
    return _build_generation(transaction, cancel_requested=cancel_requested)


async def prepare_generation(
    transaction: dict[str, Any],
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build off the event loop and finish cleanup before releasing a canceled owner."""
    from zhenxun.services.runtime_mutation import runtime_mutation_coordinator

    cancel = Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            build_generation,
            transaction,
            cancel_requested=lambda: cancel.is_set()
            or runtime_mutation_coordinator.cancellation_requested
            or bool(cancel_requested and cancel_requested()),
        )
    )
    built = None
    try:
        built = await asyncio.shield(task)
        runtime_mutation_coordinator.checkpoint()
        return built
    except asyncio.CancelledError:
        cancel.set()
        if built is None:
            with contextlib.suppress(Exception):
                built = await asyncio.shield(task)
        if built is not None:
            remove_generation(built["generation"])
        raise


def _run_installer(
    command: list[str],
    environment: dict[str, str],
    cancel_requested: Callable[[], bool] | None,
) -> subprocess.CompletedProcess:
    """Wait for an installer and its owned descendants before releasing staging."""
    import psutil

    from zhenxun.utils.process_tree import kill_tree, verified_descendants

    process = subprocess.Popen(
        command,
        cwd=str(INSTANCE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    owned: dict[int, psutil.Process] = {}
    try:
        parent = psutil.Process(process.pid)
        owned[parent.pid] = parent
    except psutil.NoSuchProcess:
        parent = None
    deadline = time.monotonic() + 900
    finished = False
    try:
        while True:
            if parent is not None:
                with contextlib.suppress(psutil.NoSuchProcess):
                    owned.update(
                        (child.pid, child) for child in verified_descendants(parent)
                    )
            if cancel_requested and cancel_requested():
                raise LayerBuildError("dependency_operation_interrupted")
            if time.monotonic() >= deadline:
                raise LayerBuildError("dependency_layer_build_timeout")
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                finished = True
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                continue
    finally:
        unexpected_children = []
        if finished:
            _, unexpected_children = psutil.wait_procs(list(owned.values()), timeout=5)
        alive = kill_tree(list(owned.values()))
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        if alive:
            raise LayerBuildError("dependency_tool_cleanup_unconfirmed")
        if unexpected_children:
            raise LayerBuildError("dependency_tool_child_incomplete")


def _build_generation(
    transaction: dict[str, Any], *, cancel_requested: Callable[[], bool] | None = None
) -> dict[str, Any]:
    from zhenxun.services.installer_network import (
        UV_BINARY_POLICY_VERSION,
        installer_environment,
        resolver_configuration_error,
        uv_binary_options,
        uv_command,
        uv_version,
    )

    packages = transaction.get("target_manifest", {}).get("packages", {})
    if not isinstance(packages, dict):
        raise LayerBuildError("dependency_plan_invalid")
    from .environment import capture_environment
    from .policy import guard_orm_changes

    before = capture_environment()
    effective = {name: dist.version for name, dist in before.base.items()}
    effective.update(
        {name: str(info.get("version")) for name, info in packages.items()}
    )
    if not transaction.get("orm_packages_allowed"):
        guard_orm_changes(before.versions, effective)
    core = protected_core()
    reused: dict[str, list[tuple[Path, Path]]] = {}
    requested: list[str] = []
    wheel_only: list[str] = []
    for raw_name, info in sorted(packages.items()):
        name = canonicalize_name(str(raw_name))
        if name in core:
            raise LayerBuildError("core_dependency_in_layer")
        if not isinstance(info, dict) or not info.get("version"):
            raise LayerBuildError("dependency_plan_invalid")
        current = before.layer.get(name)
        if current is not None and current.version == str(info["version"]):
            files = _reusable_distribution(current)
            if files is not None:
                reused[name] = files
                continue
        else:
            wheel_only.append(name)
        requested.append(f"{name}=={info['version']}")

    manifest = load_manifest()
    generation = next_generation(manifest)
    LAYER_ROOT.mkdir(parents=True, exist_ok=True)
    final_path = generation_path(generation)
    if final_path.exists():
        raise LayerBuildError("generation_already_exists")

    with tempfile.TemporaryDirectory(
        prefix=f"generation-{generation}-", dir=str(LAYER_ROOT)
    ) as temporary:
        staging = Path(temporary)
        for files in reused.values():
            _link_files(files, staging)
        if requested:
            requirements = staging.parent / f"generation-{generation}.requirements.txt"
            requirements.write_text("\n".join(requested) + "\n", encoding="utf-8")
            command = uv_command(
                "pip",
                "install",
                "--target",
                str(staging),
                "--requirements",
                str(requirements),
                "--no-deps",
                "--strict",
                "--python",
                sys.executable,
            )
            command.extend(
                uv_binary_options(
                    wheels_only=False,
                    restricted_packages=()
                    if transaction.get("source_build_confirmed")
                    else tuple(wheel_only),
                )
            )
            try:
                completed = _run_installer(
                    command, installer_environment(), cancel_requested
                )
            finally:
                requirements.unlink(missing_ok=True)
            if completed.returncode:
                detail = safe_process_error(completed.stderr or completed.stdout)
                if resolver_configuration_error(detail):
                    raise LayerBuildError(
                        "dependency_resolver_configuration",
                        (
                            f"uv={uv_version()} "
                            f"binary_policy={UV_BINARY_POLICY_VERSION}: {detail}"
                        ),
                    )
                raise LayerBuildError(
                    "dependency_layer_build_failed",
                    detail,
                    details={
                        "stage": "dependency_build",
                        "exit_code": completed.returncode,
                        "diagnostic": detail,
                    },
                )
        from .environment import capture_environment, validate_candidate

        snapshot = capture_environment(
            manifest=transaction["target_manifest"], candidate_path=staging
        )
        validate_candidate(staging, transaction, snapshot=snapshot, migrate_trials=True)
        native_files = _scan_native_extensions(staging)
        (staging / ".zhenxun-generation.json").write_text(
            json.dumps({"native_extensions": native_files}, indent=2), encoding="utf-8"
        )
        digest = layer_digest(staging)
        staging.replace(final_path)
    return {
        "generation": generation,
        "path": final_path,
        "native_extensions": native_files,
        "digest": digest,
    }


def _validate_generation(transaction: dict[str, Any], build: dict[str, Any]) -> None:
    from .environment import validate_candidate

    path = generation_path(int(build["generation"]))
    if layer_digest(path) != build.get("digest"):
        raise LayerBuildError("dependency_layer_state_mismatch")
    validate_candidate(path, transaction)


@transaction_lock()
def commit_generation(
    transaction: dict[str, Any], build: dict[str, Any], *, verify_on_start: bool
) -> dict[str, Any]:
    """把已构建的依赖层设为当前清单

    build 须为刚构建或经 _staged_build 校验过的依赖层，这里不再重复校验。
    回滚清单在每个事务中只写入一次并在事务中标记；提交后进程中断再次提交时，
    沿用已保存的回滚清单，不会用已提交的清单覆盖它。

    参数:
        transaction: 待提交的依赖事务
        build: 已构建的依赖层
        verify_on_start: 是否需要启动验证

    返回:
        dict[str, Any]: 新的清单
    """
    from zhenxun.utils.atomic_json import read_json_locked

    current = pending_transaction() or {}
    if current.get("revision") != transaction.get("revision") or current.get(
        "record_revision", 0
    ) != transaction.get("record_revision", 0):
        raise LayerBuildError("dependency_transaction_changed")
    previous = (
        read_json_locked(ROLLBACK_FILE, None, strict=True)
        if transaction.get("rollback_saved")
        else None
    )
    if transaction.get("rollback_saved") and not isinstance(previous, dict):
        raise LayerBuildError("dependency_rollback_receipt_missing")
    if not isinstance(previous, dict):
        previous = load_manifest()
        write_json(ROLLBACK_FILE, previous)
        transaction["rollback_saved"] = True
        write_json(PENDING_FILE, transaction)
    target = deepcopy(transaction["target_manifest"])
    target["active_generation"] = int(build["generation"])
    target["previous_generation"] = previous.get("active_generation")
    target["pending_verification"] = bool(verify_on_start)
    target.pop("verified_transaction_revision", None)
    target["generation_digest"] = build["digest"]
    target["environment"] = environment_signature()
    save_manifest(target)
    return target


@transaction_lock()
def stage_generation(transaction: dict[str, Any], build: dict[str, Any]) -> None:
    """保存刚构建的依赖层，留待重启后提交，不对运行中的 worker 生效

    参数:
        transaction: 依赖层事务
        build: 刚构建的依赖层
    """
    previous = transaction.get("generation")
    transaction["generation"] = int(build["generation"])
    transaction["generation_digest"] = str(build["digest"])
    transaction["native_extensions"] = list(build.get("native_extensions") or [])
    transaction["state"] = (
        "preparing" if transaction.get("preparation_only") else "pending_restart"
    )
    write_json(PENDING_FILE, transaction)
    if isinstance(previous, int) and previous != build["generation"]:
        remove_generation(previous)


def _staged_build(transaction: dict[str, Any]) -> dict[str, Any] | None:
    generation = transaction.get("generation")
    if not isinstance(generation, int):
        return None
    path = generation_path(generation)
    digest = str(transaction.get("generation_digest") or "")
    if not path.is_dir() or not digest:
        return None
    build = {
        "generation": generation,
        "path": path,
        "digest": digest,
        "native_extensions": list(transaction.get("native_extensions") or []),
    }
    try:
        _validate_generation(transaction, build)
    except LayerBuildError:
        return None
    return build


def _run_orm_migration(action: str) -> tuple[int, str]:
    """在子进程中执行 ORM 迁移操作

    参数:
        action: apply / restore / finalize

    返回:
        tuple[int, str]: 退出码，失败时的诊断输出
    """
    transaction = pending_transaction() or {}
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "zhenxun.services.nonebot_store.orm_migration",
                action,
            ],
            cwd=str(INSTANCE_ROOT),
            env={
                **os.environ,
                "PYTHONPATH": subprocess_pythonpath(),
                "ZHENXUN_ORM_TRANSACTION_REVISION": str(
                    transaction.get("revision") or ""
                ),
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        completed = subprocess.CompletedProcess([], 124, "", type(error).__name__)
    diagnostic = (
        safe_process_error(completed.stderr or completed.stdout or "")[-4000:]
        if completed.returncode
        else ""
    )
    return int(completed.returncode), diagnostic


def finalize_orm_migration() -> None:
    code, diagnostic = _run_orm_migration("finalize")
    if code:
        raise LayerBuildError("database_migration_cleanup_failed", diagnostic)


def record_manual_operations(transaction: dict, mode: str, status: str) -> None:
    """记录同一依赖层事务中各商店与手动操作的最终结果

    生效时附带启动验证的告警并标记未能加载的插件，回滚时附带失败说明。

    参数:
        transaction: 依赖层事务
        mode: 生效方式
        status: 操作状态
    """
    operations = [
        item
        for item in transaction.get("operations", [])
        if item.get("operation_id")
        and (
            item.get("kind") in {"manual_dependencies", "environment_repair"}
            or item.get("project_link")
        )
    ]
    if not operations:
        return
    from zhenxun.services.plugin_store.operation_journal import (
        operation_status,
        record_operation,
    )

    applied = mode == "applied"
    load_failures = (load_manifest().get("load_failures") or {}) if applied else {}
    details = transaction.get("warnings" if applied else "error_details") or []
    for operation in operations:
        identity = operation["operation_id"]
        previous = (operation_status(identity) or {}).get("result") or {}
        previous.pop("cleanup_error", None)
        if applied or details:
            previous["diagnostic"] = "；".join(map(str, details))
        previous["load_failed"] = operation.get("module_name") in load_failures
        record_operation(
            "dependencies:manual"
            if operation.get("kind") == "manual_dependencies"
            else "dependencies:repair"
            if operation.get("kind") == "environment_repair"
            else f"nonebot:{operation['project_link']}",
            {
                **previous,
                "operation_id": identity,
                "apply_mode": mode,
                "status": status,
                "restart_required": mode == "restart_pending",
                "restart_available": mode == "restart_pending",
                "request_binding": operation.get("request_binding")
                or previous.get("request_binding"),
                "reason": transaction.get("error_code"),
            },
        )


def reconcile_operation(value: dict[str, Any]) -> bool:
    """按依赖层事务、清单与修复状态更新依赖相关操作的记录

    参数:
        value: 操作记录

    返回:
        bool: 记录状态已由依赖层确定
    """
    if value.get("store_key") == "dependencies:manual":
        receipt = (
            load_manifest()
            .get("declaration_receipts", {})
            .get(value.get("operation_id"))
        )
        if isinstance(receipt, dict):
            value.update(
                status="completed", apply_mode="applied", result=deepcopy(receipt)
            )
            return True
    if value.get("store_key") == "dependencies:repair":
        from .storage import dependency_sync_status

        sync = dependency_sync_status()
        if sync.get("operation_id") == value.get("operation_id"):
            state = sync.get("status")
            mode, status = "verifying", "running"
            if state == "prepared":
                mode, status = "restart_pending", "pending"
            elif state == "canceled":
                mode, status = "rolled_back", "canceled"
            elif state == "failed":
                mode, status = "failed", "failed"
            elif state == "succeeded":
                from zhenxun.services.startup import startup_coordinator

                if startup_coordinator.runtime_ready:
                    mode, status = "applied", "completed"
            value["status"], value["apply_mode"] = status, mode
            value.setdefault("result", {}).update(
                status=status,
                apply_mode=mode,
                restart_required=mode == "restart_pending",
                restart_available=mode == "restart_pending",
            )
            if sync.get("code"):
                value["result"]["reason"] = sync["code"]
            return True
    transaction = pending_transaction() or {}
    if not any(
        item.get("operation_id") == value.get("operation_id")
        for item in transaction.get("operations", [])
    ):
        from .storage import settled_transactions

        transaction = next(
            (
                record
                for record in settled_transactions()
                if any(
                    item.get("operation_id") == value.get("operation_id")
                    for item in record.get("operations", [])
                )
            ),
            {},
        )
    if any(
        item.get("operation_id") == value.get("operation_id")
        for item in transaction.get("operations", [])
    ):
        state = transaction.get("state")
        mode, status = {
            "pending_restart": ("restart_pending", "pending"),
            "migration_blocked": ("restart_pending", "pending"),
            "building": ("preparing", "running"),
            "verification_pending": ("verifying", "running"),
            "recovery_required": ("recovery_required", "recovery_required"),
            "failed": (
                "rolled_back" if transaction.get("rollback_confirmed") else "failed",
                "failed",
            ),
            "committed": ("applied", "completed"),
            "cleanup_pending": ("applied", "completed"),
        }.get(state, (value.get("apply_mode"), value.get("status")))
        value["status"], value["apply_mode"] = status, mode
        value.setdefault("result", {}).update(
            status=status,
            apply_mode=mode,
            restart_required=mode == "restart_pending",
            restart_available=mode == "restart_pending",
        )
        value["result"]["transaction_state"] = state
        if state == "migration_blocked":
            value["result"].update(
                reason="external_database_manual_migration_required",
                restart_available=False,
            )
        if transaction.get("cleanup_error"):
            value["result"]["cleanup_error"] = transaction["cleanup_error"]
        else:
            value["result"].pop("cleanup_error", None)
        if state != "building":
            return True
    return False


def apply_pending_transaction() -> bool:
    """launcher 停止旧 worker 后构建并提交待应用的依赖层

    失败时活动依赖层未切换、数据库未迁移或迁移已自行恢复的事务记为已确认回滚，
    由之后的事务操作自动归档，不再阻挡新的依赖操作。

    返回:
        bool: 是否已提交并等待启动验证
    """
    import psutil

    with transaction_lock():
        transaction = pending_transaction()
        if transaction and transaction.get("parent_operation_id"):
            from zhenxun.services.migration import state as restore_state

            parent = restore_state.load(transaction["parent_operation_id"]) or {}
            if parent.get("state") == "preparing":
                return False
            if transaction.get("preparation_only"):
                raise LayerBuildError("migration_candidate_not_authorized")
        if transaction and transaction.get("state") == "building":
            raise LayerBuildError("dependency_executor_unconfirmed")
        if transaction and transaction.get("state") in {
            "pending_restart",
            "migration_blocked",
        }:
            transaction.update(
                state="building",
                executor={
                    "pid": os.getpid(),
                    "created_at": psutil.Process().create_time(),
                },
            )
            write_json(PENDING_FILE, transaction)
    if transaction is None:
        return False
    if transaction.get("state") in {"committed", "cleanup_pending", "settling"}:
        finalize_pending_transaction()
        return False
    if transaction.get("state") not in {
        "pending_restart",
        "building",
        "migration_blocked",
    }:
        return False
    migrated = False
    try:
        build = _staged_build(transaction)
        if build is None and transaction.get("prepared_for_migration"):
            raise LayerBuildError("migration_prepared_candidate_changed")
        build = build or build_generation(transaction)
        if transaction.get("database_migration_possible"):
            transaction["generation"] = int(build["generation"])
            transaction["generation_digest"] = str(build["digest"])
            transaction["native_extensions"] = list(
                build.get("native_extensions") or []
            )
            write_json(PENDING_FILE, transaction)
            migration_result, diagnostic = _run_orm_migration("apply")
            with transaction_lock():
                updated = pending_transaction()
                if (
                    not updated
                    or updated.get("revision") != transaction.get("revision")
                    or updated.get("parent_operation_id")
                    != transaction.get("parent_operation_id")
                    or updated.get("generation") != build["generation"]
                ):
                    raise LayerBuildError("dependency_transaction_changed")
                transaction = updated
            if migration_result:
                transaction["error"] = diagnostic
                transaction["state"] = (
                    "migration_blocked"
                    if migration_result == 3
                    else "recovery_required"
                    if migration_result == 5
                    else "failed"
                )
                transaction["error_code"] = (
                    "external_database_manual_migration_required"
                    if migration_result == 3
                    else "database_migration_failed"
                )
                if migration_result == 2:
                    transaction["rollback_confirmed"] = True
                write_json(PENDING_FILE, transaction)
                return False
            migrated = True
        commit_generation(transaction, build, verify_on_start=True)
    except Exception as error:
        transaction["state"] = "failed"
        transaction["rollback_confirmed"] = not migrated and load_manifest().get(
            "active_generation"
        ) == (transaction.get("base_manifest") or {}).get("active_generation")
        transaction["error_code"] = getattr(error, "code", type(error).__name__)
        transaction["error"] = safe_process_error(str(error))
        transaction["error_details"] = json.loads(
            json.dumps(getattr(error, "details", None), default=str)
        )
        write_json(PENDING_FILE, transaction)
        return False
    transaction["state"] = "verification_pending"
    transaction["generation"] = build["generation"]
    write_json(PENDING_FILE, transaction)
    return True


def load_managed_plugins() -> dict[str, Any]:
    """Load managed modules without allowing one broken plugin to stop the worker."""
    manifest = load_manifest()
    generation = manifest.get("active_generation")
    loaded: list[str] = []
    failed: list[dict[str, str]] = []
    if generation is not None:
        for store_key, plugin in sorted(manifest.get("plugins", {}).items()):
            if not isinstance(plugin, dict) or plugin.get("state") != "managed":
                continue
            module_name = str(plugin.get("module_name") or "")
            if not module_name:
                failed.append({"store_key": store_key, "code": "module_name_missing"})
                continue
            try:
                _load_managed_plugin(module_name)
                loaded.append(module_name)
            except Exception as error:
                from nonebot import logger

                logger.opt(exception=error).error(
                    f"WebUI 托管的 NoneBot 插件 {module_name} 加载失败"
                )
                failure = _safe_plugin_import_failure(error)
                failed.append(
                    {
                        "store_key": store_key,
                        "module_name": module_name,
                        **failure,
                    }
                )
    status = {
        "generation": generation,
        "loaded": loaded,
        "failed": failed,
        "checked_at": utc_now(),
    }
    if manifest.get("pending_verification"):
        from .dependencies import DependencyAnalysisError
        from .environment import base_paths, capture_environment

        try:
            paths = base_paths()
            snapshot = capture_environment(manifest=manifest, paths=paths)
            status["loaded_mismatches"] = snapshot.loaded_mismatches
            status["environment_paths"] = paths
            status["environment_versions"] = snapshot.versions
        except DependencyAnalysisError as error:
            status["environment_error"] = {"code": error.code, "details": error.details}
    write_json(STARTUP_STATUS_FILE, status)
    return status


def _load_managed_plugin(module_name: str, *, purge: bool = True) -> None:
    """Load with NoneBot's finder while preserving the original import error."""
    from nonebot.plugin import _managers
    from nonebot.plugin.manager import PluginManager
    from nonebot.plugin.model import Plugin

    manager = PluginManager([module_name])
    _managers.append(manager)
    try:
        module = importlib.import_module(module_name)
        plugin = getattr(module, "__plugin__", None)
        if not isinstance(plugin, Plugin):
            raise RuntimeError("nonebot_plugin_registration_missing")
    except BaseException:
        with contextlib.suppress(ValueError):
            _managers.remove(manager)
        if purge:
            from zhenxun.services.plugin_host.host import purge_failed_import

            purge_failed_import(module_name)
        raise


_LOCALSTORE_DEFAULT_PATH_KEY = "I_AM_SURE_I_WANT_TO_USE_DEFAULT_LOCALSTORE_FILE_PATH"


def missing_config_key(error: BaseException) -> str | None:
    """导入失败原因是缺少 NoneBot 全局配置项时，返回对应的环境变量名

    包括读取了未配置的配置项，以及 cookit 系插件拒绝使用 localstore 默认存储位置。

    参数:
        error: 导入时的异常

    返回:
        str | None: 环境变量名
    """
    from nonebot.config import Config

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, AttributeError)
            and isinstance(getattr(current, "obj", None), Config)
            and current.name
        ):
            return current.name.upper()
        # 确认使用默认存储位置不会改变已安装插件的 localstore 数据目录
        if isinstance(current, RuntimeError) and _LOCALSTORE_DEFAULT_PATH_KEY in str(
            current
        ):
            return _LOCALSTORE_DEFAULT_PATH_KEY
        current = current.__cause__ or current.__context__
    return None


_PROBE_MARKER = "ZX_CONFIG_PROBE "


def probe_plugin_import(module_name: str, layer: Path) -> dict[str, Any] | None:
    """在独立子进程中按正式启动的配置试导入候选依赖层里的插件，返回导入失败说明

    超时或子进程异常时不作判断，交由重启后的启动验证处理。

    参数:
        module_name: 插件模块名
        layer: 候选依赖层目录

    返回:
        dict[str, Any] | None: 导入失败说明，格式同启动状态中的加载失败项
    """
    code = (
        "import sys;"
        "from zhenxun.services.nonebot_store.runtime import _probe_config_main;"
        "_probe_config_main(sys.argv[1], sys.argv[2])"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, module_name, str(layer)],
            cwd=str(INSTANCE_ROOT),
            env={**os.environ, "PYTHONPATH": subprocess_pythonpath()},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(_PROBE_MARKER):
            with contextlib.suppress(ValueError):
                failure = json.loads(line[len(_PROBE_MARKER) :]).get("failure")
                return failure if isinstance(failure, dict) else None
    return None


def _probe_config_main(module_name: str, layer: str) -> None:
    """试导入子进程入口，结果以标记行写入标准输出后立即退出

    参数:
        module_name: 插件模块名
        layer: 候选依赖层目录
    """
    failure = None
    try:
        sys.path.insert(0, layer)
        import nonebot

        from zhenxun.configs.environment import environment_file, migrate_orm_data_dir

        nonebot.init(_env_file=environment_file())
        migrate_orm_data_dir()
        _load_managed_plugin(module_name, purge=False)
    except Exception as error:
        failure = {"module_name": module_name, **_safe_plugin_import_failure(error)}
    sys.stdout.write(f"\n{_PROBE_MARKER}{json.dumps({'failure': failure})}\n")
    sys.stdout.flush()
    os._exit(0)


def import_error_summary(error: BaseException) -> str:
    """把插件导入异常整理为脱敏后的简短说明

    参数:
        error: 导入时的异常

    返回:
        str: 异常类型与说明
    """
    text = safe_process_error(str(error))[:300]
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _safe_plugin_import_failure(error: Exception) -> dict[str, Any]:
    """把插件导入失败整理为错误码、缺少或无效的配置项与异常摘要

    参数:
        error: 导入时的异常

    返回:
        dict[str, Any]: code、paths，无法归为配置问题时附带 detail
    """
    if key := missing_config_key(error):
        return {"code": "plugin_configuration_required", "paths": [key]}
    current: BaseException | None = error
    seen: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, ImportError):
            return {
                "code": "dependency_import_failed",
                "paths": [],
                "detail": import_error_summary(current),
            }
        errors = getattr(current, "errors", None)
        if callable(errors) and type(current).__module__.startswith("pydantic"):
            try:
                entries = errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                )
            except (TypeError, ValueError):
                entries = []
            paths: list[str] = []
            missing = False
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                missing = missing or entry.get("type") == "missing"
                loc = entry.get("loc")
                if not isinstance(loc, list | tuple):
                    continue
                parts = [
                    str(part)
                    for part in loc
                    if isinstance(part, str | int) and len(str(part)) <= 80
                ]
                if parts:
                    paths.append(".".join(parts))
            return {
                "code": (
                    "plugin_configuration_required"
                    if missing
                    else "plugin_configuration_invalid"
                ),
                "paths": sorted(set(paths))[:20],
            }
        current = current.__cause__ or current.__context__
    return {
        "code": "plugin_import_failed",
        "paths": [],
        "detail": import_error_summary(error),
    }


def issue_text(item: dict[str, Any]) -> str:
    """把启动验证中的问题整理为简短说明

    参数:
        item: 启动状态中 failures / warnings 的一项

    返回:
        str: 说明
    """
    code = str(item.get("code") or "")
    if item.get("module_name") or item.get("store_key"):
        name = item.get("module_name") or item.get("store_key")
        if code == "plugin_configuration_required" and item.get("paths"):
            return f"插件 {name} 缺少配置项 {'、'.join(map(str, item['paths']))}"
        detail = f"：{item['detail']}" if item.get("detail") else ""
        return f"插件 {name} 加载失败（{code}）{detail}"
    if code == "loaded_source_mismatch":
        return (
            f"{item.get('name')} 已从 {item.get('loaded_path')} 加载，"
            "依赖层中的版本未生效"
        )
    if code == "plugin_not_loaded":
        return f"插件未加载：{'、'.join(map(str, item.get('modules') or []))}"
    parts = []
    details = item.get("details")
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        actual = detail.get("actual") or "未安装"
        if detail.get("requirement"):
            parts.append(
                f"{detail.get('owner')} 需要 {detail['requirement']}，当前为 {actual}"
            )
        elif detail.get("expected"):
            parts.append(
                f"{detail.get('name')} 应为 {detail['expected']}，当前为 {actual}"
            )
    return f"{code}：{'；'.join(parts[:5])}" if parts else code


def startup_verification() -> tuple[bool, dict[str, Any]]:
    """检查重启后依赖层与托管插件的加载结果

    worker 未使用当前依赖层、依赖层文件与记录不符、核心依赖被改动或未从依赖层加载、
    运行环境在加载后发生变化时验证失败。插件加载失败、非核心依赖冲突与非核心包未从
    依赖层加载记为告警：恢复备份的事务保留依赖层并记录告警，其他事务只允许本次新装
    插件与此前已记录插件的加载失败，新装插件保留安装、跳过加载。

    返回:
        tuple[bool, dict[str, Any]]: 是否通过，启动状态；failures 为导致验证失败的
            问题，warnings 为保留依赖层时记录的告警
    """
    from .dependencies import DependencyAnalysisError
    from .environment import capture_environment, validate_candidate

    manifest = load_manifest()
    status = read_json(STARTUP_STATUS_FILE, {})
    transaction = read_json(PENDING_FILE, {})
    expected = manifest.get("active_generation")
    core = protected_core()
    fatal: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def conflicts(code: str, details) -> None:
        items = details if isinstance(details, list) else []
        broken = [
            item
            for item in items
            if isinstance(item, dict) and item.get("name") in core
        ]
        rest = [item for item in items if item not in broken]
        if broken:
            fatal.append({"code": code, "details": broken})
        if rest or not items:
            warnings.append({"code": code, "details": rest or details})

    failed = status.get("failed")
    if status.get("generation") != expected or not isinstance(failed, list):
        fatal.append({"code": "dependency_generation_not_loaded"})
        failed = []
    failed = [item for item in failed if isinstance(item, dict)]
    warnings.extend(failed)
    managed = {
        str(item.get("module_name") or "")
        for item in manifest.get("plugins", {}).values()
        if item.get("state") == "managed"
    }
    attempted = set(status.get("loaded", [])) | {
        str(item.get("module_name") or "") for item in failed
    }
    if not managed <= attempted:
        fatal.append(
            {"code": "plugin_not_loaded", "modules": sorted(managed - attempted)}
        )
    try:
        if isinstance(expected, int):
            if layer_digest(generation_path(expected)) != manifest.get(
                "generation_digest"
            ):
                raise DependencyAnalysisError("dependency_layer_state_mismatch")
            check = validate_candidate(
                generation_path(expected), {**transaction, "target_manifest": manifest}
            )
        else:
            snapshot = capture_environment(manifest=manifest)
            check = {
                "state": "failed" if snapshot.conflicts else "passed",
                "remaining_conflicts": snapshot.conflicts,
            }
        status["dependency_verification"] = check
        if check["state"] == "failed":
            conflicts(
                "current_environment_dependency_conflict", check["remaining_conflicts"]
            )
        paths = status.get("environment_paths")
        snapshot = capture_environment(
            manifest=manifest,
            paths=[str(item) for item in paths] if isinstance(paths, list) else None,
        )
        loaded_versions = status.get("environment_versions")
        if status.get("environment_error"):
            fatal.append(dict(status["environment_error"]))
        elif loaded_versions != snapshot.versions:
            loaded_versions = (
                loaded_versions if isinstance(loaded_versions, dict) else {}
            )
            names = set(loaded_versions) | set(snapshot.versions)
            fatal.append(
                {
                    "code": "environment_changed_after_load",
                    "details": [
                        {
                            "name": name,
                            "expected": loaded_versions.get(name) or "未安装",
                            "actual": snapshot.versions.get(name),
                        }
                        for name in sorted(names)
                        if loaded_versions.get(name) != snapshot.versions.get(name)
                    ],
                }
            )
        for item in status.get("loaded_mismatches") or []:
            (fatal if item.get("name") in core else warnings).append(
                {"code": "loaded_source_mismatch", **item}
            )
    except DependencyAnalysisError as error:
        status["dependency_verification"] = {
            "state": "failed",
            "code": error.code,
            "details": error.details,
        }
        if error.code == "current_environment_dependency_conflict":
            conflicts(error.code, error.details)
        else:
            fatal.append({"code": error.code, "details": error.details})
    known = set(manifest.get("load_failures") or {}) | {
        str(item.get("module_name"))
        for item in transaction.get("operations") or []
        if isinstance(item, dict)
        and item.get("action") == "install"
        and item.get("module_name")
    }
    rejected = (
        []
        if transaction.get("restore")
        else [item for item in warnings if item.get("module_name") not in known]
    )
    status["failures"] = [*fatal, *rejected]
    status["warnings"] = [item for item in warnings if item not in rejected]
    status["plugin_verification"] = "passed" if not failed else "failed"
    write_json(STARTUP_STATUS_FILE, status)
    return not status["failures"], status


def finalize_pending_transaction() -> None:
    """持久化验证与提交证据，ORM 清理在文件锁外完成并可重放"""
    import psutil

    with transaction_lock():
        transaction = pending_transaction()
        if not transaction:
            return
        if transaction.get("state") not in {
            "verification_pending",
            "settling",
            "committed",
            "cleanup_pending",
        }:
            raise LayerBuildError("dependency_verification_not_confirmed")
        if transaction.get("state") == "settling":
            executor = transaction.get("settlement_executor") or {}
            try:
                process = psutil.Process(int(executor["pid"]))
                if process.create_time() == executor["created_at"]:
                    raise LayerBuildError("dependency_settlement_in_progress")
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, KeyError, TypeError, ValueError) as error:
                raise LayerBuildError("dependency_executor_unconfirmed") from error
        manifest = load_manifest()
        manifest["pending_verification"] = False
        manifest["previous_generation"] = None
        if transaction.get("revision"):
            manifest["verified_transaction_revision"] = transaction["revision"]
        status = read_json(STARTUP_STATUS_FILE, {})
        if status.get("generation") == manifest.get("active_generation"):
            manifest["load_failures"] = {
                str(item["module_name"]): str(item.get("code") or "")
                for item in status.get("failed") or []
                if isinstance(item, dict) and item.get("module_name")
            }
            if status.get("warnings"):
                transaction["warnings"] = [
                    issue_text(item) for item in status["warnings"][:20]
                ]
        save_manifest(manifest)
        transaction.update(state="settling", verification_confirmed=True)
        transaction.pop("cleanup_error", None)
        transaction["settlement_executor"] = {
            "pid": os.getpid(),
            "created_at": psutil.Process().create_time(),
        }
        write_json(PENDING_FILE, transaction)
    cleanup_error = None
    try:
        finalize_orm_migration()
    except Exception as error:
        cleanup_error = {
            "code": getattr(error, "code", type(error).__name__),
            "diagnostic": safe_process_error(str(error)),
        }
    with transaction_lock():
        current = pending_transaction()
        if not current or any(
            current.get(key) != transaction.get(key)
            for key in ("revision", "record_revision", "generation")
        ):
            raise LayerBuildError("dependency_transaction_changed")
        transaction = current
        transaction.update(state="committed", verification_confirmed=True)
        try:
            write_json(PENDING_FILE, transaction)
            record_manual_operations(transaction, "applied", "completed")
            if cleanup_error is None:
                active = load_manifest().get("active_generation")
                retained = referenced_generations()
                if isinstance(active, int):
                    retained.add(active)
                prune_generations(retained)
                ROLLBACK_FILE.unlink(missing_ok=True)
        except Exception as error:
            cleanup_error = {
                "code": getattr(error, "code", type(error).__name__),
                "diagnostic": safe_process_error(str(error)),
            }
        if cleanup_error:
            transaction.update(state="cleanup_pending", cleanup_error=cleanup_error)
            write_json(PENDING_FILE, transaction)
        from .storage import archive_settled

        archive_settled()


def rollback_pending_transaction(failed: list[dict[str, Any]] | None = None) -> None:
    """回滚待验证的依赖层事务并记录失败原因

    worker 未能完成启动时记录为 worker_startup_failed，否则记录启动验证中导致
    失败的问题及说明。

    参数:
        failed: launcher 记录的失败项
    """
    from zhenxun.utils.atomic_json import read_json_locked

    with transaction_lock():
        transaction = pending_transaction()
        if not transaction:
            return
        if transaction.get("state") == "rolling_back":
            raise LayerBuildError("dependency_rollback_in_progress")
        rollback = read_json_locked(ROLLBACK_FILE, None, strict=True)
        if not transaction.get("rollback_confirmed") and not isinstance(rollback, dict):
            transaction.update(
                state="recovery_required",
                error_code="dependency_rollback_receipt_missing",
            )
            write_json(PENDING_FILE, transaction)
            raise LayerBuildError("dependency_rollback_receipt_missing")
        transaction["state"] = "rolling_back"
        transaction["rollback_attempt"] = uuid4().hex
        write_json(PENDING_FILE, transaction)
    binding = ("revision", "parent_operation_id", "generation", "rollback_attempt")
    try:
        code, diagnostic = (
            (0, "")
            if transaction.get("rollback_confirmed")
            else _run_orm_migration("restore")
        )
        with transaction_lock():
            current = pending_transaction()
            if (
                not current
                or any(current.get(key) != transaction.get(key) for key in binding)
                or current.get("state") != "rolling_back"
            ):
                raise LayerBuildError("dependency_transaction_changed")
            if code:
                raise LayerBuildError("database_migration_rollback_failed", diagnostic)
            _finish_rollback(current, rollback, failed)
    except Exception as error:
        with transaction_lock():
            current = pending_transaction()
            if (
                current
                and all(current.get(key) == transaction.get(key) for key in binding)
                and current.get("state") == "rolling_back"
                and not current.get("rollback_confirmed")
            ):
                current.update(
                    state="recovery_required",
                    error_code=getattr(error, "code", type(error).__name__),
                    rollback_error=safe_process_error(str(error)),
                )
                write_json(PENDING_FILE, current)
                with contextlib.suppress(Exception):
                    record_manual_operations(
                        current, "recovery_required", "recovery_required"
                    )
        raise


def _finish_rollback(transaction: dict, rollback: dict | None, failed: list | None):
    """核对 ORM 子进程结果后结算依赖回滚与本任务清理"""
    current = load_manifest()
    failed_generation = transaction.get("generation", current.get("active_generation"))
    failed_status = read_json(STARTUP_STATUS_FILE, {})
    if isinstance(rollback, dict) and not transaction.get("rollback_confirmed"):
        save_manifest(rollback)
    worker_failed = any(
        item.get("code") == "worker_health_check_failed" for item in failed or []
    )
    if transaction:
        transaction["state"] = "failed"
        transaction["rollback_confirmed"] = True
        transaction["error_code"] = (
            "worker_startup_failed"
            if worker_failed
            else "plugin_startup_verification_failed"
        )
        failures = [
            item
            for item in failed_status.get("failures") or []
            if isinstance(item, dict)
        ][:10]
        if worker_failed:
            transaction["failure_reasons"] = [{"code": "worker_startup_failed"}]
        else:
            transaction["failure_reasons"] = [
                {
                    "code": str(item.get("code") or "plugin_import_failed"),
                    "store_key": str(item.get("store_key") or ""),
                    "module_name": str(item.get("module_name") or ""),
                    "paths": [
                        str(path)
                        for path in item.get("paths", [])
                        if isinstance(path, str)
                    ][:20],
                    "detail": issue_text(item),
                }
                for item in failures
            ] or [{"code": "plugin_startup_verification_failed"}]
            transaction["error_details"] = [
                item["detail"]
                for item in transaction["failure_reasons"]
                if item.get("detail")
            ]
        write_json(PENDING_FILE, transaction)
        record_manual_operations(transaction, "rolled_back", "failed")
    restored = load_manifest().get("active_generation")
    try:
        if failed_generation != restored:
            remove_generation(failed_generation)
        prune_generations({restored} if isinstance(restored, int) else set())
        ROLLBACK_FILE.unlink(missing_ok=True)
        remove_orm_backups()
        if transaction.pop("cleanup_error", None):
            write_json(PENDING_FILE, transaction)
    except OSError as error:
        transaction["cleanup_error"] = {
            "code": "dependency_rollback_cleanup_pending",
            "diagnostic": safe_process_error(str(error)),
        }
        with contextlib.suppress(OSError):
            write_json(PENDING_FILE, transaction)
