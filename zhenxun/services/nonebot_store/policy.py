"""Shared boundaries for scoped dependency changes and compatibility trials."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

# nonebot-plugin-orm 的数据库迁移依赖的包，已安装时只能在控制台改变版本
ORM_PACKAGES = frozenset({"nonebot-plugin-orm", "sqlalchemy", "alembic"})


def guard_orm_changes(before: dict, after: dict) -> None:
    """已安装的 ORM 包只能在控制台改变版本，新增与移除不受限制

    参数:
        before: 变更前的包版本
        after: 变更后的包版本

    异常:
        DependencyAnalysisError: 已安装的 ORM 包版本会变化
    """
    from .dependencies import DependencyAnalysisError

    changes = [
        {"name": name, "from": before.get(name), "to": after.get(name)}
        for name in sorted(ORM_PACKAGES)
        if before.get(name) and after.get(name) and before[name] != after[name]
    ]
    if changes:
        raise DependencyAnalysisError(
            "orm_dependency_console_required", details=changes
        )


def guard_orm_request(names, installed) -> None:
    """已安装的 ORM 包只能在控制台安装、升级或卸载

    参数:
        names: 本次操作的包名
        installed: 当前已安装的包

    异常:
        DependencyAnalysisError: 操作涉及已安装的 ORM 包
    """
    from .dependencies import DependencyAnalysisError

    packages = sorted(set(names) & ORM_PACKAGES & set(installed))
    if packages:
        raise DependencyAnalysisError(
            "orm_dependency_console_required", details=packages
        )


def scoped_roots(requirements, scope) -> list[str]:
    return [
        raw for raw in requirements if canonicalize_name(Requirement(raw).name) in scope
    ]


def retained_conflicts(snapshot, affected) -> list[dict]:
    """Carry only unchanged issues outside the mutation's consumers and packages."""
    affected = set(affected)
    return [
        {**item, "baseline": conflict_evidence(snapshot, item)}
        for item in snapshot.conflicts
        if item["owner"] not in affected
        and item["name"] not in affected
        and not item.get("kind")
    ]


def external_requirements(snapshot, scope, replaced=frozenset()) -> list[str]:
    """解析范围外的来源对范围内包的依赖声明，已存在的冲突不计入

    参数:
        snapshot: 环境快照
        scope: 解析范围内的包
        replaced: 声明不再参与解析的来源

    返回:
        list[str]: 依赖声明
    """
    conflicts = {(i["owner"], i["requirement"]) for i in snapshot.conflicts}
    return [
        item["requirement"]
        for item in snapshot.declarations
        if item["name"] in scope
        and item["owner"] not in scope
        and item["owner"] not in replaced
        and not item["owner"].startswith("manual:")
        and (item["owner"], item["requirement"]) not in conflicts
    ]


def conflict_evidence(snapshot, issue: dict) -> dict:
    """记录冲突双方当前的版本与来源

    参数:
        snapshot: 环境快照
        issue: 依赖冲突

    返回:
        dict: 包名到版本与来源的映射
    """
    return {
        name: {
            "version": snapshot.versions.get(name),
            "source": snapshot.sources.get(name, {}).get("layer"),
        }
        for name in {issue["owner"], issue["name"]}
    }


def _same_state(recorded: dict | None, current: dict) -> bool:
    """只按当前记录的键比较已保存状态

    参数:
        recorded: 已保存的状态
        current: 当前状态

    返回:
        bool: 当前状态中的每个键是否都与已保存状态一致
    """
    return isinstance(recorded, dict) and all(
        key in recorded and recorded[key] == value for key, value in current.items()
    )


def unchanged_conflict(snapshot, issue: dict, permitted: list[dict]) -> bool:
    """判断冲突是否为已允许且双方版本与来源均未变化的冲突

    参数:
        snapshot: 环境快照
        issue: 依赖冲突
        permitted: 已允许保留的冲突

    返回:
        bool: 是否可以继续保留
    """
    from .environment import conflict_key

    evidence = conflict_evidence(snapshot, issue)
    return any(
        conflict_key(item) == conflict_key(issue)
        and all(
            _same_state((item.get("baseline") or {}).get(name), value)
            for name, value in evidence.items()
        )
        for item in permitted
    )


def trial_evidence(
    name: str,
    owner_version: str | None,
    owner_source: str | None,
    version: str | None,
    source: str | None,
) -> dict | None:
    """记录当前包的适用规则和双方身份，与安装计划的全局修订号无关。"""
    from .dependencies import (
        FORBIDDEN_LAYER_PACKAGES,
        _project_requirement_map,
        dependency_floors,
        protected_core,
    )

    if (
        name in ORM_PACKAGES
        or name in FORBIDDEN_LAYER_PACKAGES
        or name in protected_core()
    ):
        return None
    floor = dependency_floors().get(name)
    project = _project_requirement_map().get(name)
    bounds = SpecifierSet(floor or "")
    if project:
        bounds &= project.specifier
    if (
        not version
        or not owner_version
        or owner_source not in {"active_generation", "base"}
        or source not in {"active_generation", "base"}
        or not bounds.contains(version, prereleases=True)
    ):
        return None
    return {
        "schema": 1,
        "constraint": str(bounds) if floor is not None or project else None,
        "owner_version": owner_version,
        "owner_source": owner_source,
        "dependency_version": version,
        "dependency_source": source,
    }


def unresolved_conflict(
    snapshot, conflict: dict, permitted: list[dict], *, migrations=None
) -> dict | None:
    """兼容例外按批准时的证据核验；其余历史冲突按双方基线核验。"""
    from .dependencies import SOLVER_POLICY_VERSION

    owner, name = conflict["owner"], conflict["name"]
    matched = False
    for plugin in snapshot.manifest.get("plugins", {}).values():
        if canonicalize_name(plugin["project_link"]) != owner:
            continue
        for item in plugin.get("compatibility_overrides", []):
            if item.get("name") != name:
                continue
            matched = True
            evidence = item.get("evidence")
            if (
                not isinstance(evidence, dict)
                or not evidence
                or conflict.get("kind")
                or not conflict.get("actual")
                or item.get("effective_version") != conflict["actual"]
            ):
                continue
            try:
                if Requirement(
                    item.get("requirement") or item.get("declared_requirement") or ""
                ) != Requirement(conflict["requirement"]):
                    continue
            except InvalidRequirement:
                continue
            expected = trial_evidence(
                name,
                snapshot.versions.get(owner),
                snapshot.sources.get(owner, {}).get("layer"),
                snapshot.versions.get(name),
                snapshot.sources.get(name, {}).get("layer"),
            )
            if not expected:
                continue
            if evidence == expected:
                return None
            # 旧收据仅在其已知策略下转换；声明方的旧来源由托管插件清单佐证。
            legacy = evidence.get("policy_version")
            if (
                "schema" not in evidence
                and SOLVER_POLICY_VERSION == 13
                and (legacy == 13 or (legacy == 12 and "blitz-py" not in {owner, name}))
                and expected["owner_source"] == "active_generation"
                and snapshot.manifest.get("packages", {}).get(owner, {}).get("version")
                == plugin.get("version")
                == expected["owner_version"]
                and item.get("tier")
                == ("shared_compatible" if expected["constraint"] else "plugin_private")
                and _same_state(
                    evidence,
                    {
                        k: expected[k]
                        for k in (
                            "owner_version",
                            "dependency_version",
                            "dependency_source",
                        )
                    },
                )
            ):
                if migrations is not None:
                    migrations.append((item, expected))
                return None
    if matched:
        return {**conflict, "reason": "compatibility_trial_recheck_required"}
    return None if unchanged_conflict(snapshot, conflict, permitted) else conflict


def summary_revision(value: dict) -> str:
    """Fingerprint user-visible effects independently of transient preview IDs."""
    keys = (
        "changes",
        "retained",
        "manual_requirements",
        "package_changes",
        "compatibility_overrides",
        "source_build_required",
        "database_type",
        "database_migration_possible",
        "warnings",
        "restart_required",
        "resolved_packages",
        "solver_policy_version",
        "plugin",
        "artifact_hashes",
    )
    return sha256(
        json.dumps(
            {key: value.get(key) for key in keys}, sort_keys=True, default=str
        ).encode()
    ).hexdigest()


def orm_console_instructions(transaction: dict) -> dict | None:
    if transaction.get("state") != "migration_blocked":
        return None
    arguments = [
        sys.executable,
        "-m",
        "zhenxun.services.nonebot_store.orm_migration",
        "manual-upgrade",
        "--revision",
        transaction["revision"],
        "--backup-confirmed",
    ]
    command = (
        ("& " + " ".join("'" + part.replace("'", "''") + "'" for part in arguments))
        if os.name == "nt"
        else shlex.join(arguments)
    )
    return {
        "working_directory": str(Path.cwd().resolve()),
        "command": command,
        "requires_stopped_instance": True,
        "requires_database_backup": True,
    }
