"""Distribution views shared by dependency analysis and generation verification."""

from __future__ import annotations

import ast
from collections.abc import Iterable
import csv
from dataclasses import dataclass
from functools import cached_property
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from urllib.parse import urldefrag

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .storage import LAYER_ROOT, generation_path, load_manifest

SOURCE_PLUGIN_ROOT = Path("zhenxun") / "plugins"
REQUIREMENT_FILES = ("requirement.txt", "requirements.txt")
"""源码插件声明依赖的文件名"""


def _invalid(owner: str, value: str):
    from .dependencies import DependencyAnalysisError

    raise DependencyAnalysisError(
        "dependency_metadata_invalid",
        details={"owner": owner, "declaration": value[:200]},
    )


class _CachedDistribution(metadata.PathDistribution):
    """只解析一次元数据的发行包"""

    @cached_property
    def metadata(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().metadata


def _recorded_files(distribution) -> list:
    """读取包安装记录中的文件路径，不逐个检查文件是否存在

    参数:
        distribution: 已安装的包

    返回:
        list: 相对包安装目录的文件路径
    """
    text = distribution.read_text("RECORD")
    if text is None:
        return list(distribution.files or ())
    return [PurePosixPath(row[0]) for row in csv.reader(text.splitlines()) if row]


def distributions(paths=None) -> tuple[dict, dict]:
    """Select the first distribution on each search path, retaining duplicates."""
    selected, duplicates = {}, {}
    for distribution in metadata.distributions(
        path=paths if paths is not None else sys.path
    ):
        if not isinstance(distribution, _CachedDistribution) and hasattr(
            distribution, "_path"
        ):
            distribution = _CachedDistribution(distribution._path)  # pyright: ignore[reportAttributeAccessIssue]
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        try:
            name = canonicalize_name(raw_name, validate=True)
            Version(distribution.version)
        except (ValueError, TypeError):
            _invalid(str(raw_name), str(distribution.version))
        if name in selected:
            if distribution.version == selected[name].version and str(
                getattr(distribution, "_path", distribution.locate_file(""))
            ) == str(getattr(selected[name], "_path", selected[name].locate_file(""))):
                continue
            duplicates.setdefault(name, []).append(str(distribution.locate_file("")))
        else:
            selected[name] = distribution
    return selected, duplicates


def base_paths() -> list[str]:
    """当前进程 sys.path 中核心虚拟环境的搜索路径

    不含依赖层，也不含包内自带的私有依赖目录（如 setuptools 导入时追加的
    setuptools/_vendor）。

    返回:
        list[str]: 搜索路径
    """
    root = LAYER_ROOT.resolve()
    paths = []
    for value in sys.path:
        path = Path(value).resolve()
        if path.is_relative_to(root) or (path.parent / "__init__.py").is_file():
            continue
        paths.append(value)
    return paths


def base_distributions(paths: list[str] | None = None) -> tuple[dict, dict]:
    """核心虚拟环境中的包，不含依赖层

    参数:
        paths: 搜索路径，为空时使用当前进程的 base_paths

    返回:
        tuple[dict, dict]: 包名 -> 已安装的包，以及重复安装的包所在路径
    """
    return distributions(base_paths() if paths is None else paths)


def dependency_roots(
    manifest: dict[str, Any], sources: dict[str, list[str]] | None = None
) -> list[tuple[str, str]]:
    """收集各依赖来源声明的依赖

    来源包括核心 pyproject.toml、手动依赖、插件 requirements 文件、托管的 NoneBot
    插件，以及传入的商店插件依赖。

    参数:
        manifest: 依赖层清单
        sources: 商店插件依赖，键为 store_key

    返回:
        list[tuple[str, str]]: (来源, 依赖声明) 列表
    """
    from .dependencies import project_requirements
    from .manual import manual_requirements

    roots = [("pyproject.toml", raw) for raw in project_requirements()]
    roots.extend(
        (f"manual:{name}", raw)
        for name, requirements in manual_requirements(manifest).items()
        for raw in requirements
    )
    roots.extend(
        (str(path), str(raw))
        for path, requirements in sorted(
            (manifest.get("plugin_requirements") or {}).items()
        )
        for raw in requirements
    )
    for plugin in manifest.get("plugins", {}).values():
        if plugin.get("state") == "managed":
            roots.extend(
                (canonicalize_name(plugin["project_link"]), raw)
                for raw in (
                    plugin.get("resolution_inputs")
                    or [f"{plugin['project_link']}=={plugin['version']}"]
                )
            )
    for key, values in sorted((sources or {}).items()):
        roots.extend((f"source:{key}", str(raw)) for raw in values)
    return roots


def reachable_packages(roots: Iterable[str], selected: dict) -> set[str]:
    """从依赖声明出发，沿已安装包的依赖元数据计算可达的包

    参数:
        roots: 依赖声明
        selected: 已安装的包

    返回:
        set[str]: 可达的规范化包名
    """
    active: dict[str, set[str]] = {}
    pending: list[tuple[str, frozenset[str]]] = [
        (raw, frozenset({""})) for raw in roots
    ]
    while pending:
        raw, contexts = pending.pop()
        try:
            requirement = Requirement(raw)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in contexts
            ):
                continue
        except (InvalidRequirement, ValueError):
            continue
        name = canonicalize_name(requirement.name)
        wanted = {""} | {canonicalize_name(extra) for extra in requirement.extras}
        new = wanted - active.setdefault(name, set())
        if not new:
            continue
        active[name].update(new)
        if (distribution := selected.get(name)) is not None:
            pending.extend(
                (value, frozenset(new)) for value in distribution.requires or []
            )
    return set(active)


def distribution_modules(distribution) -> set[str]:
    """读取包提供的顶层模块名

    参数:
        distribution: 已安装的包

    返回:
        set[str]: 顶层模块名
    """
    text = distribution.read_text("top_level.txt")
    if text:
        modules = {line.strip() for line in text.splitlines()}
    else:
        modules = set()
        for file in distribution.files or ():
            parts = file.parts
            if not parts or parts[0] == "..":
                continue
            if parts[0].endswith((".dist-info", ".data")):
                continue
            modules.add(parts[0] if len(parts) > 1 else parts[0].split(".")[0])
    return {module for module in modules if module.isidentifier()}


def imported_modules(
    paths: Iterable[Path], excluded: Iterable[Path] = ()
) -> dict[str, str]:
    """静态扫描源码导入的顶层模块，不执行任何代码

    参数:
        paths: 源码文件或目录
        excluded: 跳过的文件或目录

    返回:
        dict[str, str]: 顶层模块名到首个导入它的文件
    """
    skipped = [Path(path).resolve() for path in excluded]
    result: dict[str, str] = {}
    for root in paths:
        root = Path(root)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(root.rglob("*.py"))
        else:
            continue
        for file in files:
            resolved = file.resolve()
            if "__pycache__" in file.parts or any(
                resolved == item or resolved.is_relative_to(item) for item in skipped
            ):
                continue
            try:
                tree = ast.parse(file.read_bytes())
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if module := name.split(".")[0]:
                        result.setdefault(module, file.as_posix())
    return result


def _restorable_downgrades(snapshot, kept: set[str], roots: list[str]) -> list[str]:
    """依赖层中低于本体版本、且剩余依赖都接受本体版本的包

    参数:
        snapshot: 变化前的环境快照
        kept: 变化后仍被需要的包
        roots: 变化后各来源的依赖声明

    返回:
        list[str]: 可以从依赖层释放、改回本体版本的包
    """
    downgraded = {}
    for name, distribution in snapshot.layer.items():
        if (base := snapshot.base.get(name)) is None:
            continue
        try:
            if Version(distribution.version) < Version(base.version):
                downgraded[name] = base.version
        except InvalidVersion:
            continue
    if not downgraded:
        return []
    values = list(roots)
    for name in kept:
        if (distribution := snapshot.selected.get(name)) is not None:
            values.extend(distribution.requires or [])
    specifiers: dict[str, list[SpecifierSet]] = {name: [] for name in downgraded}
    for raw in values:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        if (name := canonicalize_name(requirement.name)) in specifiers:
            specifiers[name].append(requirement.specifier)
    return sorted(
        name
        for name, version in downgraded.items()
        if all(item.contains(version, prereleases=True) for item in specifiers[name])
    )


def released_packages(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    sources_before: dict[str, list[str]] | None = None,
    sources_after: dict[str, list[str]] | None = None,
    removed_paths: Iterable[Path] = (),
    added_paths: Iterable[Path] = (),
) -> tuple[list[str], list[dict[str, Any]]]:
    """计算依赖来源变化后可以从依赖层释放的包

    只释放变化前有来源需要、变化后不再被任何来源需要的包；变化前就没有来源需要
    的包及其依赖视为保留来源。仍被源码插件导入的包不会释放。依赖层中低于本体
    版本的包，在变化后的依赖都接受本体版本时一并释放，改回使用本体版本。

    参数:
        before: 变化前的清单
        after: 变化后的清单
        sources_before: 变化前的商店插件依赖
        sources_after: 变化后的商店插件依赖
        removed_paths: 将被移除的插件源码，不参与导入扫描
        added_paths: 将被加入的插件源码

    返回:
        tuple[list[str], list[dict[str, Any]]]: 可释放的包，因仍被导入而保留的包
    """
    from .dependencies import protected_core
    from .policy import ORM_PACKAGES

    snapshot = capture_environment(manifest=before)
    selected = snapshot.selected
    layer = set(snapshot.layer)

    def roots(manifest: dict[str, Any], sources) -> list[str]:
        values = [raw for _, raw in dependency_roots(manifest, sources)]
        values.extend(
            str(plugin["project_link"])
            for plugin in manifest.get("plugins", {}).values()
            if plugin.get("state") == "managed"
        )
        return values

    needed = reachable_packages(roots(before, sources_before), selected)
    kept = reachable_packages(
        [*roots(after, sources_after), *sorted(layer - needed)], selected
    )
    candidates = (needed - kept) & layer - set(protected_core()) - ORM_PACKAGES
    released = _restorable_downgrades(snapshot, kept, roots(after, sources_after))
    if not candidates:
        return released, []
    imports = imported_modules(
        [SOURCE_PLUGIN_ROOT, *added_paths], excluded=removed_paths
    )
    retained: list[dict[str, Any]] = []
    for name in sorted(candidates - set(released)):
        modules = sorted(distribution_modules(selected[name]) & set(imports))
        if not modules:
            released.append(name)
            continue
        retained.append(
            {
                "name": name,
                "version": selected[name].version,
                "owners": [f"插件导入 {module}" for module in modules],
                "imported_modules": modules,
                "files": sorted({imports[module] for module in modules}),
            }
        )
    return sorted(released), retained


@dataclass
class EnvironmentSnapshot:
    manifest: dict
    base: dict
    layer: dict
    selected: dict
    declarations: list[dict]
    conflicts: list[dict]
    sources: dict
    duplicates: dict
    loaded_mismatches: list[dict]

    @property
    def versions(self) -> dict[str, str]:
        return {
            name: distribution.version for name, distribution in self.selected.items()
        }


def capture_environment(
    *,
    manifest=None,
    candidate_path: Path | None = None,
    paths: list[str] | None = None,
) -> EnvironmentSnapshot:
    """读取核心环境与依赖层的包元数据并收集各来源的依赖声明，不导入插件

    本体的依赖以 pyproject.toml 为准，不读取已安装本体元数据中的依赖声明。

    参数:
        manifest: 依赖层清单，为空时使用当前清单
        candidate_path: 代替激活代的候选依赖层目录
        paths: 核心虚拟环境的搜索路径，为空时使用当前进程的 base_paths

    返回:
        EnvironmentSnapshot: 环境快照
    """
    from .dependencies import PROJECT_NAME

    manifest = load_manifest() if manifest is None else manifest
    base, duplicates = base_distributions(paths)
    generation = manifest.get("active_generation")
    layer_path = candidate_path
    if layer_path is None and isinstance(generation, int):
        layer_path = generation_path(generation)
    layer, layer_duplicates = (
        distributions([str(layer_path)]) if layer_path is not None else ({}, {})
    )
    selected = {**base, **layer}

    roots = dependency_roots(manifest)
    requirements = {}
    for name, distribution in selected.items():
        if name != PROJECT_NAME:
            requirements[name] = list(distribution.requires or [])
    extras: dict[str, set[str]] = {}
    declarations: dict[tuple[str, str], dict] = {}
    pending = roots + [
        (owner, raw) for owner, values in requirements.items() for raw in values
    ]
    while True:
        expanded = False
        for owner, raw in pending:
            try:
                requirement = Requirement(raw)
                contexts = {"", *extras.get(owner, set())}
                if requirement.marker and not any(
                    requirement.marker.evaluate(
                        {**default_environment(), "extra": extra}
                    )
                    for extra in contexts
                ):
                    continue
            except (InvalidRequirement, ValueError):
                _invalid(owner, raw)
            name = canonicalize_name(requirement.name)
            new_extras = {
                canonicalize_name(extra) for extra in requirement.extras
            } - extras.setdefault(name, set())
            if new_extras:
                extras[name].update(new_extras)
                expanded = True
            declarations[owner, str(requirement)] = {
                "owner": owner,
                "name": name,
                "requirement": str(requirement),
                "owner_version": selected[owner].version if owner in selected else None,
            }
        if not expanded:
            break
    sources = {
        name: {
            "version": dist.version,
            "layer": "active_generation" if name in layer else "base",
            "path": str(dist.locate_file("")),
        }
        for name, dist in selected.items()
    }
    conflicts = []
    for declaration in declarations.values():
        name = declaration["name"]
        requirement = Requirement(declaration["requirement"])
        actual = selected[name].version if name in selected else None
        try:
            compatible = actual is not None and requirement.specifier.contains(
                actual, prereleases=True
            )
        except InvalidVersion:
            _invalid(name, str(actual))
        if requirement.url and name in selected:
            try:
                direct = json.loads(selected[name].read_text("direct_url.json") or "{}")
            except (ValueError, TypeError):
                _invalid(name, "direct_url.json")
            if urldefrag(requirement.url).url != urldefrag(direct.get("url", "")).url:
                conflicts.append(
                    {
                        **declaration,
                        "actual": actual,
                        "kind": "source_unverified",
                        "layer": sources[name]["layer"],
                    }
                )
        if not compatible:
            conflicts.append(
                {
                    **declaration,
                    "actual": actual,
                    "layer": "active_generation" if name in layer else "base",
                }
            )
    loaded = []
    for name, dist in selected.items():
        python = dist.metadata.get("Requires-Python")
        try:
            if python and not SpecifierSet(python).contains(
                default_environment()["python_full_version"], prereleases=True
            ):
                conflicts.append(
                    {
                        "owner": name,
                        "name": name,
                        "actual": dist.version,
                        "requirement": f"Python{python}",
                        "kind": "requires_python",
                        "layer": sources[name]["layer"],
                    }
                )
        except InvalidSpecifier:
            _invalid(name, python)
        for file in _recorded_files(dist):
            parts = file.parts
            if not (
                (len(parts) == 2 and parts[1] == "__init__.py")
                or (len(parts) == 1 and parts[0].endswith(".py"))
            ):
                continue
            module_name = parts[0].removesuffix(".py")
            module = sys.modules.get(module_name)
            actual_path = getattr(module, "__file__", None)
            if not actual_path:
                continue
            expected_path = Path(dist.locate_file(file)).resolve()
            if expected_path.exists() and Path(actual_path).resolve() != expected_path:
                loaded.append(
                    {
                        "name": name,
                        "module": module_name,
                        "loaded_path": str(actual_path),
                        "expected_path": str(expected_path),
                    }
                )
    return EnvironmentSnapshot(
        manifest,
        base,
        layer,
        selected,
        list(declarations.values()),
        conflicts,
        sources,
        {**duplicates, **layer_duplicates},
        loaded,
    )


def core_environment_summary(
    snapshot: EnvironmentSnapshot, protected: dict[str, str], report: dict
) -> dict:
    """Summarize protected packages without attributing plugin constraints to core."""
    relevant = set(protected)
    relevant.update(
        item["name"] for item in snapshot.declarations if item["owner"] in protected
    )
    issues = [{**item, "kind": "core_version"} for item in report["immutable_drift"]]
    issues.extend(
        {"kind": "requirement_conflict", **item}
        for item in snapshot.conflicts
        if item["owner"] in protected
        or (item["owner"] == "pyproject.toml" and item["name"] in protected)
    )
    uncertain = [item for item in issues if item["kind"] == "source_unverified"]
    for flag in ("interpreter_mismatch", "project_lock_stale"):
        if report[flag]:
            uncertain.append({"kind": flag})
    if not protected:
        uncertain.append({"kind": "core_scope_unavailable"})
    uncertain.extend(
        {**item, "kind": "layer_mismatch"}
        for item in report["layer_mismatch"]
        if item["name"] in relevant
    )
    for name in sorted(relevant):
        if name in snapshot.selected and not snapshot.sources.get(name, {}).get("path"):
            uncertain.append({"name": name, "kind": "source_unverified"})
        if snapshot.duplicates.get(name):
            uncertain.append({"name": name, "kind": "duplicate_metadata"})
    uncertain.extend(
        {**item, "kind": "loaded_source_mismatch"}
        for item in snapshot.loaded_mismatches
        if item["name"] in relevant
    )
    issues.extend(item for item in uncertain if item not in issues)
    return {
        "status": "unconfirmed"
        if uncertain
        else "incompatible"
        if issues
        else "healthy",
        "checked_count": len(protected),
        "issues": issues,
    }


def conflict_key(issue: dict) -> tuple:
    return tuple(
        issue.get(key)
        for key in (
            "owner",
            "owner_version",
            "name",
            "requirement",
            "actual",
            "kind",
            "layer",
        )
    )


def core_packages(snapshot: EnvironmentSnapshot) -> set[str]:
    """按核心虚拟环境中已安装的本体元数据计算本体依赖的包

    依赖同步不重装本体，已安装本体的元数据保留上次完整安装时的依赖声明。

    参数:
        snapshot: 环境快照

    返回:
        set[str]: 本体及其依赖的规范化包名
    """
    from .dependencies import PROJECT_NAME

    return reachable_packages([PROJECT_NAME], snapshot.base)


def dropped_core_packages(previous: set[str]) -> list[str]:
    """本体依赖变化后可以从核心虚拟环境移除的包

    只移除变化前属于本体依赖、变化后不再被任何依赖来源（含当前 pyproject.toml）
    需要的包；被源码插件导入的包，以及 ORM 包保留。

    参数:
        previous: 变化前本体依赖的包，见 core_packages

    返回:
        list[str]: 可以移除的包
    """
    from zhenxun.services.plugin_store.plugin_store_receipts import (
        receipt_dependency_inputs,
    )

    from .dependencies import PROJECT_NAME
    from .policy import ORM_PACKAGES

    snapshot = capture_environment()
    candidates = (previous - {PROJECT_NAME}) & set(snapshot.base) - ORM_PACKAGES
    if not candidates:
        return []
    roots = [raw for _, raw in dependency_roots(snapshot.manifest)]
    roots.extend(
        raw for values in receipt_dependency_inputs().values() for raw in values
    )
    roots.extend(sorted(snapshot.layer))
    candidates -= reachable_packages(roots, snapshot.selected)
    imports = set(imported_modules([SOURCE_PLUGIN_ROOT]))
    return sorted(
        name
        for name in candidates
        if not distribution_modules(snapshot.base[name]) & imports
    )


def verify_effective_environment() -> dict:
    """Verify the base and active layer after a dependency synchronization."""
    from .dependencies import DependencyAnalysisError, environment_report
    from .policy import unresolved_conflict

    report = environment_report()
    snapshot = capture_environment()
    permitted = snapshot.manifest.get("dependency_warnings", [])
    failures = [
        *report["layer_mismatch"],
        *report["immutable_drift"],
        *report["incompatible_shared_drift"],
        *[
            error
            for item in snapshot.conflicts
            if (error := unresolved_conflict(snapshot, item, permitted))
        ],
    ]
    if failures:
        raise DependencyAnalysisError(
            "current_environment_dependency_conflict", details=failures
        )
    return report


def validate_candidate(
    path: Path,
    transaction: dict[str, Any],
    *,
    snapshot: EnvironmentSnapshot | None = None,
    migrate_trials: bool = False,
) -> dict:
    """校验候选依赖层：包版本与计划一致，不引入新的依赖冲突

    参数:
        path: 候选依赖层目录
        transaction: 依赖层事务
        snapshot: 已按目标清单与候选依赖层采集的环境快照

    返回:
        dict: state 为 passed / partial，remaining_conflicts 为保留的冲突
    """
    from .dependencies import DependencyAnalysisError, protected_core

    snapshot = snapshot or capture_environment(
        manifest=transaction["target_manifest"], candidate_path=path
    )
    expected = {
        canonicalize_name(name): item["version"]
        for name, item in transaction["target_manifest"].get("packages", {}).items()
    }
    if {name: dist.version for name, dist in snapshot.layer.items()} != expected:
        raise DependencyAnalysisError("dependency_layer_state_mismatch")
    _, duplicate_layer = distributions([str(path)])
    if duplicate_layer:
        raise DependencyAnalysisError(
            "dependency_layer_state_mismatch", details=duplicate_layer
        )
    from .policy import guard_orm_changes, unresolved_conflict

    baseline = capture_environment(
        manifest=transaction.get("base_manifest") or load_manifest()
    )
    if not transaction.get("orm_packages_allowed"):
        guard_orm_changes(baseline.versions, snapshot.versions)
    permitted = transaction.get(
        "remaining_conflicts", snapshot.manifest.get("dependency_warnings", [])
    )
    remaining_core = []
    for name, version in protected_core().items():
        if snapshot.versions.get(name) != version:
            unchanged = next(
                (
                    item
                    for item in permitted
                    if item.get("tier") == "immutable_core"
                    and item["name"] == name
                    and item.get("expected") == version
                    and item.get("actual") == snapshot.versions.get(name)
                    and name not in snapshot.layer
                ),
                None,
            )
            if unchanged:
                remaining_core.append(unchanged)
                continue
            raise DependencyAnalysisError(
                "core_dependency_conflict",
                details=[
                    {
                        "name": name,
                        "actual": snapshot.versions.get(name),
                        "expected": version,
                    }
                ],
            )
    migrations = [] if migrate_trials else None
    errors = [
        error
        for item in snapshot.conflicts
        if (
            error := unresolved_conflict(
                snapshot, item, permitted, migrations=migrations
            )
        )
    ]
    if errors:
        raise DependencyAnalysisError(
            "current_environment_dependency_conflict", details=errors
        )
    for item, evidence in migrations or []:
        item["evidence"] = evidence
    return {
        "state": "partial" if snapshot.conflicts or remaining_core else "passed",
        "remaining_conflicts": [*snapshot.conflicts, *remaining_core],
    }
