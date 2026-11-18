from __future__ import annotations

import asyncio
from collections import deque
import contextlib
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import quote

from dotenv import dotenv_values
import httpx
from packaging.markers import InvalidMarker, Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from zhenxun.utils.pythonpath import INSTANCE_ROOT

from .storage import load_manifest

SOLVER_POLICY_VERSION = 13
LOCK_FILE = INSTANCE_ROOT / "uv.lock"
_REQ_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)")
_SENSITIVE = re.compile(r"(?i)(authorization|token|password|secret)=?[^\s]*")
PROJECT_FILE = INSTANCE_ROOT / "pyproject.toml"
PROJECT_NAME = "zhenxun-bot"
IMMUTABLE_PACKAGES = frozenset(
    {
        "nonebot2",
        "nonebot-adapter-onebot",
        "nonebot-adapter-qq",
        "nonebot-plugin-alconna",
        "arclet-alconna",
        "arclet-alconna-tools",
        "nepattern",
        "tarina",
        "nonebot-plugin-apscheduler",
        "nonebot-plugin-session",
        "nonebot-plugin-uninfo",
        "nonebot-plugin-waiter",
        "pydantic",
        "pydantic-core",
        "pydantic-settings",
        "fastapi",
        "starlette",
        "uvicorn",
        "tortoise-orm",
        "pypika-tortoise",
        "aiosqlite",
        "asyncpg",
        "aiomysql",
        "pymysql",
        "redis",
        "aiocache",
    }
)
# 本体代码在 pyproject 声明的整个版本范围内都兼容的包，依赖层可在范围内改用更低版本
RANGE_PACKAGES = frozenset(
    {"aiofiles", "blitz-py", "cryptography", "numpy", "pillow", "ujson"}
)
FORBIDDEN_LAYER_PACKAGES = frozenset({PROJECT_NAME, "pip", "uv"})


class DependencyAnalysisError(RuntimeError):
    def __init__(self, code: str, message: str | None = None, *, details: Any = None):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.details = details


def safe_process_error(value: str) -> str:
    value = re.sub(r"https?://[^\s<>]+", "<package-source>", value)
    value = _SENSITIVE.sub(r"\1=<redacted>", value)
    return " ".join(value.split())[-1200:]


async def _dependency_process(
    command: list[str], *, stage: str, env: dict[str, str], cwd: Path | None = None
) -> asyncio.subprocess.Process:
    """启动依赖子进程并提供不含命令参数或环境凭据的阶段诊断。"""
    working_directory = cwd
    if working_directory is None:
        with contextlib.suppress(OSError):
            working_directory = Path.cwd()
    try:
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as error:
        if working_directory is None or not working_directory.is_dir():
            code = "dependency_working_directory_missing"
        elif isinstance(error, FileNotFoundError):
            code = "dependency_tool_unavailable"
        else:
            code = "dependency_process_start_failed"
        raise DependencyAnalysisError(
            code,
            details={
                "stage": stage,
                "executable": safe_process_error(command[0]),
                "working_directory": safe_process_error(
                    str(working_directory)
                    if working_directory is not None
                    else "<current>"
                ),
                "errno": error.errno,
                "filename": safe_process_error(str(error.filename or "")),
            },
        ) from None


_inventory_cache: tuple[tuple, dict[str, str]] | None = None


def _inventory_key() -> tuple:
    from .environment import generation_path

    generation = load_manifest().get("active_generation")
    paths = list(sys.path)
    if isinstance(generation, int):
        paths.append(str(generation_path(generation)))
    key: list[tuple[str, int]] = [("generation", int(generation or -1))]
    for value in paths:
        try:
            key.append((value, os.stat(value or ".").st_mtime_ns))
        except OSError:
            continue
    return tuple(key)


def installed_inventory() -> dict[str, str]:
    """已安装包的版本，搜索路径与依赖层目录都没有变化时复用上次的结果

    返回:
        dict[str, str]: 包名 -> 版本
    """
    global _inventory_cache
    from .environment import capture_environment

    key = _inventory_key()
    if _inventory_cache is not None and _inventory_cache[0] == key:
        return dict(_inventory_cache[1])
    versions = capture_environment().versions
    _inventory_cache = (key, versions)
    return dict(versions)


def _marker_enabled(marker: Any) -> bool:
    text = str(marker or "").strip()
    if not text:
        return True
    try:
        return Marker(text).evaluate(default_environment())
    except InvalidMarker as error:
        raise DependencyAnalysisError(
            "dependency_marker_invalid", text[:200]
        ) from error


def _lock_package_enabled(package: dict[str, Any]) -> bool:
    markers = package.get("resolution-markers") or []
    if not markers:
        return True
    return any(_marker_enabled(marker) for marker in markers)


_lock_cache: tuple[tuple[str, int, int], Any] | None = None


def _lock_packages() -> tuple[dict[str, set[str]], dict[str, str], set[str], set[str]]:
    """解析 uv.lock 中项目可达的依赖图，按文件修改时间和大小缓存

    返回:
        tuple: 依赖图、锁定版本、项目直接依赖、开发依赖组可达的包
    """
    global _lock_cache
    if not LOCK_FILE.is_file():
        raise DependencyAnalysisError("project_lock_missing")
    stat = LOCK_FILE.stat()
    key = (str(LOCK_FILE.resolve()), stat.st_mtime_ns, stat.st_size)
    if _lock_cache is None or _lock_cache[0] != key:
        _lock_cache = (key, _parse_lock_packages())
    graph, versions, roots, dev = _lock_cache[1]
    return (
        {name: set(items) for name, items in graph.items()},
        dict(versions),
        set(roots),
        set(dev),
    )


def _parse_lock_packages() -> (
    tuple[dict[str, set[str]], dict[str, str], set[str], set[str]]
):
    data = tomllib.loads(LOCK_FILE.read_text(encoding="utf-8"))
    packages: dict[str, dict[str, Any]] = {}
    for package in data.get("package", []):
        if (
            not isinstance(package, dict)
            or not package.get("name")
            or not _lock_package_enabled(package)
        ):
            continue
        name = canonicalize_name(str(package["name"]))
        if name in packages and (
            packages[name].get("version") != package.get("version")
            or packages[name].get("source") != package.get("source")
        ):
            raise DependencyAnalysisError("project_lock_branch_ambiguous", name)
        packages[name] = package

    versions: dict[str, str] = {
        name: str(package["version"])
        for name, package in packages.items()
        if package.get("version")
    }

    def walk(declared: list) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}
        selected_extras: dict[str, set[str]] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, frozenset[str]]] = deque([(PROJECT_NAME, frozenset())])
        while queue:
            name, extras = queue.popleft()
            previous_extras = selected_extras.setdefault(name, set())
            new_extras = set(extras) - previous_extras
            first_visit = name not in visited
            if not first_visit and not new_extras:
                continue
            visited.add(name)
            previous_extras.update(extras)
            package = packages.get(name, {})
            dependencies = []
            if first_visit:
                dependencies = (
                    list(declared)
                    if name == PROJECT_NAME
                    else list(package.get("dependencies", []))
                )
            optional = package.get("optional-dependencies", {})
            for extra in extras if first_visit else new_extras:
                dependencies.extend(optional.get(extra, []))
            for item in dependencies:
                if (
                    not isinstance(item, dict)
                    or not item.get("name")
                    or not _lock_dependency_enabled(item)
                ):
                    continue
                dependency = canonicalize_name(str(item["name"]))
                selected = packages.get(dependency)
                if (
                    selected is None
                    or (
                        item.get("version")
                        and item["version"] != selected.get("version")
                    )
                    or (item.get("source") and item["source"] != selected.get("source"))
                ):
                    raise DependencyAnalysisError(
                        "project_lock_branch_ambiguous", dependency
                    )
                graph.setdefault(name, set()).add(dependency)
                queue.append(
                    (
                        dependency,
                        frozenset(str(extra) for extra in item.get("extra", [])),
                    )
                )
        return graph

    project = packages.get(PROJECT_NAME, {})
    graph = walk(list(project.get("dependencies", [])))
    groups = project.get("dev-dependencies") or {}
    dev_graph = walk([item for values in groups.values() for item in values])
    dev = {name for items in dev_graph.values() for name in items}
    roots = set(graph.get(PROJECT_NAME, set()))
    return graph, versions, roots, dev


def _lock_dependency_enabled(item: dict[str, Any]) -> bool:
    return _marker_enabled(item.get("marker"))


def project_closure() -> dict[str, str]:
    graph, versions, roots, _ = _lock_packages()
    selected: set[str] = set()
    queue = deque(roots)
    while queue:
        name = queue.popleft()
        if name in selected:
            continue
        selected.add(name)
        queue.extend(graph.get(name, set()) - selected)
    return {name: versions[name] for name in selected if versions.get(name)}


def dev_packages() -> set[str]:
    """uv.lock 中只被开发依赖组需要的包，依赖同步不安装它们

    返回:
        set[str]: 规范化包名
    """
    *_, dev = _lock_packages()
    return dev - set(project_closure())


def protected_core() -> dict[str, str]:
    """核心不可变包的锁定版本

    只包含本体依赖中属于 IMMUTABLE_PACKAGES 的包，依赖层不能改变它们的版本。

    返回:
        dict[str, str]: 包名 -> uv.lock 中的版本
    """
    closure = project_closure()
    return {
        name: version
        for name, version in closure.items()
        if name in IMMUTABLE_PACKAGES or name in FORBIDDEN_LAYER_PACKAGES
    }


def shared_dependencies() -> dict[str, str]:
    """本体依赖中可变包的锁定版本

    这些包可以随插件需要在依赖层中改变版本，解析时的约束见 dependency_floors。

    返回:
        dict[str, str]: 包名 -> uv.lock 中的版本
    """
    closure = project_closure()
    immutable = protected_core()
    return {name: version for name, version in closure.items() if name not in immutable}


def dependency_floors() -> dict[str, str]:
    """解析依赖层时本体可变依赖的版本约束

    一般不低于锁定版本；RANGE_PACKAGES 中的包限定在 pyproject 声明的范围内，插件
    需要时可以改用范围内更低的版本。

    返回:
        dict[str, str]: 包名 -> 版本约束，如 ">=1.2.0"
    """
    requirements = _project_requirement_map()
    floors = {}
    for name, version in shared_dependencies().items():
        requirement = requirements.get(name)
        if name in RANGE_PACKAGES and requirement and requirement.specifier:
            floors[name] = str(requirement.specifier)
        else:
            floors[name] = f">={version}"
    return floors


def project_requirements() -> list[str]:
    if not PROJECT_FILE.is_file():
        raise DependencyAnalysisError("project_metadata_missing")
    data = tomllib.loads(PROJECT_FILE.read_text(encoding="utf-8"))
    values = data.get("project", {}).get("dependencies", [])
    result: list[str] = []
    for raw in values:
        try:
            requirement = Requirement(str(raw))
        except InvalidRequirement as error:
            raise DependencyAnalysisError(
                "project_dependency_invalid", str(raw)[:160]
            ) from error
        if requirement.marker and not requirement.marker.evaluate(
            default_environment()
        ):
            continue
        result.append(str(requirement))
    return result


def environment_drift() -> list[dict[str, str | None]]:
    expected = protected_core()
    current = installed_inventory()
    return [
        {
            "name": name,
            "expected": version,
            "actual": current.get(name),
        }
        for name, version in sorted(expected.items())
        if current.get(name) != version
    ]


def _version_relation(actual: str | None, expected: str) -> str:
    if actual is None:
        return "missing"
    try:
        actual_version = Version(actual)
        expected_version = Version(expected)
    except InvalidVersion:
        return "version_mismatch"
    if actual_version > expected_version:
        return "newer"
    if actual_version < expected_version:
        return "older"
    return "matched"


def _project_requirement_map() -> dict[str, Requirement]:
    result: dict[str, Requirement] = {}
    for raw in project_requirements():
        requirement = Requirement(raw)
        result[canonicalize_name(requirement.name)] = requirement
    return result


def _project_venv_active() -> bool:
    expected = os.environ.get("UV_PROJECT_ENVIRONMENT") or os.environ.get("VIRTUAL_ENV")
    try:
        actual = Path(sys.prefix).resolve()
        requested = (
            Path(expected).resolve() if expected else (Path.cwd() / ".venv").resolve()
        )
        return actual == requested and actual != Path(sys.base_prefix).resolve()
    except OSError:
        return False


def environment_report(*, check_lock: bool = False, snapshot=None) -> dict[str, Any]:
    immutable = protected_core()
    shared = shared_dependencies()
    from .environment import capture_environment, core_environment_summary

    snapshot = snapshot or capture_environment()
    base = {name: dist.version for name, dist in snapshot.base.items()}
    layer = {name: dist.version for name, dist in snapshot.layer.items()}
    effective = snapshot.versions
    manifest = snapshot.manifest
    recorded = {
        canonicalize_name(name): str(info.get("version", ""))
        for name, info in manifest.get("packages", {}).items()
        if isinstance(info, dict)
    }
    layer_mismatch = [
        {
            "name": name,
            "recorded": recorded.get(name),
            "actual": layer.get(name),
            "layer": "active_generation",
            "generation": manifest.get("active_generation"),
        }
        for name in sorted(set(recorded) | set(layer))
        if recorded.get(name) != layer.get(name)
    ]
    requirements = _project_requirement_map()
    requirement_conflicts = snapshot.conflicts
    immutable_drift: list[dict[str, Any]] = []
    incompatible_shared_drift: list[dict[str, Any]] = []

    for name, expected in sorted(immutable.items()):
        actual = effective.get(name)
        if actual == expected:
            continue
        immutable_drift.append(
            {
                "name": name,
                "tier": "immutable_core",
                "expected": expected,
                "actual": actual,
                "layer": "active_generation" if name in layer else "base",
                "requirement": f"{name}=={expected}",
                "kind": _version_relation(actual, expected),
            }
        )

    for name, expected in sorted(shared.items()):
        actual = effective.get(name)
        if actual == expected:
            continue
        requirement = requirements.get(name)
        compatible = bool(actual)
        if compatible and requirement and requirement.specifier:
            try:
                compatible = requirement.specifier.contains(
                    Version(str(actual)), prereleases=True
                )
            except InvalidVersion:
                compatible = False
        if compatible:
            continue
        incompatible_shared_drift.append(
            {
                "name": name,
                "tier": "shared_compatible",
                "expected": expected,
                "actual": actual,
                "kind": _version_relation(actual, expected),
                "requirement": str(requirement) if requirement else None,
                "layer": "active_generation" if name in layer else "base",
            }
        )

    interpreter_mismatch = not _project_venv_active()
    lock_stale = False
    if check_lock:
        from zhenxun.services.installer_network import (
            project_environment,
            uv_executable,
        )

        uv = uv_executable()
        if uv is None:
            lock_stale = True
        else:
            try:
                checked = subprocess.run(
                    [uv, "lock", "--check"],
                    cwd=Path.cwd(),
                    capture_output=True,
                    timeout=30,
                    check=False,
                    env=project_environment(),
                )
            except (OSError, subprocess.TimeoutExpired):
                lock_stale = True
            else:
                lock_stale = checked.returncode != 0
    lock_digest = (
        sha256(LOCK_FILE.read_bytes()).hexdigest() if LOCK_FILE.exists() else "missing"
    )
    payload = {
        "immutable_drift": immutable_drift,
        "layer_mismatch": layer_mismatch,
        "requirement_conflicts": requirement_conflicts,
        "incompatible_shared_drift": incompatible_shared_drift,
        "interpreter_mismatch": interpreter_mismatch,
        "project_lock_stale": lock_stale,
        "repairable": bool(
            not interpreter_mismatch
            and not lock_stale
            and (
                layer_mismatch
                or immutable_drift
                or incompatible_shared_drift
                or requirement_conflicts
            )
            and LOCK_FILE.is_file()
        ),
        "repair_mode": "base" if immutable_drift and not layer_mismatch else "layer",
    }
    payload["fingerprint"] = sha256(
        json.dumps(
            {
                "lock": lock_digest,
                "project": sha256(PROJECT_FILE.read_bytes()).hexdigest(),
                "versions": sorted(snapshot.versions.items()),
                "policy": SOLVER_POLICY_VERSION,
                "base": sorted(base.items()),
                "layer": sorted(layer.items()),
                "manifest": {
                    key: manifest.get(key)
                    for key in (
                        "active_generation",
                        "packages",
                        "plugins",
                        "manual_requirements",
                        "plugin_requirements",
                        "generation_digest",
                        "pending_verification",
                    )
                },
                "python": platform.python_version(),
                "platform": sys.platform,
                "lock_stale": lock_stale,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload["core_summary"] = core_environment_summary(snapshot, immutable, payload)
    return payload


async def environment_report_async(
    *, check_lock: bool = False, snapshot=None
) -> dict[str, Any]:
    """在线程中生成依赖环境报告，避免扫描元数据和运行 uv 时阻塞事件循环

    参数:
        check_lock: 是否检查 uv.lock 是否过期
        snapshot: 已采集的环境快照

    返回:
        dict[str, Any]: 依赖环境报告
    """
    return await asyncio.to_thread(
        environment_report, check_lock=check_lock, snapshot=snapshot
    )


async def preflight_environment_repair() -> tuple[bool, str]:
    guard_orm_sync()
    from zhenxun.services.installer_network import project_environment, uv_command

    process = await _dependency_process(
        uv_command(
            "sync",
            "--locked",
            "--inexact",
            "--no-install-project",
            "--no-dev",
            "--dry-run",
        ),
        stage="environment_preflight",
        cwd=INSTANCE_ROOT,
        env=project_environment(),
    )
    stdout, stderr = await process.communicate()
    detail = safe_process_error((stderr or stdout).decode(errors="replace"))
    return process.returncode == 0, detail


def guard_orm_sync() -> None:
    """Reject lock synchronization that would change any ORM distribution."""
    from .environment import base_distributions
    from .policy import guard_orm_changes

    current = {name: dist.version for name, dist in base_distributions()[0].items()}
    guard_orm_changes(current, {**current, **project_closure()})


def environment_fingerprint(
    registry_plugin: dict[str, Any], *, pending_revision: str = ""
) -> str:
    lock_digest = (
        sha256(LOCK_FILE.read_bytes()).hexdigest() if LOCK_FILE.exists() else "missing"
    )
    report = environment_report()
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "lock": lock_digest,
        "environment": report["fingerprint"],
        "project": registry_plugin.get("project_link"),
        "version": registry_plugin.get("version"),
        "solver_policy": SOLVER_POLICY_VERSION,
        "pending_revision": pending_revision,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def fetch_pypi_metadata(project: str, version: str) -> dict[str, Any]:
    # 延迟导入：nonebot_store 要保证"被 import 时不拉起服务层"。
    # 模块级 import 会经 runtime.py 把 zhenxun.services 带进 sys.modules，
    # 破坏 launcher 的导入隔离（见 tests/test_launcher_import_safety.py）。
    from zhenxun.services.network_proxy import ManagedAsyncClient

    safe_project = quote(project, safe="-._")
    safe_version = quote(version, safe="-._+")
    url = f"https://pypi.org/pypi/{safe_project}/{safe_version}/json"
    async with ManagedAsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(url, headers={"Accept": "application/json"})
        if response.status_code == 404:
            raise DependencyAnalysisError("registry_version_not_on_pypi")
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise DependencyAnalysisError("pypi_metadata_unavailable") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise DependencyAnalysisError("pypi_metadata_invalid")
    return payload


def metadata_compatibility(metadata: dict[str, Any]) -> list[dict[str, str]]:
    info = metadata["info"]
    reasons: list[dict[str, str]] = []
    requires_python = str(info.get("requires_python") or "").strip()
    if requires_python:
        try:
            if not SpecifierSet(requires_python).contains(
                Version(platform.python_version()), prereleases=True
            ):
                reasons.append(
                    {"code": "python_version_incompatible", "message": requires_python}
                )
        except (InvalidVersion, ValueError):
            reasons.append(
                {"code": "python_requirement_invalid", "message": requires_python}
            )

    installed = installed_inventory()
    environment = default_environment()
    for raw in info.get("requires_dist") or []:
        try:
            requirement = Requirement(str(raw))
        except InvalidRequirement:
            reasons.append(
                {"code": "dependency_metadata_invalid", "message": str(raw)[:160]}
            )
            continue
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        if name == "nonebot":
            reasons.append({"code": "nonebot1_plugin", "message": str(requirement)})
            continue
        if name not in {"nonebot2", "pydantic"} or not requirement.specifier:
            continue
        current = installed.get(name)
        if current and not requirement.specifier.contains(current, prereleases=True):
            reasons.append(
                {
                    "code": f"{name}_version_incompatible",
                    "message": f"{requirement}; current={current}",
                }
            )
    return reasons


def _write_constraints(
    path: Path, values: dict[str, str], floors: dict[str, str] | None = None
) -> None:
    """写入 uv 约束文件

    参数:
        path: 约束文件
        values: 固定版本的包
        floors: 包名 -> 版本约束，已固定版本的包忽略
    """
    lines = [f"{name}=={version}\n" for name, version in sorted(values.items())]
    lines.extend(
        f"{name}{specifier}\n"
        for name, specifier in sorted((floors or {}).items())
        if name not in values
    )
    path.write_text("".join(lines), encoding="utf-8")


def _parse_compiled(
    path: Path, graph: dict[str, set[str]] | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _REQ_LINE.match(line.strip())
        if match:
            name = canonicalize_name(match.group(1))
            result[name] = match.group(2)
            if graph is not None and "# via " in line:
                graph[name] = {
                    canonicalize_name(item.strip())
                    for item in line.split("# via ", 1)[1].split(",")
                    if item.strip() and not item.strip().startswith("-")
                }
    if not result:
        raise DependencyAnalysisError("dependency_plan_empty")
    return result


async def _compile(
    requirements: list[str],
    constraints: dict[str, str],
    *,
    wheels_only: bool,
    restricted_packages: tuple[str, ...] = (),
    preferences: dict[str, str] | None = None,
    no_dependencies: bool = False,
    allow_build: bool = False,
    target: tuple[str, str] | None = None,
    graph: dict[str, set[str]] | None = None,
    floors: dict[str, str] | None = None,
    _metadata_links: tuple[Path, ...] = (),
) -> tuple[dict[str, str] | None, str]:
    from zhenxun.services.installer_network import (
        UV_BINARY_POLICY_VERSION,
        installer_environment,
        resolver_configuration_error,
        uv_binary_options,
        uv_compile_command,
        uv_version,
    )

    with tempfile.TemporaryDirectory(prefix="zhenxun_nb_analyze_") as root:
        directory = Path(root)
        source = directory / "requirements.in"
        constraint_file = directory / "constraints.txt"
        output = directory / "requirements.txt"
        source.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        _write_constraints(constraint_file, constraints, floors)
        if preferences:
            # uv reuses existing output pins as preferences, not hard constraints.
            _write_constraints(output, preferences)
        command = uv_compile_command(source, output, constraint_file)
        if wheels_only:
            command.extend(uv_binary_options(wheels_only=True))
        elif not allow_build:
            command.append("--no-build")
        if no_dependencies:
            command.append("--no-deps")
        if target is not None:
            command.extend(
                ["--python-platform", target[0], "--python-version", target[1]]
            )
        if graph is not None:
            command.extend(["--annotate", "--annotation-style", "line"])
        environment = installer_environment()
        if _metadata_links:
            # Command-line --find-links replaces UV_FIND_LINKS, so carry the
            # configured local sources along with the projected metadata.
            configured = re.split(r"[,\s]+", environment.get("UV_FIND_LINKS", ""))
            for link in [*filter(None, configured), *map(str, _metadata_links)]:
                command.extend(["--find-links", link])
            command.append("--no-cache")
        process = await _dependency_process(
            command,
            stage="dependency_resolve",
            cwd=INSTANCE_ROOT,
            env=environment,
        )
        import psutil

        from zhenxun.utils.process_tree import kill_tree, verified_descendants

        try:
            owned = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            owned = None
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 180)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            tree = [owned] if owned is not None else []
            if owned is not None:
                with contextlib.suppress(psutil.NoSuchProcess):
                    tree += verified_descendants(owned)
            alive = await asyncio.to_thread(kill_tree, tree)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await asyncio.wait_for(process.wait(), 10)
            if alive:
                raise DependencyAnalysisError("dependency_tool_cleanup_unconfirmed")
            raise
        details = safe_process_error((stderr or stdout).decode(errors="replace"))
        if resolver_configuration_error(details):
            raise DependencyAnalysisError(
                "dependency_resolver_configuration",
                details,
                details={
                    "uv_version": uv_version(),
                    "binary_policy_version": UV_BINARY_POLICY_VERSION,
                    "wheels_only": wheels_only,
                    "diagnostic": details,
                },
            )
        if process.returncode != 0 or not output.exists():
            _check_resolution_failure(details)
            missing = re.search(r"Wheels are required for `([A-Za-z0-9_.-]+)`", details)
            if (
                not wheels_only
                and not allow_build
                and missing
                and len(_metadata_links) < 32
            ):
                name = canonicalize_name(missing.group(1))
                projected = any(
                    list(link.glob(name.replace("-", "_") + "-*.whl"))
                    for link in _metadata_links
                )
                if name not in restricted_packages and not projected:
                    from .source_metadata import project_static_sources

                    projection = directory / "source-metadata"
                    specifiers = [
                        str(requirement.specifier)
                        for raw in requirements
                        if canonicalize_name((requirement := Requirement(raw)).name)
                        == name
                    ]
                    if name in constraints:
                        specifiers.append("==" + constraints[name])
                    elif floors and name in floors:
                        specifiers.append(floors[name])
                    if await project_static_sources(
                        name, projection, installer_environment(), specifiers
                    ):
                        return await _compile(
                            requirements,
                            constraints,
                            wheels_only=wheels_only,
                            restricted_packages=restricted_packages,
                            preferences=preferences,
                            no_dependencies=no_dependencies,
                            target=target,
                            graph=graph,
                            floors=floors,
                            _metadata_links=(*_metadata_links, projection),
                        )
            return None, details
        return _parse_compiled(output, graph), details


async def dependency_graph(
    requirements: list[str],
    preferences: dict[str, str],
    target: tuple[str, str] | None = None,
) -> dict[str, set[str]]:
    """解析依赖声明，得到每个包被哪些包直接依赖

    参数:
        requirements: 依赖声明
        preferences: 优先选用的版本
        target: 解析的目标平台（uv --python-platform 的取值）与 Python 版本，
            为 None 时按当前解释器解析

    返回:
        dict[str, set[str]]: 包名 -> 直接依赖它的包名，只由依赖声明引入的包为空集
    """
    graph: dict[str, set[str]] = {}
    resolved, details = await _compile(
        requirements,
        {},
        wheels_only=False,
        preferences=preferences,
        target=target,
        graph=graph,
    )
    if resolved is None:
        raise DependencyAnalysisError("dependency_resolution_failed", details)
    return graph


async def prebuild_packages(packages: dict[str, str]) -> dict[str, str]:
    """把包安装到临时目录，提前构建源码包并留在 uv 缓存中

    整体安装失败时逐个安装，找出失败的包。

    参数:
        packages: 包名 -> 版本

    返回:
        dict[str, str]: 安装失败的包 -> 错误说明
    """
    import psutil

    from zhenxun.services.installer_network import installer_environment, uv_command
    from zhenxun.utils.process_tree import kill_tree, verified_descendants

    async def install(names: list[str]) -> str | None:
        with tempfile.TemporaryDirectory(prefix="zhenxun_prebuild_") as root:
            requirements = Path(root) / "requirements.txt"
            requirements.write_text(
                "".join(f"{name}=={packages[name]}\n" for name in names),
                encoding="utf-8",
            )
            process = await _dependency_process(
                uv_command(
                    "pip",
                    "install",
                    "--target",
                    str(Path(root) / "target"),
                    "--requirements",
                    str(requirements),
                    "--no-deps",
                    "--python",
                    sys.executable,
                ),
                stage="dependency_prebuild",
                env=installer_environment(),
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), 1800)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                with contextlib.suppress(psutil.NoSuchProcess):
                    owned = psutil.Process(process.pid)
                    await asyncio.to_thread(
                        kill_tree, [owned, *verified_descendants(owned)]
                    )
                raise
            if process.returncode == 0:
                return None
            return safe_process_error(stderr.decode(errors="replace"))

    if not packages or await install(list(packages)) is None:
        return {}
    failed = {}
    for name in packages:
        error = await install([name])
        if error:
            failed[name] = error
    return failed


def _check_resolution_failure(details: str) -> None:
    from zhenxun.services.installer_network import resolver_failure

    failure = resolver_failure(details)
    if failure["code"] not in {
        "dependency_conflict",
        "dependency_source_required",
    }:
        raise DependencyAnalysisError(failure["code"], details)


def _active_metadata_requirements(metadata: dict[str, Any]) -> list[Requirement]:
    result: list[Requirement] = []
    for raw in metadata.get("info", {}).get("requires_dist") or []:
        try:
            requirement = Requirement(str(raw))
        except InvalidRequirement as error:
            raise DependencyAnalysisError(
                "dependency_metadata_invalid", str(raw)[:160]
            ) from error
        if requirement.marker and not requirement.marker.evaluate(
            default_environment()
        ):
            continue
        result.append(requirement)
    return result


def _wheel_availability(metadata: dict[str, Any]) -> dict[str, Any]:
    supported_tags = set(sys_tags())
    wheels: list[str] = []
    compatible: list[str] = []
    for item in metadata.get("urls") or []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        if not filename.endswith(".whl"):
            continue
        wheels.append(filename)
        try:
            _, _, _, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if supported_tags.intersection(tags):
            compatible.append(filename)
    return {
        "compatible": bool(compatible),
        "compatible_wheel_files": compatible,
    }


def _compatibility_overrides(
    metadata: dict[str, Any],
    *,
    immutable: dict[str, str],
    shared: dict[str, str],
    current: dict[str, str],
) -> tuple[list[dict[str, str]], list[str]]:
    from .policy import ORM_PACKAGES, guard_orm_request

    overrides, inputs = [], []
    for requirement in _active_metadata_requirements(metadata):
        name = canonicalize_name(requirement.name)
        version = current.get(name)
        if (
            not version
            or requirement.url
            or requirement.specifier.contains(version, prereleases=True)
        ):
            inputs.append(str(requirement))
            continue
        if name in ORM_PACKAGES:
            guard_orm_request({name}, current)
        if name in immutable:
            raise DependencyAnalysisError(
                "core_dependency_conflict",
                details=[
                    {
                        "name": name,
                        "tier": "immutable_core",
                        "kind": "version_mismatch",
                        "declared_requirement": str(requirement),
                        "expected": immutable[name],
                        "actual": version,
                    }
                ],
            )
        overrides.append(
            {
                "name": name,
                "declared_requirement": str(requirement),
                "effective_version": version,
                "tier": "shared_compatible" if name in shared else "plugin_private",
                "risk": "unverified_runtime_compatibility",
            }
        )
        extras = (
            "[" + ",".join(sorted(requirement.extras)) + "]"
            if requirement.extras
            else ""
        )
        inputs.append(f"{name}{extras}=={version}")
    return overrides, inputs


async def _diagnose_failure(
    requests: list[str], detail: str, *, allow_build: bool, invalid_code: str
) -> DependencyAnalysisError:
    """依赖层解析失败时判断原因

    参数:
        requests: 本次请求的依赖声明
        detail: 解析失败的输出
        allow_build: 是否允许构建源码包读取依赖信息
        invalid_code: 请求自身无法解析时的错误码

    返回:
        DependencyAnalysisError: 与其他依赖冲突、与本体依赖冲突或请求自身无效
    """
    core, core_error = await _compile(
        [*project_requirements(), *requests],
        protected_core(),
        wheels_only=False,
        allow_build=allow_build,
        floors=dependency_floors(),
    )
    if core is not None:
        return DependencyAnalysisError("third_party_dependency_conflict", detail)
    free, free_error = await _compile(
        requests, {}, wheels_only=False, allow_build=allow_build
    )
    if free is not None:
        return DependencyAnalysisError(
            "project_dependency_conflict", core_error or detail
        )
    return DependencyAnalysisError(invalid_code, free_error or core_error or detail)


async def resolve_layer(
    requests: list[str],
    *,
    snapshot,
    manifest: dict[str, Any],
    sources: dict[str, list[str]] | None = None,
    replaced: frozenset[str] = frozenset(),
    allow_build: bool = False,
    restore: bool = False,
    preferences: dict[str, str] | None = None,
    invalid_code: str = "dependency_resolution_failed",
) -> dict[str, Any]:
    """解析本次请求后的依赖层，所有安装入口共用

    从请求的包出发解析，新解析到的包连同其他来源（本体、NoneBot 插件、真寻商店
    插件、手动依赖、已安装包）对它们的声明一起加入下一轮，直到不再出现新包。
    核心不可变包与已安装的 ORM 包固定版本，其余本体依赖按 dependency_floors 约束，
    当前版本优先保留（RANGE_PACKAGES 优先使用本体版本）；请求之外的包在必要时也会
    变化，作为连带变化列出。
    解析失败且放开 ORM 固定后会改变已安装的 ORM 包时，提示需在控制台操作。

    参数:
        requests: 本次请求的依赖声明
        snapshot: 当前环境快照
        manifest: 目标清单，其中的依赖来源与依赖层包参与解析
        sources: 真寻商店插件的依赖声明，键为 store_key
        replaced: 声明不再参与解析的来源（被更新或替换的插件）
        allow_build: 是否允许构建源码包读取依赖信息
        restore: 是否为恢复备份，允许改变 ORM 包
        preferences: 当前环境中没有的包优先选用的版本
        invalid_code: 请求自身无法解析时的错误码

    返回:
        dict[str, Any]: resolved 为解析出的包版本，packages 为目标依赖层的包，
            changes 为相对当前环境的版本变化，scope 为首轮解析到的包，
            source_build_required 为变化的包是否需要源码构建，diagnostic 为说明
    """
    from .environment import dependency_roots
    from .policy import ORM_PACKAGES, external_requirements, guard_orm_changes

    def active(values: list[str]) -> list[str]:
        result = []
        for raw in values:
            marker = Requirement(raw).marker
            if (not marker or marker.evaluate()) and raw not in result:
                result.append(raw)
        return result

    current = snapshot.versions
    base = {name: dist.version for name, dist in snapshot.base.items()}
    core = protected_core()
    floors = dependency_floors()
    packages = deepcopy(manifest.get("packages", {}))
    frozen = {
        **({} if restore else {n: current[n] for n in ORM_PACKAGES if n in current}),
        **core,
    }
    preferred = {
        **(preferences or {}),
        **current,
        **{name: str(info["version"]) for name, info in packages.items()},
        **{name: base[name] for name in RANGE_PACKAGES if name in base},
    }
    roots = [
        (owner, raw)
        for owner, raw in dependency_roots(manifest, sources)
        if owner not in replaced
    ]
    requests = active(requests)
    scope = {canonicalize_name(Requirement(raw).name) for raw in requests}
    first: set[str] | None = None
    resolved: dict[str, str] = {}
    for _ in range(6):
        requirements = active(
            [
                *requests,
                *(
                    raw
                    for _, raw in roots
                    if canonicalize_name(Requirement(raw).name) in scope
                ),
                *external_requirements(snapshot, scope, replaced),
            ]
        )
        result, detail = await _compile(
            requirements,
            frozen,
            wheels_only=False,
            allow_build=allow_build,
            preferences=preferred,
            floors=floors,
        )
        if result is None:
            if set(frozen) - set(core):
                relaxed, _ = await _compile(
                    requirements,
                    core,
                    wheels_only=False,
                    allow_build=allow_build,
                    preferences=preferred,
                    floors=floors,
                )
                if relaxed is not None:
                    guard_orm_changes(current, {**current, **relaxed})
            raise await _diagnose_failure(
                requests, detail, allow_build=allow_build, invalid_code=invalid_code
            )
        resolved = result
        first = set(resolved) | scope if first is None else first
        if set(resolved) <= scope:
            break
        scope |= set(resolved)
    core_changes = [
        {"name": name, "tier": "immutable_core", "expected": core[name], "to": version}
        for name, version in resolved.items()
        if name in core and core[name] != version
    ]
    if core_changes:
        raise DependencyAnalysisError("core_dependency_conflict", details=core_changes)
    if not restore:
        guard_orm_changes(current, {**current, **resolved})
    for name, version in resolved.items():
        if name in core or name in FORBIDDEN_LAYER_PACKAGES:
            continue
        if base.get(name) == version:
            packages.pop(name, None)
        else:
            packages[name] = {**packages.get(name, {}), "version": version}
    effective = {
        **base,
        **{name: str(info["version"]) for name, info in packages.items()},
    }
    changes = [
        {"name": name, "from": current.get(name), "to": effective.get(name)}
        for name in sorted(set(current) | set(effective))
        if current.get(name) != effective.get(name)
    ]
    installed = {name: dist.version for name, dist in snapshot.layer.items()}
    changed = [
        f"{name}=={info['version']}"
        for name, info in packages.items()
        if installed.get(name) != str(info["version"])
    ]
    wheels, diagnostic = (
        await _compile(changed, {}, wheels_only=True, no_dependencies=True)
        if changed
        else ({}, "")
    )
    return {
        "resolved": resolved,
        "packages": packages,
        "changes": changes,
        "scope": sorted(first or scope),
        "source_build_required": wheels is None,
        "diagnostic": diagnostic if wheels is None else "",
    }


async def solve_install(
    plugin: dict[str, Any],
    metadata: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    prefer_existing: bool = True,
    allow_build: bool = False,
) -> dict[str, Any]:
    """解析安装 NoneBot 插件后的依赖层

    参数:
        plugin: 商店中的插件信息
        metadata: 插件在 PyPI 上的元数据
        manifest: 作为基线的依赖层清单，为空时使用当前清单
        prefer_existing: 插件声明不包含已安装版本时是否优先保留已安装版本
        allow_build: 是否允许解析时构建源码包读取依赖信息

    返回:
        dict[str, Any]: 安装计划
    """
    from .environment import capture_environment
    from .policy import guard_orm_request, retained_conflicts

    snapshot = capture_environment()
    report = environment_report(snapshot=snapshot)
    if report.get("layer_mismatch"):
        raise DependencyAnalysisError(
            "dependency_layer_state_mismatch", details=report["layer_mismatch"]
        )
    blocking_drift = [
        *(
            {
                **item,
                "tier": "immutable_core",
                "kind": _version_relation(item.get("actual"), item["expected"]),
            }
            for item in report["immutable_drift"]
        ),
        *report["incompatible_shared_drift"],
    ]
    if report["interpreter_mismatch"]:
        raise DependencyAnalysisError(
            "interpreter_mismatch", details={"command": "uv run zx"}
        )
    if blocking_drift:
        raise DependencyAnalysisError(
            "current_environment_dependency_conflict",
            ", ".join(
                f"{item['name']}={item['actual'] or 'missing'}"
                f" (lock={item['expected']})"
                for item in blocking_drift[:8]
            ),
            details=blocking_drift,
        )
    manifest = manifest or load_manifest()
    candidate_name = canonicalize_name(str(plugin["project_link"]))
    current = snapshot.versions
    guard_orm_request({candidate_name}, current)
    overrides, candidate_dependencies = _compatibility_overrides(
        metadata,
        immutable=protected_core(),
        shared=shared_dependencies(),
        current=current,
    )
    if not prefer_existing:
        overrides = []
    candidate_inputs = (
        candidate_dependencies
        if overrides
        else [f"{plugin['project_link']}=={plugin['version']}"]
    )
    result = await resolve_layer(
        candidate_inputs,
        snapshot=snapshot,
        manifest=manifest,
        replaced=frozenset({candidate_name}),
        allow_build=allow_build,
        invalid_code="plugin_dependency_invalid",
    )
    packages = result["packages"]
    if overrides:
        packages[candidate_name] = {"version": str(plugin["version"])}
    layer_packages = {name: str(info["version"]) for name, info in packages.items()}
    shared = shared_dependencies()
    changes = list(result["changes"])
    if overrides and current.get(candidate_name) != str(plugin["version"]):
        changes.append(
            {
                "name": candidate_name,
                "from": current.get(candidate_name),
                "to": str(plugin["version"]),
            }
        )
    package_changes = {
        "added": [
            {"name": item["name"], "version": item["to"]}
            for item in changes
            if item["to"] and not item["from"]
        ],
        "changed": [item for item in changes if item["to"] and item["from"]],
        "removed": [],
    }
    shared_changes = [item for item in changes if item["name"] in shared]
    scope = set(result["scope"]) | {candidate_name}
    active_packages = load_manifest().get("packages", {})
    pending = {
        name
        for name, info in (manifest.get("packages") or {}).items()
        if active_packages.get(name) != info
    }
    change_sources = {
        item["name"]: {
            "kind": (
                "requested_plugin"
                if item["name"] == candidate_name
                else "candidate_dependency"
                if item["name"] in scope
                else "pending_transaction"
                if item["name"] in pending
                else "linked_change"
            ),
            "required_by": candidate_name if item["name"] in scope else None,
        }
        for item in changes
    }
    wheel_status = _wheel_availability(metadata)
    pure_root_wheel = any(
        filename.endswith(("-py3-none-any.whl", "-py2.py3-none-any.whl"))
        for filename in wheel_status["compatible_wheel_files"]
    )
    source_required = result["source_build_required"] or not wheel_status["compatible"]
    candidate_requirement_names = {
        canonicalize_name(requirement.name)
        for requirement in _active_metadata_requirements(metadata)
    }
    database_migration_possible = (
        candidate_name == "nonebot-plugin-orm"
        or "nonebot-plugin-orm" in candidate_requirement_names
    )
    return {
        "requirements": candidate_inputs,
        "resolved_packages": layer_packages,
        "package_changes": package_changes,
        "change_sources": change_sources,
        "candidate_packages": sorted(scope),
        "shared_changes": shared_changes,
        "compatibility_overrides": overrides,
        "remaining_conflicts": retained_conflicts(
            snapshot, {candidate_name, *(item["name"] for item in changes)}
        ),
        "non_core_changes": [*shared_changes, *package_changes["changed"]],
        "source_build_required": source_required,
        "pure_python_candidate": (
            pure_root_wheel and not source_required and not overrides
        ),
        "solver_policy_version": SOLVER_POLICY_VERSION,
        "candidate_inputs": candidate_inputs,
        "database_migration_possible": database_migration_possible,
        "database_type": (
            orm_database_type() if database_migration_possible else "none"
        ),
    }


def orm_database_type() -> str:
    """读取 nonebot-plugin-orm 使用的数据库类型

    返回:
        str: 未配置或 SQLite 为 sqlite，其余为 external
    """
    env_file = Path(".env.dev") if Path(".env.dev").exists() else Path(".env")
    raw_url = str(dotenv_values(env_file).get("SQLALCHEMY_DATABASE_URL") or "")
    if not raw_url or raw_url.casefold().startswith("sqlite"):
        return "sqlite"
    return "external"


def uninstall_plan(
    plugin: dict[str, Any], *, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """生成卸载 NoneBot 插件的依赖计划

    只移除插件包本身，插件的依赖保留在依赖层中；插件包仍被其他来源声明时同样保留。

    参数:
        plugin: 已安装的插件信息
        manifest: 作为基线的依赖层清单，为空时使用当前清单

    返回:
        dict[str, Any]: 卸载计划
    """
    from .environment import capture_environment
    from .policy import guard_orm_request, retained_conflicts

    manifest = manifest or load_manifest()
    packages = {
        canonicalize_name(name): str(info["version"])
        for name, info in manifest.get("packages", {}).items()
        if isinstance(info, dict) and info.get("version")
    }
    project = canonicalize_name(str(plugin["project_link"]))
    snapshot = capture_environment()
    guard_orm_request({project}, snapshot.versions)
    owners = sorted(
        {
            item["owner"]
            for item in snapshot.declarations
            if item["name"] == project and item["owner"] != project
        }
    )
    if not owners:
        packages.pop(project, None)
    return {
        "requirements": [],
        "resolved_packages": packages,
        "package_changes": {
            "added": [],
            "changed": [],
            "removed": (
                []
                if owners
                else [
                    {
                        "name": project,
                        "version": str(plugin["version"]),
                        "reason": "plugin",
                    }
                ]
            ),
        },
        "shared_changes": [],
        "compatibility_overrides": [],
        "retained": (
            [
                {
                    "name": project,
                    "owners": owners,
                    "version": snapshot.versions.get(project),
                }
            ]
            if owners
            else []
        ),
        "remaining_conflicts": retained_conflicts(
            snapshot, set() if owners else {project}
        ),
        "non_core_changes": [],
        "source_build_required": False,
        "pure_python_candidate": True,
        "solver_policy_version": SOLVER_POLICY_VERSION,
        "candidate_inputs": [],
        "database_migration_possible": False,
        "database_type": "none",
    }


def read_requirement_files(files: list[Path]) -> list[str]:
    """读取插件 requirements 文件中的依赖声明

    只接受包依赖声明，安装器参数（如 -r、--index-url）和直链依赖会被拒绝。

    参数:
        files: requirements 文件列表

    返回:
        list[str]: 依赖声明
    """
    return [
        str(parse_requirement(line))
        for path in files
        for line in requirement_lines(path)
    ]


def requirement_lines(path: Path) -> list[str]:
    """读取 requirements 文件中去掉注释与空行后的各行

    参数:
        path: requirements 文件

    返回:
        list[str]: 依赖声明行
    """
    lines = []
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        code = (
            "dependency_source_missing"
            if isinstance(error, FileNotFoundError)
            else "dependency_source_unreadable"
        )
        raise DependencyAnalysisError(
            code,
            details={
                "stage": "dependency_requirements",
                "file": safe_process_error(str(path)),
                "errno": error.errno,
            },
        ) from None
    for raw_line in content.splitlines():
        line = re.sub(r"\s+#.*$", "", raw_line).strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def parse_requirement(value: str) -> Requirement:
    """解析单条依赖声明，拒绝安装器参数、直链依赖和 NoneBot 1

    参数:
        value: 依赖声明

    返回:
        Requirement: 包名已规范化的依赖

    异常:
        DependencyAnalysisError: 依赖声明无效或不受支持
    """
    line = str(value).strip()
    if line.startswith("-"):
        raise DependencyAnalysisError(
            "dependency_requirement_option_forbidden", line[:160]
        )
    try:
        requirement = Requirement(line)
    except InvalidRequirement as error:
        raise DependencyAnalysisError(
            "dependency_requirement_invalid", line[:160]
        ) from error
    requirement.name = canonicalize_name(requirement.name)
    if requirement.url:
        raise DependencyAnalysisError(
            "dependency_requirement_direct_url", requirement.name
        )
    if requirement.name == "nonebot":
        raise DependencyAnalysisError("nonebot1_plugin")
    return requirement
