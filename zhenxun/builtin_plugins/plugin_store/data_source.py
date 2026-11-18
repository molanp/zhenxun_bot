import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, ClassVar

import ujson as json

from zhenxun.builtin_plugins.plugin_store.models import StorePluginInfo
from zhenxun.configs.repo_config import aliyun_mirrored
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.cache.bounded_ttl import BoundedTTLCache
from zhenxun.services.log import logger
from zhenxun.services.network_proxy import core_network_operation
from zhenxun.services.plugin_init import PluginInitManager
from zhenxun.services.plugin_store.plugin_store_receipts import (
    StoreReceiptStore,
    store_receipt,
)
from zhenxun.services.plugin_store.plugin_store_transaction import stage_operation
from zhenxun.services.runtime_mutation import (
    coordinated_store_operation,
)
from zhenxun.utils.enum import PluginType
from zhenxun.utils.image_utils import BuildImage, ImageTemplate, RowStyle
from zhenxun.utils.repo_utils import RepoFileManager
from zhenxun.utils.repo_utils.models import RepoFileInfo, RepoType
from zhenxun.utils.repo_utils.utils import redact_git_output
from zhenxun.utils.utils import is_number, win_on_rm_error

from .config import (
    BASE_PATH,
    DEFAULT_GITHUB_URL,
    EXTRA_GITHUB_URL,
    LOG_COMMAND,
)
from .exceptions import PluginStoreException

_PLUGIN_STORE_DATA_CACHE = BoundedTTLCache[
    str, tuple[list[StorePluginInfo], list[StorePluginInfo]]
](
    "PLUGIN_STORE_DATA",
    ttl_seconds=5 * 60,
    max_items=1,
)
_PLUGIN_STORE_DATA_LOCK = asyncio.Lock()
_PLUGIN_STORE_GENERATION = 0
_PLUGIN_STORE_STALE: tuple[list[StorePluginInfo], list[StorePluginInfo]] | None = None


@dataclass(frozen=True, slots=True)
class StoreInstallResult:
    dependency_plan: dict[str, Any] = field(default_factory=dict)
    download_source: str = "auto"

    @property
    def dependency_changes(self) -> bool:
        changes = self.dependency_plan.get("package_changes", {})
        return bool(changes.get("added") or changes.get("changed"))

    @property
    def dependency_inputs(self) -> list[str]:
        """插件声明的依赖"""
        values = self.dependency_plan.get("candidate_inputs") or []
        return [str(value) for value in values if isinstance(value, str)]

    @property
    def dependency_packages(self) -> dict[str, str]:
        """需要新增或变更的依赖包版本"""
        changes = self.dependency_plan.get("package_changes") or {}
        result = {
            str(item["name"]): str(item["version"])
            for item in changes.get("added", [])
            if isinstance(item, dict) and item.get("name") and item.get("version")
        }
        result.update(
            {
                str(item["name"]): str(item["to"])
                for item in changes.get("changed", [])
                if isinstance(item, dict) and item.get("name") and item.get("to")
            }
        )
        return result


def row_style(column: str, text: str) -> RowStyle:
    """被动技能文本风格

    参数:
        column: 表头
        text: 文本内容

    返回:
        RowStyle: RowStyle
    """
    style = RowStyle()
    if column == "-" and text == "已安装":
        style.font_color = "#67C23A"
    return style


class StoreManager:
    _catalog_files: ClassVar[dict[RepoType, tuple[float, set[str] | None]]] = {}
    _refresh_tasks: ClassVar[dict[str, asyncio.Task]] = {}
    _SOURCE_NAMES: ClassVar[dict[RepoType, str]] = {
        RepoType.ALIYUN: "阿里云",
        RepoType.GITHUB: "GitHub",
    }
    _BINARY_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        ".7z .avi .bin .bmp .class .dat .db .dll .doc .docx .dylib .eot .exe .flv "
        ".gif .gz .ico .jpeg .jpg .mov .mp3 .mp4 .otf .pdf .png .ppt .pptx .pyc "
        ".rar .so .svg .tar .tif .tiff .ttf .webp .wmv .woff .woff2 .xls .xlsx .xz "
        ".zip".split()
    )

    @classmethod
    def _resolve_local_plugin_path(cls, plugin_info: StorePluginInfo) -> Path:
        """将商店插件信息映射到本地插件文件/目录路径。"""
        module_parts = plugin_info.module_path.replace("/", ".").split(".")
        plugin_name = next(
            (part for part in reversed(module_parts) if part and part.isidentifier()),
            plugin_info.module,
        )

        if plugin_info.is_dir:
            return BASE_PATH / "plugins" / plugin_name

        return BASE_PATH / "plugins" / f"{plugin_name}.py"

    @classmethod
    def runtime_module(cls, plugin_info: StorePluginInfo) -> str:
        """插件安装后的运行时模块名

        参数:
            plugin_info: 插件信息

        返回:
            str: 模块名
        """
        path = cls._resolve_local_plugin_path(plugin_info)
        return ".".join(path.with_suffix("").parts)

    @classmethod
    def _refresh_later(cls, key: str, refresh) -> None:
        """在后台刷新过期的缓存，同一项同时只刷新一次

        参数:
            key: 缓存项
            refresh: 返回刷新协程的函数
        """
        task = cls._refresh_tasks.get(key)
        if task is not None and not task.done():
            return

        async def run():
            try:
                await refresh()
            except Exception as error:
                logger.warning(
                    f"后台刷新插件商店缓存 {key} 失败: {type(error).__name__}",
                    LOG_COMMAND,
                )

        cls._refresh_tasks[key] = asyncio.create_task(
            run(), name=f"plugin-store-refresh-{key}"
        )

    @classmethod
    @core_network_operation
    async def get_data(
        cls, *, refresh: bool = False, source: str | None = None
    ) -> tuple[list[StorePluginInfo], list[StorePluginInfo]]:
        """获取插件信息数据，缓存过期后先返回上次的数据并在后台重新拉取

        参数:
            refresh: 是否忽略缓存重新拉取
            source: 拉取目录的下载源，ali / git，为空时先阿里云后 GitHub

        返回:
            tuple[list[StorePluginInfo], list[StorePluginInfo]]:
                原生插件信息数据，第三方插件信息数据
        """
        global _PLUGIN_STORE_GENERATION, _PLUGIN_STORE_STALE

        cache_key = "plugins_json"
        requested_generation = _PLUGIN_STORE_GENERATION

        def parse_entries(content: str, source: str) -> list[StorePluginInfo]:
            try:
                raw_entries = json.loads(content)
            except Exception as error:
                raise PluginStoreException(f"{source}_catalog_invalid") from error
            if not isinstance(raw_entries, list):
                raise PluginStoreException(f"{source}_catalog_invalid")
            entries: list[StorePluginInfo] = []
            for index, raw in enumerate(raw_entries):
                try:
                    entries.append(StorePluginInfo(**raw))
                except Exception as error:
                    logger.warning(
                        f"{source} 插件列表第 {index} 项无效: {type(error).__name__}",
                        LOG_COMMAND,
                    )
            return entries

        if not refresh:
            cached_data = await _PLUGIN_STORE_DATA_CACHE.get(cache_key)
            if cached_data is not None:
                return deepcopy(cached_data)
            if _PLUGIN_STORE_STALE is not None:
                cls._refresh_later(
                    cache_key, lambda: cls.get_data(refresh=True, source=source)
                )
                return deepcopy(_PLUGIN_STORE_STALE)

        async with _PLUGIN_STORE_DATA_LOCK:
            cached_data = await _PLUGIN_STORE_DATA_CACHE.get(cache_key)
            if cached_data is not None and (
                not refresh or _PLUGIN_STORE_GENERATION != requested_generation
            ):
                return deepcopy(cached_data)

            order = cls._get_source_order(source, aliyun_mirrored("plugin"))
            repo_type = order[0] if len(order) == 1 else None
            plugins, extra_plugins = await asyncio.gather(
                RepoFileManager.get_text_content(
                    DEFAULT_GITHUB_URL, "plugins.json", repo_type=repo_type
                ),
                RepoFileManager.get_text_content(
                    EXTRA_GITHUB_URL, "plugins.json", "index", repo_type=repo_type
                ),
            )
            result = (
                parse_entries(plugins, "official"),
                parse_entries(extra_plugins, "community"),
            )
            await _PLUGIN_STORE_DATA_CACHE.set(cache_key, result)
            _PLUGIN_STORE_STALE = result
            _PLUGIN_STORE_GENERATION += 1
            return deepcopy(result)

    @classmethod
    async def invalidate_cache(cls) -> None:
        global _PLUGIN_STORE_GENERATION, _PLUGIN_STORE_STALE

        async with _PLUGIN_STORE_DATA_LOCK:
            await _PLUGIN_STORE_DATA_CACHE.delete("plugins_json")
            _PLUGIN_STORE_STALE = None
            _PLUGIN_STORE_GENERATION += 1
            cls._catalog_files = {}

    @classmethod
    @core_network_operation
    async def catalog_files(
        cls, repo_type: RepoType, *, refresh: bool = False
    ) -> set[str] | None:
        """列出下载源中官方仓库的文件，成功结果缓存 5 分钟，失败结果缓存 1 分钟，
        成功结果过期后先返回上次的结果并在后台重新列出

        参数:
            repo_type: 下载源
            refresh: 是否忽略缓存重新拉取

        返回:
            set[str] | None: 文件路径集合，列出失败时为 None
        """
        now = time.monotonic()
        if not refresh and (cached := cls._catalog_files.get(repo_type)):
            checked_at, files = cached
            if now - checked_at < (5 * 60 if files is not None else 60):
                return files
            if files is not None:
                cls._refresh_later(
                    f"catalog-{repo_type.value}",
                    lambda: cls.catalog_files(repo_type, refresh=True),
                )
                return files
        try:
            listed = await RepoFileManager.list_directory_files(
                DEFAULT_GITHUB_URL, "", "main", repo_type=repo_type
            )
            files = {Path(file.path).as_posix() for file in listed if not file.is_dir}
        except Exception:
            files = None
        cls._catalog_files[repo_type] = (now, files)
        return files

    @classmethod
    async def catalog_health(
        cls,
        plugins: list[StorePluginInfo],
        *,
        refresh: bool = False,
        source: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """按下载顺序检查官方插件在下载源目录中是否存在，前面的源已包含全部插件时
        不再列出后面的源

        参数:
            plugins: 插件信息
            refresh: 是否忽略缓存重新检查
            source: 检查的下载源，ali / git，为空时先阿里云后 GitHub

        返回:
            dict[str, dict[str, Any]]: 模块名: 检查结果
        """
        result: dict[str, dict[str, Any]] = {}
        expected: dict[str, str] = {}
        for plugin in plugins:
            if plugin.github_url not in {None, DEFAULT_GITHUB_URL} or plugin.ali_url:
                result[plugin.module] = {"status": "unknown", "reason": None}
                continue
            module_path = plugin.module_path.replace(".", "/").strip("/")
            expected[plugin.module] = (
                f"{module_path}/__init__.py" if plugin.is_dir else f"{module_path}.py"
            )
        found: set[str] = set()
        failed: list[str] = []
        for repo_type in cls._get_source_order(source, aliyun_mirrored("plugin")):
            if len(found) == len(expected):
                break
            files = await cls.catalog_files(repo_type, refresh=refresh)
            if files is None:
                failed.append(repo_type.value)
                continue
            found.update(module for module, path in expected.items() if path in files)
        for module, path in expected.items():
            if module in found:
                status, reason = "available", None
            elif failed:
                status, reason = "unknown", "catalog_health_unavailable"
            else:
                status, reason = "missing", "catalog_source_missing"
            result[module] = {
                "status": status,
                "reason": reason,
                "expected_path": path,
                "failed_sources": failed,
            }
        return result

    @classmethod
    def version_check(cls, plugin_info: StorePluginInfo, suc_plugin: dict[str, str]):
        """版本检查

        参数:
            plugin_info: StorePluginInfo
            suc_plugin: 模块名: 版本号

        返回:
            str: 版本号
        """
        module = plugin_info.module
        if suc_plugin.get(module) and not cls.check_version_is_new(
            plugin_info, suc_plugin
        ):
            return f"{suc_plugin[module]} (有更新->{plugin_info.version})"
        return plugin_info.version

    @classmethod
    def check_version_is_new(
        cls, plugin_info: StorePluginInfo, suc_plugin: dict[str, str]
    ):
        """检查版本是否有更新

        参数:
            plugin_info: StorePluginInfo
            suc_plugin: 模块名: 版本号

        返回:
            bool: 是否有更新
        """
        module = plugin_info.module
        return suc_plugin.get(module) and plugin_info.version == suc_plugin[module]

    @classmethod
    async def get_installed_plugins(cls) -> dict[str, str]:
        """获取已安装插件的模块与版本。

        返回:
            dict[str, str]: 模块 -> 版本
        """
        db_plugin_list = await PluginInfo.get_plugins_values_list(
            "module", "version", load_status=True, filter_parent=False
        )
        return {p[0]: (p[1] or "0.1") for p in db_plugin_list}

    @classmethod
    async def get_plugins_info(cls) -> list[BuildImage] | str:
        """插件列表

        返回:
            BuildImage | str: 返回消息
        """
        plugin_list, extra_plugin_list = await cls.get_data()
        column_name = ["-", "ID", "名称", "简介", "作者", "版本", "类型"]
        suc_plugin = await cls.get_installed_plugins()
        index = 0
        data_list = []
        extra_data_list = []
        for plugin_info in plugin_list:
            data_list.append(
                [
                    "已安装" if plugin_info.module in suc_plugin else "",
                    index,
                    plugin_info.name,
                    plugin_info.description,
                    plugin_info.author,
                    cls.version_check(plugin_info, suc_plugin),
                    plugin_info.plugin_type_name,
                ]
            )
            index += 1
        for plugin_info in extra_plugin_list:
            extra_data_list.append(
                [
                    "已安装" if plugin_info.module in suc_plugin else "",
                    index,
                    plugin_info.name,
                    plugin_info.description,
                    plugin_info.author,
                    cls.version_check(plugin_info, suc_plugin),
                    plugin_info.plugin_type_name,
                ]
            )
            index += 1
        return [
            await ImageTemplate.table_page(
                "原生插件列表",
                "通过添加/移除插件 ID 来管理插件",
                column_name,
                data_list,
                text_style=row_style,
            ),
            await ImageTemplate.table_page(
                "第三方插件列表",
                "通过添加/移除插件 ID 来管理插件",
                column_name,
                extra_data_list,
                text_style=row_style,
            ),
        ]

    @classmethod
    async def get_plugin_by_value(
        cls,
        index_or_module: str,
        is_update: bool = False,
        is_remove: bool = False,
        source: str | None = None,
    ) -> tuple[StorePluginInfo, bool]:
        """获取插件信息

        参数:
            index_or_module: 插件索引或模块名
            is_update: 是否是更新插件
            is_remove: 是否是移除插件

        异常:
            PluginStoreException: 插件不存在
            PluginStoreException: 插件已安装

        返回:
            StorePluginInfo: 插件信息
            bool: 是否是外部插件
        """
        plugin_list: list[StorePluginInfo]
        extra_plugin_list: list[StorePluginInfo]
        plugin_list, extra_plugin_list = await cls.get_data(source=source)
        plugin_info = None
        is_external = False
        try:
            plugin_key = await cls._resolve_plugin_key(index_or_module)
        except PluginStoreException:
            if not is_remove:
                raise
            # 移除时插件可能已不在商店列表，回退到数据库查找
            plugin_key = None

        if plugin_key is not None:
            for p in plugin_list:
                if p.module == plugin_key:
                    is_external = False
                    plugin_info = p
                    break
            for p in extra_plugin_list:
                if p.module == plugin_key:
                    is_external = True
                    plugin_info = p
                    break

        installed_modules = set((await cls.get_installed_plugins()).keys())

        def is_installed(info: StorePluginInfo) -> bool:
            return (
                info.module in installed_modules
                or cls._resolve_local_plugin_path(info).exists()
            )

        if is_remove:
            # 商店列表中找不到时，从数据库构建最小插件信息
            if not plugin_info:
                db_obj = await PluginInfo.get_plugin(
                    module=index_or_module, plugin_type=PluginType.PARENT
                ) or await PluginInfo.get_plugin(module=index_or_module)
                if db_obj is None:
                    db_obj = await PluginInfo.get_or_none(name=index_or_module)
                if db_obj is None:
                    raise PluginStoreException("插件 Module / 名称 不存在...")
                _mp = db_obj.module_path
                _path = BASE_PATH.parent / Path(_mp.replace(".", os.sep))
                plugin_info = StorePluginInfo(
                    name=db_obj.name,
                    module=db_obj.module,
                    module_path=_mp,
                    description="",
                    usage="",
                    author=db_obj.author or "",
                    version=db_obj.version or "0.0.0",
                    plugin_type=db_obj.plugin_type or PluginType.NORMAL,
                    is_dir=_path.is_dir(),
                )
                is_external = True
            if not is_installed(plugin_info):
                raise PluginStoreException(f"插件 {plugin_info.name} 未安装，无法移除")
            if plugin_obj := await PluginInfo.get_plugin(
                module=plugin_info.module,
                plugin_type=PluginType.PARENT,
                load_status=True,
            ):
                plugin_info.module_path = plugin_obj.module_path
            elif plugin_obj := await PluginInfo.get_plugin(
                module=plugin_info.module, load_status=True
            ):
                plugin_info.module_path = plugin_obj.module_path
            return plugin_info, is_external

        if not plugin_info:
            raise PluginStoreException(f"插件不存在: {plugin_key}")

        if is_update:
            if not is_installed(plugin_info):
                raise PluginStoreException(f"插件 {plugin_info.name} 未安装，无法更新")
            return plugin_info, is_external

        if is_installed(plugin_info):
            raise PluginStoreException(f"插件 {plugin_info.name} 已安装，无需重复安装")

        return plugin_info, is_external

    @classmethod
    async def add_plugin(
        cls,
        index_or_module: str,
        source: str | None = None,
        *,
        confirm_source_build: bool = False,
    ) -> str:
        """添加插件

        参数:
            index_or_module: 插件id或模块名
            source: 下载源
            confirm_source_build: 是否允许从源码构建依赖

        返回:
            str: 返回消息
        """
        cls._get_source_order(source)
        plugin_info, is_external = await cls.get_plugin_by_value(index_or_module)
        logger.info(f"正在安装插件 {plugin_info.name}...", LOG_COMMAND)
        operation = await cls.apply_plugin(
            plugin_info,
            is_external,
            source,
            confirm_source_build=confirm_source_build,
        )
        return await cls._command_message("安装", plugin_info.name, operation)

    @classmethod
    async def _command_message(
        cls, action: str, plugin_name: str, operation: dict[str, Any]
    ) -> str:
        """生成指令回复，需要重启时提交重启请求

        参数:
            action: 操作名称
            plugin_name: 插件名称
            operation: 操作结果

        返回:
            str: 返回消息
        """
        message = cls.operation_info(action, plugin_name, operation)
        if operation.get("apply_mode") == "restart_pending":
            from zhenxun.utils._restart_utils import request_restart

            _, restart = await request_restart("plugin_store")
            message += f"\n- {restart}"
        return message

    @staticmethod
    def operation_info(action: str, plugin_name: str, operation: dict) -> str:
        """商店操作结果说明

        参数:
            action: 操作名称
            plugin_name: 插件名称
            operation: 操作结果

        返回:
            str: 结果说明
        """
        if operation.get("unchanged"):
            return f"插件 {plugin_name} 已是最新版本"
        mode = operation.get("apply_mode")
        if mode == "hot_reloaded":
            if action in {"热重载", "卸载"}:
                return f"插件 {plugin_name} 已{action}"
            return f"插件 {plugin_name} 已{action}并热加载"
        if mode == "restart_requested":
            return f"插件 {plugin_name} 已{action}，已请求受控重启"
        if mode == "restart_pending":
            return f"插件 {plugin_name} 已{action}，等待重启后生效"
        reason = operation.get("reason") or ""
        if reason.startswith("plugin_configuration_required:"):
            key = reason.partition(":")[2]
            return (
                f"插件 {plugin_name} {action}失败，缺少配置项 {key}，"
                "请在 系统设置 → 配置中心 → 环境配置 中填写并保存，重启生效后重试"
            )
        if operation.get("rolled_back"):
            if operation.get("rollback_runtime") in {None, "hot_reloaded"}:
                return f"插件 {plugin_name} {action}后加载失败，已回滚"
            return (
                f"插件 {plugin_name} {action}后加载失败，文件已回滚，重启后恢复原版本"
            )
        if reason.startswith("plugin_dependency_missing:"):
            missing = reason.partition(":")[2]
            return (
                f"插件 {plugin_name} {action}失败，缺少依赖 {missing}，"
                "请在插件的 requirements.txt 中声明后重启"
            )
        if stopped := operation.get("stopped"):
            return (
                f"插件 {plugin_name} {action}失败，{'、'.join(stopped)} 已停用，"
                "修复后再次热重载即可恢复"
            )
        if action == "热重载":
            if operation.get("reason") == "plugin_not_loaded":
                return f"插件 {plugin_name} 未加载，无法热重载"
            return f"插件 {plugin_name} 热重载失败，原版本仍在运行"
        return f"插件 {plugin_name} 文件已{action}，但运行时应用失败"

    @classmethod
    @coordinated_store_operation
    async def apply_plugin(
        cls,
        plugin_info: StorePluginInfo,
        is_external: bool,
        source: str | None = None,
        *,
        confirm_source_build: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """下载插件并在运行中生效

        依赖无变化时部署并热加载，加载失败时恢复原文件与原版本；依赖有变化或需要
        重启时，插件源码写入待重启事务，重启后经启动验证生效。

        参数:
            plugin_info: 插件信息
            is_external: 是否是第三方插件
            source: 下载源，ali 为阿里云，git 为 GitHub，不指定时优先阿里云
            confirm_source_build: 是否允许从源码构建依赖
            operation_id: 操作 id

        返回:
            dict[str, Any]: 操作结果
        """
        from zhenxun.services.plugin_host import ApplyMode, plugin_host

        if plugin_info.github_url is None:
            plugin_info.github_url = DEFAULT_GITHUB_URL
        if len(version_split := plugin_info.version.split("-")) > 1:
            repo_url = plugin_info.github_url.split("/tree/")[0]
            plugin_info.github_url = f"{repo_url}/tree/{version_split[1]}"
        path = cls._resolve_local_plugin_path(plugin_info)
        module = cls.runtime_module(plugin_info)
        store_key = cls.store_key(plugin_info, is_external)
        update = path.exists()
        with tempfile.TemporaryDirectory(prefix="zhenxun_plugin_store_") as temp_dir:
            root = Path(temp_dir)
            candidate, result = await cls._download_plugin(
                plugin_info, source, root, store_key, confirm_source_build
            )
            receipt = store_receipt(
                source="community" if is_external else "official",
                plugin_info=plugin_info,
                runtime_module=module,
                path=candidate,
                dependency_inputs=result.dependency_inputs,
            )

            def stage(reason: str) -> dict[str, Any]:
                return stage_operation(
                    action="update" if update else "install",
                    store_key=store_key,
                    module=plugin_info.module,
                    runtime_module=module,
                    live_path=path,
                    candidate_path=candidate,
                    receipt=receipt,
                    reason=reason,
                    dependency_inputs=result.dependency_inputs,
                    dependency_packages=result.dependency_packages,
                    source_build_confirmed=confirm_source_build,
                    operation_id=operation_id,
                )

            if result.dependency_changes:
                reason = "plugin_dependencies_changed"
                runtime = await plugin_host.request_restart(
                    {module}, reason, submit_launcher=False
                )
                operation = {**runtime.public_dict(), **stage(reason)}
                logger.info(
                    f"插件 {plugin_info.name} 及其依赖将在重启时安装", LOG_COMMAND
                )
                return {**operation, "actual_download_source": result.download_source}
            backup = root / "backup" / path.name
            if update:
                cls._copy(path, backup)
            try:
                cls._deploy(candidate, path)
                if update:
                    runtime = await plugin_host.recover_plugin(
                        module, submit_restart=False
                    )
                else:
                    runtime = await plugin_host.load_new_plugin(
                        module, path, submit_restart=False, dependencies_verified=True
                    )
            except Exception:
                cls._restore(path, backup)
                if update:
                    await plugin_host.recover_plugin(module, submit_restart=False)
                raise
            operation = runtime.public_dict()
            if runtime.mode is ApplyMode.HOT_RELOADED:
                StoreReceiptStore.set(store_key, receipt)
            elif runtime.mode is ApplyMode.FAILED:
                logger.warning(
                    f"插件 {plugin_info.name} 加载失败: {runtime.reason}，撤销本次安装",
                    LOG_COMMAND,
                )
                cls._restore(path, backup)
                operation["rolled_back"] = True
                if update:
                    recovery = await plugin_host.recover_plugin(
                        module, submit_restart=False
                    )
                    operation["rollback_runtime"] = recovery.mode.value
            else:
                cls._restore(path, backup)
                operation.update(stage(runtime.reason or "plugin_restart_required"))
        return {**operation, "actual_download_source": result.download_source}

    @staticmethod
    def _copy(source: Path, target: Path) -> None:
        if not source.exists():
            raise PluginStoreException(
                "plugin_source_missing",
                code="plugin_source_missing",
                details={"stage": "source_copy", "file": source.name},
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        except OSError as error:
            code = (
                "plugin_source_missing"
                if isinstance(error, FileNotFoundError) and not source.exists()
                else "plugin_file_access_failed"
            )
            raise PluginStoreException(
                code,
                code=code,
                details={
                    "stage": "source_copy",
                    "file": source.name,
                    "errno": error.errno,
                },
            ) from None

    @staticmethod
    def _remove_local(path: Path) -> None:
        if path.is_dir():
            # 处理 Windows 下 .git 等目录内只读文件导致的 WinError 5
            shutil.rmtree(path, onerror=win_on_rm_error)
        elif path.exists():
            path.unlink()

    @classmethod
    def _deploy(cls, candidate: Path, path: Path) -> None:
        """将候选源码复制到暂存目录后整体替换插件目录或文件

        参数:
            candidate: 候选源码
            path: 插件路径
        """
        if not candidate.exists():
            raise PluginStoreException(
                "plugin_candidate_missing",
                code="plugin_candidate_missing",
                details={"stage": "candidate_publish", "file": candidate.name},
            )
        staging = BASE_PATH.parent / "data" / "runtime" / "plugin-store-staging"
        try:
            staging.mkdir(parents=True, exist_ok=True)
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise PluginStoreException(
                "plugin_publish_failed",
                code="plugin_publish_failed",
                details={
                    "stage": "publish_directory",
                    "file": path.name,
                    "errno": error.errno,
                },
            ) from None
        with tempfile.TemporaryDirectory(prefix=f"{path.name}.", dir=staging) as root:
            new, old = Path(root) / "new", Path(root) / "old"
            cls._copy(candidate, new)
            try:
                if path.exists():
                    os.replace(path, old)
                os.replace(new, path)
            except OSError as error:
                if old.exists() and not path.exists():
                    os.replace(old, path)
                raise PluginStoreException(
                    "plugin_publish_failed",
                    code="plugin_publish_failed",
                    details={
                        "stage": "candidate_publish",
                        "file": path.name,
                        "errno": error.errno,
                    },
                ) from None
            cls._remove_local(old)

    @classmethod
    def _restore(cls, path: Path, backup: Path) -> None:
        """恢复插件原文件，没有备份时删除插件

        参数:
            path: 插件路径
            backup: 备份路径
        """
        if backup.exists():
            cls._deploy(backup, path)
        else:
            cls._remove_local(path)

    @classmethod
    async def _download_plugin(
        cls,
        plugin_info: StorePluginInfo,
        source: str | None,
        root: Path,
        store_key: str,
        confirm_source_build: bool = False,
        branch: str = "main",
    ) -> tuple[Path, StoreInstallResult]:
        """下载插件到候选目录并预检依赖

        参数:
            plugin_info: 插件信息
            source: 强制使用的源，ali 为阿里云，git 为 GitHub；
                不指定时优先阿里云，失败后回退 GitHub
            root: 临时目录
            store_key: 商店插件标识
            confirm_source_build: 是否允许从源码构建依赖
            branch: 默认分支

        返回:
            tuple[Path, StoreInstallResult]: 候选源码路径与依赖预检结果
        """
        source_order = cls._get_source_order(source, cls._aliyun_available(plugin_info))
        errors: list[str] = []
        source_errors: list[dict[str, Any]] = []
        staged_result: tuple[list[tuple[Path, Path]], list[Path]] | None = None
        selected_source: RepoType | None = None
        for repo_type in source_order:
            source_name = cls._SOURCE_NAMES[repo_type]
            try:
                staged_result = await cls._download_plugin_to_staging(
                    plugin_info, repo_type, branch, root / repo_type.value
                )
                selected_source = repo_type
                logger.info(
                    f"插件 {plugin_info.name} 使用{source_name}下载成功", LOG_COMMAND
                )
                break
            except Exception as e:
                errors.append(f"{source_name}: {redact_git_output(e)}")
                source_errors.append(
                    {
                        "source": repo_type.value,
                        "reason": getattr(e, "code", None) or "plugin_download_failed",
                        "details": getattr(e, "details", None),
                    }
                )
                if repo_type != source_order[-1]:
                    logger.warning(
                        f"插件 {plugin_info.name} 使用{source_name}下载失败，"
                        "尝试 GitHub",
                        LOG_COMMAND,
                    )
        if staged_result is None or selected_source is None:
            raise PluginStoreException(
                f"插件 {plugin_info.name} 下载失败（{'；'.join(errors)}）",
                code=source_errors[-1]["reason"],
                details={"stage": "source_download", "sources": source_errors},
            )

        deploy_files, requirement_files = staged_result
        dependency_plan: dict[str, Any] = {}
        if requirement_files:
            from zhenxun.services.nonebot_store.dependencies import (
                DependencyAnalysisError,
            )
            from zhenxun.services.plugin_store.plugin_store_transaction import (
                preflight_source_requirements,
            )

            try:
                dependency_plan = await preflight_source_requirements(
                    requirement_files,
                    store_key=store_key,
                    allow_build=confirm_source_build,
                )
            except DependencyAnalysisError as error:
                if error.code == "dependency_source_metadata_unavailable":
                    raise PluginStoreException(
                        "source_build_confirmation_required"
                    ) from error
                raise PluginStoreException(
                    cls._dependency_error(error),
                    code=error.code,
                    details={
                        "stage": "dependency_preflight",
                        "diagnostic": error.details,
                    },
                ) from None
        result = StoreInstallResult(
            dependency_plan=dependency_plan,
            download_source="ali" if selected_source == RepoType.ALIYUN else "git",
        )
        if result.dependency_changes and dependency_plan.get("source_build_required"):
            if not confirm_source_build:
                raise PluginStoreException("source_build_confirmation_required")
            from zhenxun.services.nonebot_store.dependencies import prebuild_packages

            if failed := await prebuild_packages(result.dependency_packages):
                for name, error in failed.items():
                    logger.warning(f"依赖 {name} 源码构建失败: {error}", LOG_COMMAND)
                raise PluginStoreException(
                    f"{'、'.join(failed)}: dependency_source_build_failed"
                )
        local_path = cls._resolve_local_plugin_path(plugin_info)
        candidate = root / "candidate" / local_path.name
        for staged_path, destination_path in deploy_files:
            cls._copy(staged_path, candidate / destination_path.relative_to(local_path))
        if not candidate.exists():
            raise PluginStoreException(
                "plugin_candidate_missing",
                code="plugin_candidate_missing",
                details={"stage": "candidate_prepare", "file": candidate.name},
            )
        return candidate, result

    @staticmethod
    def _dependency_error(error: Exception) -> str:
        """依赖预检错误的提示，涉及具体包时附上包名

        与本体依赖冲突时，从解析输出中取出涉及的本体依赖包名。

        参数:
            error: DependencyAnalysisError

        返回:
            str: 错误码，或“包名: 错误码”
        """
        code = getattr(error, "code", "")
        if code == "dependency_requirement_unsupported":
            return f"{str(error).partition(': ')[2]}: {code}"
        if code == "orm_dependency_console_required":
            names = [str(item.get("name")) for item in getattr(error, "details", [])]
            return f"{'、'.join(names)}: {code}"
        if code == "project_dependency_conflict":
            from packaging.utils import canonicalize_name

            from zhenxun.services.nonebot_store.dependencies import project_closure

            project = set(project_closure())
            names = []
            for raw in re.findall(
                r"([A-Za-z0-9][\w.-]*)(?:\[[^\]]*\])?\s*[<>=!~]", str(error)
            ):
                name = canonicalize_name(raw)
                if name in project and name not in names:
                    names.append(name)
            if names:
                return f"{'、'.join(names)}: {code}"
        return code

    @classmethod
    def store_key(cls, plugin_info: StorePluginInfo, is_external: bool) -> str:
        """商店插件标识

        参数:
            plugin_info: 插件信息
            is_external: 是否是第三方插件

        返回:
            str: 商店插件标识
        """
        return f"{'community' if is_external else 'official'}:{plugin_info.module}"

    @staticmethod
    def _get_source_order(
        source: str | None, aliyun: bool = True
    ) -> tuple[RepoType, ...]:
        """解析插件下载源

        参数:
            source: ali 为阿里云，git 为 GitHub，为空时先阿里云后 GitHub
            aliyun: 仓库是否有阿里云镜像，没有时只使用 GitHub

        返回:
            tuple[RepoType, ...]: 依次尝试的下载源
        """
        if source not in {None, "ali", "git"}:
            raise PluginStoreException(f"源类型错误: {source}，请使用 ali 或 git")
        if not aliyun or source == "git":
            return (RepoType.GITHUB,)
        if source == "ali":
            return (RepoType.ALIYUN,)
        return (RepoType.ALIYUN, RepoType.GITHUB)

    @staticmethod
    def _aliyun_available(plugin_info: StorePluginInfo) -> bool:
        """判断插件是否可从阿里云下载，伴生插件仓库非官方时没有阿里云镜像

        参数:
            plugin_info: 插件信息

        返回:
            bool: 是否可从阿里云下载
        """
        repo_url = (plugin_info.github_url or DEFAULT_GITHUB_URL).split("/tree/")[0]
        return (
            bool(plugin_info.ali_url)
            or repo_url != DEFAULT_GITHUB_URL.split("/tree/")[0]
            or aliyun_mirrored("plugin")
        )

    @staticmethod
    def _get_repository_url(plugin_info: StorePluginInfo, repo_type: RepoType) -> str:
        """获取指定下载源对应的仓库地址。"""
        if repo_type == RepoType.ALIYUN and plugin_info.ali_url:
            return plugin_info.ali_url
        if plugin_info.github_url:
            return plugin_info.github_url
        raise PluginStoreException(f"插件 {plugin_info.name} 缺少仓库地址")

    @staticmethod
    def _get_repository_branch(repo_url: str | None, default_branch: str) -> str:
        """优先使用仓库 URL 中显式指定的分支、标签或提交。"""
        if repo_url and "/tree/" in repo_url:
            _, _, ref = repo_url.partition("/tree/")
            if ref := ref.strip("/"):
                return ref
        return default_branch

    @classmethod
    def _get_plugin_repository_branch(
        cls,
        plugin_info: StorePluginInfo,
        repo_type: RepoType,
        default_branch: str,
    ) -> str:
        """按下载源独立解析分支，避免把 GitHub 分支套到阿里云镜像。"""
        branch_source_url = (
            plugin_info.ali_url
            if repo_type == RepoType.ALIYUN
            else plugin_info.github_url
        )
        return cls._get_repository_branch(branch_source_url, default_branch)

    @classmethod
    @core_network_operation
    async def _download_plugin_to_staging(
        cls,
        plugin_info: StorePluginInfo,
        repo_type: RepoType,
        default_branch: str,
        staging_root: Path,
    ) -> tuple[list[tuple[Path, Path]], list[Path]]:
        """从单一仓库源完整下载插件到临时目录。"""
        repo_url = cls._get_repository_url(plugin_info, repo_type)
        branch = cls._get_plugin_repository_branch(
            plugin_info,
            repo_type,
            default_branch,
        )
        module_path = plugin_info.module_path
        repository_plugin_path = module_path.replace(".", "/").strip("/")

        if plugin_info.is_dir:
            files = await RepoFileManager.list_directory_files(
                repo_url,
                repository_plugin_path,
                branch,
                repo_type=repo_type,
            )
        else:
            if not repository_plugin_path:
                raise PluginStoreException(
                    f"插件 {plugin_info.name} 的模块路径不能为空"
                )
            files = [RepoFileInfo(path=f"{repository_plugin_path}.py", is_dir=False)]

        files = [file for file in files if not file.is_dir]
        if not files:
            raise PluginStoreException(
                f"仓库中未找到插件目录: {plugin_info.module_path}"
            )

        local_plugin_path = cls._resolve_local_plugin_path(plugin_info)
        target_root = (
            local_plugin_path if plugin_info.is_dir else local_plugin_path.parent
        )
        download_files: list[tuple[str, Path]] = []
        deploy_files: list[tuple[Path, Path]] = []

        for file in files:
            source_path = Path(file.path)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise PluginStoreException(f"仓库包含不安全的文件路径: {file.path}")

            staged_path = staging_root / source_path
            if plugin_info.is_dir:
                plugin_root = (
                    Path(repository_plugin_path) if repository_plugin_path else Path()
                )
                try:
                    relative_path = source_path.relative_to(plugin_root)
                except ValueError as e:
                    raise PluginStoreException(
                        f"插件文件不在模块目录内: {file.path}"
                    ) from e
                destination_path = target_root / relative_path
            else:
                destination_path = local_plugin_path

            download_files.append((file.path, staged_path))
            deploy_files.append((staged_path, destination_path))

        required_download_files = download_files.copy()
        requirement_files = [
            staging_root / Path(file.path)
            for file in files
            if Path(file.path).name in {"requirement.txt", "requirements.txt"}
        ]
        root_requirements: list[tuple[str, Path]] = []
        if not requirement_files:
            root_requirements = [
                ("requirement.txt", staging_root / "requirement.txt"),
                ("requirements.txt", staging_root / "requirements.txt"),
            ]
            download_files.extend(root_requirements)

        result = await RepoFileManager.download_files(
            repo_url,
            download_files,
            branch,
            repo_type=repo_type,
            ignore_error=bool(root_requirements),
        )
        if not result.success:
            raise PluginStoreException(result.error_message or "未知下载错误")

        for source_path, staged_path in required_download_files:
            if not staged_path.is_file():
                raise PluginStoreException(
                    "plugin_source_missing",
                    code="plugin_source_missing",
                    details={"stage": "download_validation", "file": staged_path.name},
                )
            if (
                Path(source_path).suffix.lower() in cls._BINARY_EXTENSIONS
                and staged_path.stat().st_size == 0
            ):
                raise PluginStoreException(f"二进制文件下载为空: {source_path}")

        requirement_files = [path for path in requirement_files if path.is_file()]
        if root_requirements:
            requirement_files = [
                path for _, path in root_requirements if path.is_file()
            ]

        return deploy_files, requirement_files

    @classmethod
    async def remove_plugin(cls, index_or_module: str) -> str:
        """移除插件

        参数:
            index_or_module: 插件id或模块名

        返回:
            str: 返回消息
        """
        plugin_info, is_external = await cls.get_plugin_by_value(
            index_or_module, is_remove=True
        )
        operation = await cls.uninstall_plugin(plugin_info, is_external)
        return await cls._command_message("卸载", plugin_info.name, operation)

    @classmethod
    @coordinated_store_operation
    async def uninstall_plugin(
        cls,
        plugin_info: StorePluginInfo,
        is_external: bool,
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """卸载并删除插件及只属于它的配置组，无法热卸载的插件写入待重启事务，重启后删除

        参数:
            plugin_info: 插件信息
            is_external: 是否是第三方插件
            operation_id: 操作 id

        返回:
            dict[str, Any]: 操作结果
        """
        from nonebot import get_loaded_plugins

        from zhenxun.configs.config import Config
        from zhenxun.services.plugin_host import ApplyMode, plugin_host

        path = cls._resolve_local_plugin_path(plugin_info)
        if not path.exists():
            raise PluginStoreException(f"插件 {plugin_info.name} 不存在...")
        module = cls.runtime_module(plugin_info)
        store_key = cls.store_key(plugin_info, is_external)
        classification = plugin_host.classification_for(module)
        reasons = classification.get("reload_reasons") or []
        loaded = "not_loaded" not in reasons
        if loaded and classification.get("reload_support") != "hot_reloadable":
            if any(
                registered == module or registered.startswith(f"{module}.")
                for registered, model in PluginInitManager.plugins.items()
                if model.remove
            ):
                raise PluginStoreException("legacy_lifecycle_not_transactional")
            reason = str((reasons or ["plugin_not_hot_reloadable"])[0])
            operation = stage_operation(
                action="uninstall",
                store_key=store_key,
                module=plugin_info.module,
                runtime_module=module,
                live_path=path,
                candidate_path=None,
                receipt=None,
                reason=reason,
                operation_id=operation_id,
            )
            return {
                **operation,
                "status": "pending_restart",
                "changed": [plugin_info.module],
                "reason": reason,
                "generation": plugin_host.generation,
            }
        pending_reason = (
            plugin_host.last_operation.reason
            if not loaded and plugin_host.last_operation is not None
            else None
        )
        operation = {
            "apply_mode": "hot_reloaded",
            "status": "completed",
            "changed": [],
            "reason": None,
            "generation": plugin_host.generation,
        }
        config_groups = Config.plugin_groups(
            {plugin_info.module, module.rsplit(".", 1)[-1]}
            | {
                plugin.name
                for plugin in get_loaded_plugins()
                if plugin.module_name == module
                or plugin.module_name.startswith(f"{module}.")
            }
        )
        if loaded:
            runtime = await plugin_host.unload_plugin(module)
            operation = runtime.public_dict()
            if runtime.mode is ApplyMode.FAILED:
                operation["rolled_back"] = True
                return operation
        logger.debug(f"尝试移除插件 {plugin_info.name} 文件: {path}", LOG_COMMAND)
        with tempfile.TemporaryDirectory(prefix="zhenxun_plugin_remove_") as root:
            backup = Path(root) / path.name
            cls._copy(path, backup)
            try:
                cls._remove_local(path)
            except Exception:
                cls._restore(path, backup)
                if loaded:
                    await plugin_host.recover_plugin(module, submit_restart=False)
                raise
        if pending_reason:
            plugin_host.clear_pending_restart(pending_reason)
        Config.remove_groups(config_groups)
        StoreReceiptStore.delete(store_key)
        await PluginInitManager.remove(plugin_info.module_path)
        for plugin_record in await PluginInfo.get_plugins(
            load_status=None,
            filter_parent=False,
            module_path=plugin_info.module_path,
        ):
            await plugin_record.delete()
        return operation

    @classmethod
    async def search_plugin(cls, plugin_name_or_author: str) -> BuildImage | str:
        """搜索插件

        参数:
            plugin_name_or_author: 插件名称或作者

        返回:
            BuildImage | str: 返回消息
        """
        plugin_list, extra_plugin_list = await cls.get_data()
        all_plugin_list = plugin_list + extra_plugin_list
        suc_plugin = await cls.get_installed_plugins()
        filtered_data = [
            (id, plugin_info)
            for id, plugin_info in enumerate(all_plugin_list)
            if plugin_name_or_author.lower() in plugin_info.name.lower()
            or plugin_name_or_author.lower() in plugin_info.author.lower()
        ]

        data_list = [
            [
                "已安装" if plugin_info.module in suc_plugin else "",
                id,
                plugin_info.name,
                plugin_info.description,
                plugin_info.author,
                cls.version_check(plugin_info, suc_plugin),
                plugin_info.plugin_type_name,
            ]
            for id, plugin_info in filtered_data
        ]
        if not data_list:
            return "未找到相关插件..."
        column_name = ["-", "ID", "名称", "简介", "作者", "版本", "类型"]
        return await ImageTemplate.table_page(
            "商店插件列表",
            "通过添加/移除插件 ID 来管理插件",
            column_name,
            data_list,
            text_style=row_style,
        )

    @classmethod
    async def is_latest(cls, plugin_info: StorePluginInfo) -> bool:
        """插件是否已是商店中的版本

        参数:
            plugin_info: 插件信息

        返回:
            bool: 是否已是最新版本
        """
        installed = await cls.get_installed_plugins()
        return bool(cls.check_version_is_new(plugin_info, installed))

    @classmethod
    async def update_plugin(
        cls,
        index_or_module: str,
        source: str | None = None,
        *,
        confirm_source_build: bool = False,
    ) -> str:
        """更新插件

        参数:
            index_or_module: 插件id或模块名
            source: 下载源
            confirm_source_build: 是否允许从源码构建依赖

        返回:
            str: 返回消息
        """
        cls._get_source_order(source)
        plugin_info, is_external = await cls.get_plugin_by_value(index_or_module, True)
        logger.info(f"尝试更新插件 {plugin_info.name}", LOG_COMMAND)
        if await cls.is_latest(plugin_info):
            return f"插件 {plugin_info.name} 已是最新版本"
        operation = await cls.apply_plugin(
            plugin_info,
            is_external,
            source,
            confirm_source_build=confirm_source_build,
        )
        return await cls._command_message("更新", plugin_info.name, operation)

    @classmethod
    async def update_all_plugin(
        cls, source: str | None = None, *, confirm_source_build: bool = False
    ) -> str:
        """更新全部已安装插件并在运行中生效，需要重启的插件在重启后一并生效

        参数:
            source: 下载源
            confirm_source_build: 是否允许从源码构建依赖

        返回:
            str: 返回消息
        """
        cls._get_source_order(source)
        plugin_list, extra_plugin_list = await cls.get_data()
        all_plugin_list = plugin_list + extra_plugin_list
        plugin_name_list = [p.name for p in all_plugin_list]
        update_failed_list = []
        update_success_list = []
        restart_pending = False
        result = "--已更新{}个插件 {}个失败 {}个成功--"
        logger.info(f"尝试更新全部插件 {plugin_name_list}", LOG_COMMAND)
        suc_plugin = await cls.get_installed_plugins()
        external_modules = {plugin.module for plugin in extra_plugin_list}
        for plugin_info in all_plugin_list:
            try:
                if plugin_info.module not in suc_plugin:
                    logger.debug(
                        f"插件 {plugin_info.name}({plugin_info.module}) 未安装，跳过",
                        LOG_COMMAND,
                    )
                    continue
                if cls.check_version_is_new(plugin_info, suc_plugin):
                    logger.debug(
                        f"插件 {plugin_info.name}({plugin_info.module}) "
                        "已是最新版本，跳过",
                        LOG_COMMAND,
                    )
                    continue
                logger.info(
                    f"正在更新插件 {plugin_info.name}({plugin_info.module})",
                    LOG_COMMAND,
                )
                operation = await cls.apply_plugin(
                    plugin_info,
                    plugin_info.module in external_modules,
                    source,
                    confirm_source_build=confirm_source_build,
                )
            except Exception as e:
                logger.error(
                    f"更新插件 {plugin_info.name}({plugin_info.module}) 失败",
                    LOG_COMMAND,
                    e=e,
                )
                update_failed_list.append(plugin_info.name)
                continue
            if operation.get("apply_mode") == "failed":
                update_failed_list.append(plugin_info.name)
                continue
            restart_pending |= operation.get("apply_mode") == "restart_pending"
            update_success_list.append(plugin_info.name)
        if not update_success_list and not update_failed_list:
            return "全部插件已是最新版本"
        if update_success_list:
            result += "\n* 以下插件更新成功:\n\t- {}".format(
                "\n\t- ".join(update_success_list)
            )
        if update_failed_list:
            result += "\n* 以下插件更新失败:\n\t- {}".format(
                "\n\t- ".join(update_failed_list)
            )
        result = result.format(
            len(update_success_list) + len(update_failed_list),
            len(update_failed_list),
            len(update_success_list),
        )
        if restart_pending:
            from zhenxun.utils._restart_utils import request_restart

            _, message = await request_restart("plugin_store")
            result += f"\n- 部分插件需要重启后生效\n- {message}"
        return result

    @classmethod
    async def _resolve_plugin_key(cls, plugin_id: str) -> str:
        """获取插件module

        参数:
            plugin_id: module，id或插件名称

        异常:
            PluginStoreException: 插件不存在
            PluginStoreException: 插件不存在

        返回:
            str: 插件模块名
        """
        plugin_list, extra_plugin_list = await cls.get_data()
        all_plugin_list = plugin_list + extra_plugin_list
        if is_number(plugin_id):
            idx = int(plugin_id)
            if idx < 0 or idx >= len(all_plugin_list):
                raise PluginStoreException("插件ID不存在...")
            return all_plugin_list[idx].module
        elif isinstance(plugin_id, str):
            if plugin_id in [v.module for v in all_plugin_list]:
                return plugin_id

            for plugin_info in all_plugin_list:
                if plugin_info.name.lower() == plugin_id.lower():
                    return plugin_info.module

            raise PluginStoreException("插件 Module / 名称 不存在...")
