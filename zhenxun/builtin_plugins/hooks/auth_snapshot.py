from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

from zhenxun.services.cache.cache_containers import CacheDict
from zhenxun.services.cache.keyed import KeyedLocks
from zhenxun.services.cache.runtime_cache import (
    BotMemoryCache,
    GroupMemoryCache,
    PluginInfoMemoryCache,
    PluginLimitMemoryCache,
)
from zhenxun.services.cache.runtime_snapshot import BotSnapshot, GroupSnapshot
from zhenxun.services.cache.runtime_sync import (
    RuntimeCacheMutation,
    RuntimeMemoryCache,
)
from zhenxun.services.db_context import with_db_timeout
from zhenxun.services.log import logger
from zhenxun.services.message_load import is_db_unhealthy
from zhenxun.services.platform_identity import is_namespaced_scope
from zhenxun.utils.enum import BlockType, PluginType
from zhenxun.utils.platform import PlatformUtils

from .auth.config import LOGGER_COMMAND
from .auth.context import EventContext

if TYPE_CHECKING:
    from nonebot.adapters import Bot
    from nonebot_plugin_uninfo import Uninfo

    from zhenxun.models.plugin_info import PluginInfo


async def require_cache(cache_cls: type[RuntimeMemoryCache]) -> None:
    """确保运行态缓存已加载且无待对账的发布失败

    参数:
        cache_cls: 运行态缓存类

    异常:
        CacheUnavailable: 缓存不可用，跳过本条消息
    """
    if not cache_cls.is_loaded():
        await cache_cls.ensure_loaded()
    RuntimeCacheMutation.require_fresh(cache_cls)


async def get_plugin(module: str) -> PluginInfo | None:
    """从运行态缓存读取插件信息

    参数:
        module: 插件模块

    返回:
        PluginInfo | None: 插件信息
    """
    await require_cache(PluginInfoMemoryCache)
    return await PluginInfoMemoryCache.get_by_module(module)


@dataclass(frozen=True, slots=True)
class PluginAuthProfile:
    module: str
    name: str
    hidden: bool = False
    status: bool = True
    block_type: BlockType | None = None
    plugin_type: PluginType | None = None
    need_admin: bool = False
    need_group_check: bool = False
    has_limit: bool = False
    cost_gold: int = 0
    admin_level: int = 0
    limit_superuser: bool = False
    level: int = 0

    @property
    def superuser_only(self) -> bool:
        return self.plugin_type == PluginType.SUPERUSER


def _int_attr(plugin, name: str) -> int:
    try:
        return int(getattr(plugin, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def build_plugin_auth_profile(plugin, *, has_limit: bool = False) -> PluginAuthProfile:
    plugin_type = getattr(plugin, "plugin_type", None)
    admin_level = _int_attr(plugin, "admin_level")
    block_type = getattr(plugin, "block_type", None)
    module = str(getattr(plugin, "module", "") or "")
    need_admin = bool(admin_level > 0) or plugin_type in {
        PluginType.ADMIN,
        PluginType.SUPERUSER,
        PluginType.SUPER_AND_ADMIN,
    }
    return PluginAuthProfile(
        module=module,
        name=str(getattr(plugin, "name", "") or module),
        hidden=plugin_type == PluginType.HIDDEN,
        status=bool(getattr(plugin, "status", True)),
        block_type=block_type,
        plugin_type=plugin_type,
        need_admin=need_admin,
        need_group_check=block_type
        in {BlockType.ALL, BlockType.GROUP, BlockType.PRIVATE},
        has_limit=bool(has_limit),
        cost_gold=_int_attr(plugin, "cost_gold"),
        admin_level=admin_level,
        limit_superuser=bool(getattr(plugin, "limit_superuser", False)),
        level=int(getattr(plugin, "level", 0) or 0),
    )


async def get_plugin_auth_profile(plugin) -> PluginAuthProfile:
    """读取插件限制并构建权限配置

    参数:
        plugin: 插件信息

    返回:
        PluginAuthProfile: 插件权限配置
    """
    await require_cache(PluginLimitMemoryCache)
    limits = await PluginLimitMemoryCache.get_limits(
        str(getattr(plugin, "module", "") or "")
    )
    return build_plugin_auth_profile(plugin, has_limit=bool(limits))


GROUP_REPAIR_TTL = 60
_GROUP_REPAIR_FAILURES = CacheDict[float](
    "AUTH_GROUP_REPAIR_FAILURES", expire=GROUP_REPAIR_TTL, max_items=4096
)
_GROUP_REPAIR_LOCKS = KeyedLocks(capacity=4096)


def _build_runtime_group_snapshot(context: EventContext) -> GroupSnapshot | None:
    """QQ 官方适配器使用不落库的默认群配置

    参数:
        context: EventContext

    返回:
        GroupSnapshot | None: QQ 官方群事件的默认群配置，其余为 None
    """
    if context.platform_scope != "qq_api" or not context.group_id:
        return None
    return GroupSnapshot(
        group_id=context.group_id,
        channel_id=context.channel_id,
        group_name="",
        max_member_count=0,
        member_count=0,
        status=True,
        level=5,
        is_super=False,
        group_flag=0,
        platform=context.platform,
    )


def _group_repair_key(context: EventContext) -> tuple[str, str] | None:
    if not context.group_id:
        return None
    if context.platform_scope == "qq_client":
        return (context.group_id, context.channel_id or "")
    if is_namespaced_scope(context.platform_scope) and context.storage_group_id:
        return (context.storage_group_id, "")
    return None


def _group_repair_on_cooldown(key: tuple[str, str]) -> bool:
    expire_at = _GROUP_REPAIR_FAILURES.get(key)
    if not expire_at:
        return False
    if expire_at <= time.time():
        _GROUP_REPAIR_FAILURES.pop(key, None)
        return False
    return True


async def _repair_missing_group(
    context: EventContext,
) -> GroupSnapshot | None:
    """为缺少群记录的 OneBot 群或前缀作用域群补建最小群记录

    参数:
        context: EventContext

    返回:
        GroupSnapshot | None: 补建后的群配置，失败或不支持时为 None
    """
    if is_db_unhealthy():
        return None
    key = _group_repair_key(context)
    if key is None or not GroupMemoryCache.is_loaded():
        return None
    group_id, channel_id = key
    if _group_repair_on_cooldown(key):
        return None

    try:
        from zhenxun.models.group_console import GroupConsole

        async with _GROUP_REPAIR_LOCKS.hold(key):
            if _group_repair_on_cooldown(key):
                return None
            existing = GroupMemoryCache.get_if_ready(group_id, channel_id or None)
            if existing is not None:
                return existing
            defaults = {
                "group_name": "",
                "max_member_count": 0,
                "member_count": 0,
                "group_flag": 0,
                "platform": context.platform,
            }
            group, _ = await with_db_timeout(
                GroupConsole.get_or_create_root_group(
                    group_id=group_id,
                    defaults=defaults,
                ),
                timeout=2.0,
                operation="GroupConsole.get_or_create_root_group",
                source="auth_snapshot.repair_missing_group",
            )
        await GroupMemoryCache.upsert_from_model(group)
        return GroupSnapshot.from_model(group)
    except Exception as exc:
        _GROUP_REPAIR_FAILURES[key] = time.time() + GROUP_REPAIR_TTL
        logger.warning(
            "协议端群记录缺失自愈失败，已短期跳过重复修复",
            LOGGER_COMMAND,
            group_id=context.group_id,
            e=exc,
        )
        return None


@dataclass(slots=True)
class AuthSnapshot:
    context: EventContext
    plugin: object
    profile: PluginAuthProfile
    bot_data: BotSnapshot | None = None
    group: GroupSnapshot | None = None
    is_group_admin: bool = False

    @property
    def module(self) -> str:
        return self.profile.module

    @property
    def is_superuser(self) -> bool:
        return self.context.is_superuser

    @property
    def user_id(self) -> str:
        return self.context.user_id

    @property
    def group_id(self) -> str | None:
        return self.context.group_id

    @property
    def channel_id(self) -> str | None:
        return self.context.channel_id


async def build_auth_snapshot(
    *,
    context: EventContext,
    plugin: object,
    profile: PluginAuthProfile,
    bot: Bot,
    session: Uninfo,
) -> AuthSnapshot:
    """从运行态缓存读取 Bot、群组，并判断群管理员身份，构建权限快照

    参数:
        context: EventContext
        plugin: 插件信息
        profile: 插件权限配置
        bot: Bot
        session: Uninfo

    异常:
        CacheUnavailable: 缓存不可用，跳过本条消息

    返回:
        AuthSnapshot: 权限快照
    """
    entity = context.entity
    await require_cache(BotMemoryCache)
    bot_data = await BotMemoryCache.get(context.storage_bot_id or str(bot.self_id))

    group = _build_runtime_group_snapshot(context)
    if group is None and entity.group_id:
        await require_cache(GroupMemoryCache)
        if is_namespaced_scope(context.platform_scope):
            group = await GroupMemoryCache.get(context.storage_group_id)
        else:
            group = await GroupMemoryCache.get(entity.group_id, entity.channel_id)
        if group is None:
            group = await _repair_missing_group(context)

    is_group_admin = (
        profile.admin_level > 0
        and not context.is_superuser
        and bool(entity.group_id)
        and await PlatformUtils.is_group_admin(bot, session, entity.user_id)
    )

    return AuthSnapshot(
        context=context,
        plugin=plugin,
        profile=profile,
        bot_data=bot_data,
        group=group,
        is_group_admin=is_group_admin,
    )


__all__ = [
    "AuthSnapshot",
    "PluginAuthProfile",
    "build_auth_snapshot",
    "build_plugin_auth_profile",
    "get_plugin",
    "get_plugin_auth_profile",
    "require_cache",
]
