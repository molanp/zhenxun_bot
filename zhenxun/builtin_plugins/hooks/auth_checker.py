import asyncio

from nonebot.adapters import Bot, Event
from nonebot.exception import IgnoredException
from nonebot.matcher import Matcher
from nonebot_plugin_alconna import UniMsg
from nonebot_plugin_uninfo import Uninfo
from tortoise.exceptions import IntegrityError

from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.user_console import UserConsole
from zhenxun.services.cache import Cache
from zhenxun.services.log import logger
from zhenxun.utils.enum import (
    CacheType,
    GoldHandle,
    PluginType,
)
from zhenxun.utils.exception import InsufficientGold
from zhenxun.utils.platform import PlatformUtils
from zhenxun.utils.utils import get_entity_ids

from .auth.auth_admin import auth_admin
from .auth.auth_ban import auth_ban
from .auth.auth_bot import auth_bot
from .auth.auth_cost import auth_cost
from .auth.auth_group import auth_group
from .auth.auth_limit import LimitManager, auth_limit
from .auth.auth_plugin import auth_plugin
from .auth.bot_filter import bot_filter
from .auth.config import LOGGER_COMMAND
from .auth.exception import (
    IsSuperuserException,
    PermissionExemption,
    SkipPluginException,
)


async def get_plugin_and_user(
    module: str, user_id: str
) -> tuple[PluginInfo, UserConsole]:
    """获取用户数据和插件信息

    参数:
        module: 模块名
        user_id: 用户id

    异常:
        PermissionExemption: 插件数据不存在
        PermissionExemption: 插件类型为HIDDEN
        PermissionExemption: 重复创建用户
        PermissionExemption: 用户数据不存在

    返回:
        tuple[PluginInfo, UserConsole]: 插件信息，用户信息
    """
    user_cache = Cache[UserConsole](CacheType.USERS)
    plugin = await Cache[PluginInfo](CacheType.PLUGINS).get(module)
    if not plugin:
        raise PermissionExemption(f"插件:{module} 数据不存在，已跳过权限检查...")
    if plugin.plugin_type == PluginType.HIDDEN:
        raise PermissionExemption(
            f"插件: {plugin.name}:{plugin.module} 为HIDDEN，已跳过权限检查..."
        )
    user = None
    try:
        user = await user_cache.get(user_id)
    except IntegrityError as e:
        raise PermissionExemption("重复创建用户，已跳过该次权限检查...") from e
    if not user:
        raise PermissionExemption("用户数据不存在，已跳过权限检查...")
    return plugin, user


async def get_plugin_cost(
    bot: Bot, user: UserConsole, plugin: PluginInfo, session: Uninfo
) -> int:
    """获取插件费用

    参数:
        bot: Bot
        user: 用户数据
        plugin: 插件数据
        session: Uninfo

    异常:
        IsSuperuserException: 超级用户
        IsSuperuserException: 超级用户

    返回:
        int: 调用插件金币费用
    """
    cost_gold = await auth_cost(user, plugin, session)
    if session.user.id in bot.config.superusers:
        if plugin.plugin_type == PluginType.SUPERUSER:
            raise IsSuperuserException()
        if not plugin.limit_superuser:
            raise IsSuperuserException()
    return cost_gold


async def reduce_gold(user_id: str, module: str, cost_gold: int, session: Uninfo):
    """扣除用户金币

    参数:
        user_id: 用户id
        module: 插件模块名称
        cost_gold: 消耗金币
        session: Uninfo
    """
    user_cache = Cache[UserConsole](CacheType.USERS)
    try:
        await UserConsole.reduce_gold(
            user_id,
            cost_gold,
            GoldHandle.PLUGIN,
            module,
            PlatformUtils.get_platform(session),
        )
    except InsufficientGold:
        if u := await UserConsole.get_user(user_id):
            u.gold = 0
            await u.save(update_fields=["gold"])
    # 更新缓存
    await user_cache.update(user_id)
    logger.debug(f"调用功能花费金币: {cost_gold}", LOGGER_COMMAND, session=session)


async def auth(
    matcher: Matcher,
    event: Event,
    bot: Bot,
    session: Uninfo,
    message: UniMsg,
):
    """权限检查

    参数:
        matcher: matcher
        event: Event
        bot: bot
        session: Uninfo
        message: UniMsg
    """
    cost_gold = 0
    ignore_flag = False
    entity = get_entity_ids(session)
    module = matcher.plugin_name or ""
    try:
        if not module:
            raise PermissionExemption("Matcher插件名称不存在...")
        plugin, user = await get_plugin_and_user(module, entity.user_id)
        cost_gold = await get_plugin_cost(bot, user, plugin, session)
        bot_filter(session)
        await asyncio.gather(
            *[
                auth_ban(matcher, bot, session),
                auth_bot(plugin, bot.self_id),
                auth_group(plugin, entity, message),
                auth_admin(plugin, session),
                auth_plugin(plugin, session, event),
                auth_limit(plugin, session),
            ]
        )
    except SkipPluginException as e:
        LimitManager.unblock(module, entity.user_id, entity.group_id, entity.channel_id)
        logger.info(str(e), LOGGER_COMMAND, session=session)
        ignore_flag = True
    except IsSuperuserException:
        logger.debug("超级用户跳过权限检测...", LOGGER_COMMAND, session=session)
    except PermissionExemption as e:
        logger.info(str(e), LOGGER_COMMAND, session=session)
    if not ignore_flag and cost_gold > 0:
        await reduce_gold(entity.user_id, module, cost_gold, session)
    if ignore_flag:
        raise IgnoredException("权限检测 ignore")
