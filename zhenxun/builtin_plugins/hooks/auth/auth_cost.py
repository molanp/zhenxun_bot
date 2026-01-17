import asyncio

from nonebot.adapters import Bot
from nonebot_plugin_uninfo import Uninfo

from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.user_console import UserConsole
from zhenxun.services.data_access import DataAccess
from zhenxun.services.log import logger
from zhenxun.utils.enum import GoldHandle, PluginType
from zhenxun.utils.platform import PlatformUtils

from .config import LOGGER_COMMAND
from .exception import IsSuperuserException, SkipPluginException
from .utils import send_message, with_timeout


async def auth_cost(
    bot: Bot, user: UserConsole, plugin: PluginInfo, session: Uninfo
) -> int:
    """处理插件金币逻辑：检查并扣除金币。

    对于非超级用户，先检查是否满足金币条件，如果不足则提示并抛出跳过异常。
    对于满足条件的用户，直接扣除对应金币并清理缓存。

    参数:
        bot: Bot
        user: 用户数据
        plugin: 插件数据
        session: Uninfo

    异常:
        IsSuperuserException: 超级用户不消耗金币且跳过后续处理。
        SkipPluginException: 金币不足时跳过插件执行。

    返回:
        int: 实际扣除的金币数量（超级用户返回 0）。
    """
    # 超级用户接跳过插件
    if session.user.id in bot.config.superusers:
        if plugin.plugin_type == PluginType.SUPERUSER:
            raise IsSuperuserException()
        if not plugin.limit_superuser:
            raise IsSuperuserException()
        return 0

    cost_gold = plugin.cost_gold

    # 金币检查
    if cost_gold > 0:
        if user.gold < cost_gold:
            await send_message(session, f"金币不足..该功能需要{plugin.cost_gold}金币..")
            raise SkipPluginException(f"{plugin.name}({plugin.module}) 金币限制...")

        # 扣款并写日志
        user_dao = DataAccess(UserConsole)
        try:
            await with_timeout(
                UserConsole.reduce_gold(
                    user.user_id,
                    cost_gold,
                    GoldHandle.PLUGIN,
                    plugin.module,
                    PlatformUtils.get_platform(session),
                    no_raise=True,
                ),
                name="reduce_gold",
            )
        except asyncio.TimeoutError:
            logger.error(
                f"扣除金币超时，用户: {user.user_id}, 金币: {cost_gold}",
                LOGGER_COMMAND,
                session=session,
            )
            # 扣费超时视为本次不扣费，避免影响整体流程
            return 0

        # 清除缓存，使下次查询时从数据库获取最新数据
        await user_dao.clear_cache(user_id=user.user_id)
        logger.debug(f"调用功能花费金币: {cost_gold}", LOGGER_COMMAND, session=session)

    return cost_gold
