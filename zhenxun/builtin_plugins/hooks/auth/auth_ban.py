import asyncio

from nonebot.adapters import Bot
from nonebot.matcher import Matcher
from nonebot_plugin_alconna import At
from nonebot_plugin_uninfo import Uninfo
from tortoise.exceptions import MultipleObjectsReturned

from zhenxun.configs.config import Config
from zhenxun.models.ban_console import BanConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.cache import Cache
from zhenxun.services.log import logger
from zhenxun.utils.enum import CacheType, PluginType
from zhenxun.utils.utils import EntityIDs, get_entity_ids

from .config import LOGGER_COMMAND
from .exception import SkipPluginException
from .utils import freq, send_message

Config.add_plugin_config(
    "hook",
    "BAN_RESULT",
    "才不会给你发消息.",
    help="对被ban用户发送的消息",
)


async def is_ban(user_id: str | None, group_id: str | None) -> int:
    if not user_id and not group_id:
        return 0
    cache = Cache[BanConsole](CacheType.BAN)
    group_user, user = await asyncio.gather(
        cache.get(user_id, group_id), cache.get(user_id)
    )
    results = []
    if group_user:
        results.append(group_user)
    if user:
        results.append(user)
    if not results:
        return 0
    for result in results:
        if result.duration > 0 or result.duration == -1:
            return await BanConsole.check_ban_time(user_id, group_id)
    return 0


def check_plugin_type(matcher: Matcher) -> bool:
    """判断插件类型是否是隐藏插件

    参数:
        matcher: Matcher

    返回:
        bool: 是否为隐藏插件
    """
    if plugin := matcher.plugin:
        if metadata := plugin.metadata:
            extra = metadata.extra
            if extra.get("plugin_type") in [PluginType.HIDDEN]:
                return False
    return True


def format_time(time: float) -> str:
    """格式化时间

    参数:
        time: ban时长

    返回:
        str: 格式化时间文本
    """
    if time == -1:
        return "∞"
    time = abs(int(time))
    if time < 60:
        time_str = f"{time!s} 秒"
    else:
        minute = int(time / 60)
        if minute > 60:
            hours = minute // 60
            minute %= 60
            time_str = f"{hours} 小时 {minute}分钟"
        else:
            time_str = f"{minute} 分钟"
    return time_str


async def group_handle(cache: Cache[list[BanConsole]], group_id: str):
    """群组ban检查

    参数:
        cache: cache
        group_id: 群组id

    异常:
        SkipPluginException: 群组处于黑名单
    """
    try:
        if await is_ban(None, group_id):
            raise SkipPluginException("群组处于黑名单中...")
    except MultipleObjectsReturned:
        logger.warning(
            "群组黑名单数据重复，过滤该次hook并移除多余数据...", LOGGER_COMMAND
        )
        ids = await BanConsole.filter(user_id="", group_id=group_id).values_list(
            "id", flat=True
        )
        await BanConsole.filter(id__in=ids[:-1]).delete()
        await cache.reload()


async def user_handle(
    module: str, cache: Cache[list[BanConsole]], entity: EntityIDs, session: Uninfo
):
    """用户ban检查

    参数:
        module: 插件模块名
        cache: cache
        user_id: 用户id
        session: Uninfo

    异常:
        SkipPluginException: 用户处于黑名单
    """
    ban_result = Config.get_config("hook", "BAN_RESULT")
    try:
        time = await is_ban(entity.user_id, entity.group_id)
        if not time:
            return
        time_str = format_time(time)
        db_plugin = await Cache[PluginInfo](CacheType.PLUGINS).get(module)
        if (
            db_plugin
            # and not db_plugin.ignore_prompt
            and time != -1
            and ban_result
            and freq.is_send_limit_message(db_plugin, entity.user_id, False)
        ):
            await send_message(
                session,
                [
                    At(flag="user", target=entity.user_id),
                    f"{ban_result}\n在..在 {time_str} 后才会理你喔",
                ],
                entity.user_id,
            )
        raise SkipPluginException("用户处于黑名单中...")
    except MultipleObjectsReturned:
        logger.warning(
            "用户黑名单数据重复，过滤该次hook并移除多余数据...", LOGGER_COMMAND
        )
        ids = await BanConsole.filter(user_id=entity.user_id, group_id="").values_list(
            "id", flat=True
        )
        await BanConsole.filter(id__in=ids[:-1]).delete()
        await cache.reload()


async def auth_ban(matcher: Matcher, bot: Bot, session: Uninfo):
    if not check_plugin_type(matcher):
        return
    if not matcher.plugin_name:
        return
    entity = get_entity_ids(session)
    if entity.user_id in bot.config.superusers:
        return
    cache = Cache[list[BanConsole]](CacheType.BAN)
    if entity.group_id:
        await group_handle(cache, entity.group_id)
    if entity.user_id:
        await user_handle(matcher.plugin_name, cache, entity, session)
