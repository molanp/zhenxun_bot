import asyncio
import time

from nonebot.adapters import Bot
from nonebot.matcher import Matcher
from nonebot_plugin_alconna import At
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import Config
from zhenxun.models.ban_console import BanConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.data_access import DataAccess
from zhenxun.services.db_context import DB_TIMEOUT_SECONDS
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.utils import EntityIDs, get_entity_ids

from .config import LOGGER_COMMAND, WARNING_THRESHOLD
from .exception import SkipPluginException
from .utils import freq, send_message

Config.add_plugin_config(
    "hook",
    "BAN_RESULT",
    "才不会给你发消息.",
    help="对被ban用户发送的消息",
)


async def calculate_ban_time(ban_record: BanConsole | None) -> int:
    """根据ban记录计算剩余ban时间

    参数:
        ban_record: BanConsole记录

    返回:
        int: ban剩余时长，-1时为永久ban，0表示未被ban
    """
    if not ban_record:
        return 0

    if ban_record.duration == -1:
        return -1

    _time = time.perf_counter() - (ban_record.ban_time + ban_record.duration)
    if _time < 0:
        return int(abs(_time))
    await ban_record.delete()
    return 0


async def is_ban(user_id: str | None, group_id: str | None) -> int:
    """检查用户或群组是否被 ban。

    会优先检查群组的 ban 状态，仅在群组未被 ban 时才继续检查用户 ban

    参数:
        user_id: 用户 ID。
        group_id: 群组 ID。

    返回:
        int: ban 的剩余时间，0 表示未被 ban，-1 表示永久 ban。
    """
    if not user_id and not group_id:
        return 0

    start_time = time.perf_counter()
    ban_dao = DataAccess(BanConsole)

    # 分别获取用户在群组中的 ban 记录和全局 ban 记录
    group_user = None
    user = None

    try:
        # 优先检查群组维度的 ban（群组 ban > 用户 ban）
        if group_id:
            try:
                group_user = await asyncio.wait_for(
                    ban_dao.safe_get_or_none(
                        user_id=user_id or None,
                        group_id=group_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"查询 群组 ban 记录超时: user_id={user_id}, group_id={group_id}",
                    LOGGER_COMMAND,
                )
                group_user = None

        # 如果群组已经被 ban，直接根据群组记录计算剩余时间
        if group_user:
            logger.debug(f"查询到的 群组 ban 记录: {group_user}", LOGGER_COMMAND)
            return await calculate_ban_time(group_user)

        # 群组未被 ban，再检查用户维度 ban（全局 ban）
        if user_id:
            try:
                user = await asyncio.wait_for(
                    ban_dao.safe_get_or_none(user_id=user_id, group_id__isnull=True),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"查询 用户 ban 记录超时: user_id={user_id}, group_id={group_id}",
                    LOGGER_COMMAND,
                )
                user = None

        # 如果没有任何记录，返回 0
        if not user:
            return 0

        logger.debug(f"查询到的用户 ban 记录: {user}", LOGGER_COMMAND)
        # 直接计算用户 ban 剩余时间
        return await calculate_ban_time(user)
    finally:
        # 记录执行时间
        elapsed = time.perf_counter() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过阈值的检查
            logger.warning(
                f"is_ban 耗时: {elapsed:.3f}s",
                LOGGER_COMMAND,
                session=user_id,
                group_id=group_id,
            )


def is_hidden_plugin(matcher: Matcher) -> bool:
    """判断插件是否为隐藏插件

    参数:
        matcher: 当前触发的 Matcher 实例。

    返回:
        bool: 隐藏插件为 True
    """
    plugin = matcher.plugin
    if not plugin:
        return False

    metadata = plugin.metadata
    if not metadata:
        return False

    extra = metadata.extra
    # 使用字典查找并提前返回，避免多层嵌套判断带来的额外开销
    return extra.get("plugin_type") in {PluginType.HIDDEN}


def format_time(time_val: float) -> str:
    """格式化时间

    参数:
        time_val: ban时长

    返回:
        str: 格式化时间文本
    """
    if time_val == -1:
        return "∞"
    time_val = abs(int(time_val))
    if time_val < 60:
        time_str = f"{time_val!s} 秒"
    else:
        minute = int(time_val / 60)
        if minute > 60:
            hours = minute // 60
            minute %= 60
            time_str = f"{hours} 小时 {minute}分钟"
        else:
            time_str = f"{minute} 分钟"
    return time_str


async def group_handle(group_id: str) -> None:
    """群组ban检查

    参数:
        group_id: 群组id

    异常:
        SkipPluginException: 群组处于黑名单
    """
    start_time = time.perf_counter()
    try:
        if await is_ban(None, group_id):
            raise SkipPluginException("群组处于黑名单中...")
    finally:
        # 记录执行时间
        elapsed = time.perf_counter() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"group_handle 耗时: {elapsed:.3f}s",
                LOGGER_COMMAND,
                group_id=group_id,
            )


async def user_handle(plugin: PluginInfo, entity: EntityIDs, session: Uninfo) -> None:
    """用户ban检查

    参数:
        module: 插件模块名
        entity: 实体ID信息
        session: Uninfo

    异常:
        SkipPluginException: 用户处于黑名单
    """
    start_time = time.perf_counter()
    try:
        ban_result = Config.get_config("hook", "BAN_RESULT")
        time_val = await is_ban(entity.user_id, entity.group_id)
        if not time_val:
            return
        time_str = format_time(time_val)

        if (
            plugin
            and time_val != -1
            and ban_result
            and freq.is_send_limit_message(plugin, entity.user_id, False)
        ):
            try:
                await asyncio.wait_for(
                    send_message(
                        session,
                        [
                            At(flag="user", target=entity.user_id),
                            f"{ban_result}\n在..在 {time_str} 后才会理你喔",
                        ],
                        entity.user_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(f"发送消息超时: {entity.user_id}", LOGGER_COMMAND)
        raise SkipPluginException("用户处于黑名单中...")
    finally:
        # 记录执行时间
        elapsed = time.perf_counter() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"user_handle 耗时: {elapsed:.3f}s",
                LOGGER_COMMAND,
                session=session,
            )


async def auth_ban(
    matcher: Matcher, bot: Bot, session: Uninfo, plugin: PluginInfo
) -> None:
    """权限检查 - ban 检查

    参数:
        matcher: Matcher
        bot: Bot
        session: Uninfo
    """
    start_time = time.perf_counter()
    try:
        if is_hidden_plugin(matcher):
            return
        if not matcher.plugin_name:
            return
        entity = get_entity_ids(session)
        if entity.user_id in bot.config.superusers:
            return
        if entity.group_id:
            try:
                await asyncio.wait_for(
                    group_handle(entity.group_id), timeout=DB_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(f"群组ban检查超时: {entity.group_id}", LOGGER_COMMAND)
                # 超时时不阻塞，继续执行

        if entity.user_id:
            try:
                await asyncio.wait_for(
                    user_handle(plugin, entity, session),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(f"用户ban检查超时: {entity.user_id}", LOGGER_COMMAND)
                # 超时时不阻塞，继续执行
    finally:
        # 记录总执行时间
        elapsed = time.perf_counter() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"auth_ban 总耗时: {elapsed:.3f}s, plugin={matcher.plugin_name}",
                LOGGER_COMMAND,
                session=session,
            )
