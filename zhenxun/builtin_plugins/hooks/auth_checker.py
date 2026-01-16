import asyncio
import time

from nonebot.adapters import Bot, Event
from nonebot.exception import IgnoredException
from nonebot.matcher import Matcher
from nonebot_plugin_alconna import UniMsg
from nonebot_plugin_uninfo import Uninfo
from tortoise.exceptions import IntegrityError

from zhenxun.models.group_console import GroupConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.user_console import UserConsole
from zhenxun.services.data_access import DataAccess
from zhenxun.services.log import logger
from zhenxun.utils.enum import GoldHandle, PluginType
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
from .auth.config import LOGGER_COMMAND, WARNING_THRESHOLD
from .auth.exception import (
    IsSuperuserException,
    PermissionExemption,
    SkipPluginException,
)

# 超时设置（秒）
TIMEOUT_SECONDS = 5.0


# 简单超时封装：只做一次调用，不做熔断/重试
async def with_timeout(coro, timeout=TIMEOUT_SECONDS, name=None):
    """带超时控制的协程执行（单次）。

    仅包装 asyncio.wait_for，在超时时打日志并抛出异常，不做熔断和重试逻辑。

    参数:
        coro: 要执行的协程。
        timeout: 超时时间（秒）。
        name: 操作名称，用于日志记录。

    返回:
        协程的返回值；超时时抛出 TimeoutError。
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as e:
        if name:
            logger.error(f"{name} 操作超时 (>{timeout}s)", LOGGER_COMMAND, e=e)
        raise



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
    user_dao = DataAccess(UserConsole)
    plugin_dao = DataAccess(PluginInfo)

    # 并行查询插件和用户数据
    plugin_task = plugin_dao.safe_get_or_none(module=module)
    user_task = user_dao.get_by_func_or_none(
        UserConsole.get_user, False, user_id=user_id
    )

    try:
        plugin, user = await with_timeout(
            asyncio.gather(plugin_task, user_task), name="get_plugin_and_user"
        )
    except asyncio.TimeoutError:
        # 超时直接抛出，由上层统一处理
        raise
    except IntegrityError as e:
        # 数据竞争时直接提示跳过本次权限检查，避免多次重试
        raise PermissionExemption("用户数据存在竞争，已跳过该次权限检查...") from e

    if not plugin:
        raise PermissionExemption(f"插件:{module} 数据不存在，已跳过权限检查...")
    if plugin.plugin_type == PluginType.HIDDEN:
        raise PermissionExemption(
            f"插件: {plugin.name}:{plugin.module} 为HIDDEN，已跳过权限检查..."
        )
    user = None
    try:
        user = await user_dao.get_by_func_or_none(
            UserConsole.get_user, False, user_id=user_id
        )
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
    cost_gold = await with_timeout(auth_cost(user, plugin, session), name="auth_cost")
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
    user_dao = DataAccess(UserConsole)
    try:
        await with_timeout(
            UserConsole.reduce_gold(
                user_id,
                cost_gold,
                GoldHandle.PLUGIN,
                module,
                PlatformUtils.get_platform(session),
            ),
            name="reduce_gold",
        )
    except InsufficientGold:
        if u := await UserConsole.get_user(user_id):
            u.gold = 0
            await u.save(update_fields=["gold"])
    except asyncio.TimeoutError:
        logger.error(
            f"扣除金币超时，用户: {user_id}, 金币: {cost_gold}",
            LOGGER_COMMAND,
            session=session,
        )

    # 清除缓存，使下次查询时从数据库获取最新数据
    await user_dao.clear_cache(user_id=user_id)
    logger.debug(f"调用功能花费金币: {cost_gold}", LOGGER_COMMAND, session=session)


# 辅助函数，用于记录每个 hook 的执行时间
async def time_hook(coro, name, time_dict):
    start = time.perf_counter()
    try:
        # 添加超时控制（单次执行）
        return await with_timeout(coro, name=name)
    except asyncio.TimeoutError:
        time_dict[name] = f"超时 (>{TIMEOUT_SECONDS}s)"
    finally:
        if name not in time_dict:
            time_dict[name] = f"{time.perf_counter() - start:.3f}s"


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
    start_time = time.perf_counter()
    cost_gold = 0
    ignore_flag = False
    entity = get_entity_ids(session)
    module = matcher.plugin_name or ""

    # 用于记录各个 hook 的执行时间
    hook_times = {}
    hooks_start = time.perf_counter()

    try:
        if not module:
            raise PermissionExemption("Matcher插件名称不存在...")

        # 获取插件和用户数据
        plugin_user_start = time.perf_counter()
        try:
            plugin, user = await with_timeout(
                get_plugin_and_user(module, entity.user_id), name="get_plugin_and_user"
            )
            hook_times["get_plugin_user"] = (
                f"{time.perf_counter() - plugin_user_start:.3f}s"
            )
        except asyncio.TimeoutError as e:
            logger.error(
                f"获取插件和用户数据超时，模块: {module}",
                LOGGER_COMMAND,
                session=session,
            )
            raise PermissionExemption("获取插件和用户数据超时，请稍后再试...") from e

        # 获取插件费用
        cost_start = time.perf_counter()
        try:
            cost_gold = await with_timeout(
                get_plugin_cost(bot, user, plugin, session), name="get_plugin_cost"
            )
            hook_times["cost_gold"] = f"{time.perf_counter() - cost_start:.3f}s"
        except asyncio.TimeoutError:
            logger.error(
                f"获取插件费用超时，模块: {module}", LOGGER_COMMAND, session=session
            )
            # 继续执行，不阻止权限检查

        # 执行 bot_filter
        bot_filter(session)

        group = None
        if entity.group_id:
            group_dao = DataAccess(GroupConsole)
            group = await with_timeout(
                group_dao.safe_get_or_none(
                    group_id=entity.group_id, channel_id__isnull=True
                ),
                name="get_group",
            )

        # 创建所有 hook 任务
        hook_tasks = [
            time_hook(auth_ban(matcher, bot, session, plugin), "auth_ban", hook_times),
            time_hook(auth_bot(plugin, bot.self_id), "auth_bot", hook_times),
            time_hook(
                auth_group(plugin, group, message, entity.group_id),
                "auth_group",
                hook_times,
            ),
            time_hook(auth_admin(plugin, session), "auth_admin", hook_times),
            time_hook(
                auth_plugin(plugin, group, session, event), "auth_plugin", hook_times
            ),
            time_hook(auth_limit(plugin, session), "auth_limit", hook_times),
        ]

        # 使用 gather 并行执行所有 hook，但添加总体超时控制
        try:
            await with_timeout(
                asyncio.gather(*hook_tasks),
                timeout=TIMEOUT_SECONDS * 2,  # 给总体执行更多时间
                name="auth_hooks_gather",
            )
        except asyncio.TimeoutError:
            logger.error(
                f"权限检查 hooks 总体执行超时，模块: {module}",
                LOGGER_COMMAND,
                session=session,
            )
            # 不抛出异常，允许继续执行

    except SkipPluginException as e:
        LimitManager.unblock(module, entity.user_id, entity.group_id, entity.channel_id)
        logger.info(str(e), LOGGER_COMMAND, session=session)
        ignore_flag = True
    except IsSuperuserException:
        logger.debug("超级用户跳过权限检测...", LOGGER_COMMAND, session=session)
    except PermissionExemption as e:
        logger.info(str(e), LOGGER_COMMAND, session=session)
    finally:
        hooks_time = time.perf_counter() - hooks_start
        logger.debug(
            f"hooks gather 实际耗时: {hooks_time:.3f}s, 详情: {hook_times}",
            LOGGER_COMMAND,
            session=session,
        )
    # 扣除金币
    if not ignore_flag and cost_gold > 0:
        gold_start = time.perf_counter()
        try:
            await with_timeout(
                reduce_gold(entity.user_id, module, cost_gold, session),
                name="reduce_gold",
            )
            hook_times["reduce_gold"] = f"{time.perf_counter() - gold_start:.3f}s"
        except asyncio.TimeoutError:
            logger.error(
                f"扣除金币超时，模块: {module}", LOGGER_COMMAND, session=session
            )

    # 记录总执行时间
    total_time = time.perf_counter() - start_time
    if total_time > WARNING_THRESHOLD:  # 如果总时间超过500ms，记录详细信息
        logger.warning(
            f"权限检查耗时过长: {total_time:.3f}s, 模块: {module}, "
            f"hooks时间: {hooks_time:.3f}s, "
            f"详情: {hook_times}",
            LOGGER_COMMAND,
            session=session,
        )

    if ignore_flag:
        raise IgnoredException("权限检测 ignore")
