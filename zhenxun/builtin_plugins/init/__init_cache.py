from zhenxun.models.ban_console import BanConsole
from zhenxun.models.bot_console import BotConsole
from zhenxun.models.group_console import GroupConsole
from zhenxun.models.level_user import LevelUser
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.user_console import UserConsole
from zhenxun.services.cache import CacheData, CacheRoot
from zhenxun.services.log import logger
from zhenxun.utils.enum import CacheType


@CacheRoot.new(CacheType.PLUGINS)
async def _():
    """初始化插件缓存"""
    data_list = await PluginInfo.get_plugins()
    return {p.module: p for p in data_list}


@CacheRoot.getter(CacheType.PLUGINS, result_model=PluginInfo)
async def _(cache_data: CacheData, module: str):
    """获取插件缓存"""
    data = await cache_data.get_key(module)
    if not data:
        if plugin := await PluginInfo.get_plugin(module=module):
            await cache_data.set_key(module, plugin)
            logger.debug(f"插件 {module} 数据已设置到缓存")
            return plugin
    return data


@CacheRoot.with_refresh(CacheType.PLUGINS)
async def _(cache_data: CacheData, data: dict[str, PluginInfo] | None):
    """刷新插件缓存"""
    if not data:
        return
    plugins = await PluginInfo.filter(module__in=data.keys(), load_status=True).all()
    for plugin in plugins:
        await cache_data.set_key(plugin.module, plugin)


@CacheRoot.new(CacheType.GROUPS)
async def _():
    """初始化群组缓存"""
    data_list = await GroupConsole.all()
    return {p.group_id: p for p in data_list if not p.channel_id}


@CacheRoot.getter(CacheType.GROUPS, result_model=GroupConsole)
async def _(cache_data: CacheData, group_id: str):
    """获取群组缓存"""
    data = await cache_data.get_key(group_id)
    if not data:
        if group := await GroupConsole.get_group(group_id=group_id):
            await cache_data.set_key(group_id, group)
            return group
    return data


@CacheRoot.with_refresh(CacheType.GROUPS)
async def _(cache_data: CacheData, data: dict[str, GroupConsole] | None):
    """刷新群组缓存"""
    if not data:
        return
    groups = await GroupConsole.filter(
        group_id__in=data.keys(), channel_id__isnull=True
    ).all()
    for group in groups:
        await cache_data.set_key(group.group_id, group)


@CacheRoot.new(CacheType.BOT)
async def _():
    """初始化机器人缓存"""
    data_list = await BotConsole.all()
    return {p.bot_id: p for p in data_list}


@CacheRoot.getter(CacheType.BOT, result_model=BotConsole)
async def _(cache_data: CacheData, bot_id: str):
    """获取机器人缓存"""
    data = await cache_data.get_key(bot_id)
    if not data:
        if bot := await BotConsole.get_or_none(bot_id=bot_id):
            await cache_data.set_key(bot_id, bot)
            return bot
    return data


@CacheRoot.with_refresh(CacheType.BOT)
async def _(cache_data: CacheData, data: dict[str, BotConsole] | None):
    """刷新机器人缓存"""
    if not data:
        return
    bots = await BotConsole.filter(bot_id__in=data.keys()).all()
    for bot in bots:
        await cache_data.set_key(bot.bot_id, bot)


@CacheRoot.new(CacheType.USERS)
async def _():
    """初始化用户缓存"""
    data_list = await UserConsole.all()
    return {p.user_id: p for p in data_list}


@CacheRoot.getter(CacheType.USERS, result_model=UserConsole)
async def _(cache_data: CacheData, user_id: str):
    """获取用户缓存"""
    data = await cache_data.get_key(user_id)
    if not data:
        if user := await UserConsole.get_user(user_id=user_id):
            await cache_data.set_key(user_id, user)
            return user
    return data


@CacheRoot.with_refresh(CacheType.USERS)
async def _(cache_data: CacheData, data: dict[str, UserConsole] | None):
    """刷新用户缓存"""
    if not data:
        return
    users = await UserConsole.filter(user_id__in=data.keys()).all()
    for user in users:
        await cache_data.set_key(user.user_id, user)


@CacheRoot.new(CacheType.LEVEL)
async def _():
    """初始化等级缓存"""
    data_list = await LevelUser().all()
    return {f"{d.user_id}:{d.group_id or ''}": d for d in data_list}


@CacheRoot.getter(CacheType.LEVEL, result_model=list[LevelUser])
async def _(cache_data: CacheData, user_id: str, group_id: str | None = None):
    """获取等级缓存"""
    key = f"{user_id}:{group_id or ''}"
    data = await cache_data.get_key(key)
    if not data:
        if group_id:
            data = await LevelUser.filter(user_id=user_id, group_id=group_id).all()
        else:
            data = await LevelUser.filter(user_id=user_id, group_id__isnull=True).all()
        if data:
            await cache_data.set_key(key, data)
            return data
    return data or []


@CacheRoot.new(CacheType.BAN, False)
async def _():
    """初始化封禁缓存"""
    data_list = await BanConsole.all()
    return {f"{d.group_id or ''}:{d.user_id or ''}": d for d in data_list}


@CacheRoot.getter(CacheType.BAN, result_model=BanConsole)
async def _(cache_data: CacheData, user_id: str | None, group_id: str | None = None):
    """获取封禁缓存"""
    if not user_id and not group_id:
        return []
    key = f"{group_id or ''}:{user_id or ''}"
    data = await cache_data.get_key(key)
    # if not data:
    #     start = time.time()
    #     if user_id and group_id:
    #         data = await BanConsole.filter(user_id=user_id, group_id=group_id).all()
    #     elif user_id:
    #     data = await BanConsole.filter(user_id=user_id, group_id__isnull=True).all()
    #     elif group_id:
    #         data = await BanConsole.filter(
    #             user_id__isnull=True, group_id=group_id
    #         ).all()
    #     logger.info(
    #         f"获取封禁缓存耗时: {time.time() - start:.2f}秒, key: {key}, data: {data}"
    #     )
    #     if data:
    #         await cache_data.set_key(key, data)
    #         return data
    return data or []


# @CacheRoot.new(CacheType.LIMIT)
# async def _():
#     """初始化限制缓存"""
#     data_list = await PluginLimit.filter(status=True).all()
#     return {data.module: data for data in data_list}


# @CacheRoot.getter(CacheType.LIMIT, result_model=list[PluginLimit])
# async def _(cache_data: CacheData, module: str):
#     """获取限制缓存"""
#     data = await cache_data.get_key(module)
#     if not data:
#         if limits := await PluginLimit.filter(module=module, status=True):
#             await cache_data.set_key(module, limits)
#             return limits
#     return data or []


# @CacheRoot.with_refresh(CacheType.LIMIT)
# async def _(cache_data: CacheData, data: dict[str, list[PluginLimit]] | None):
#     """刷新限制缓存"""
#     if not data:
#         return
#     limits = await PluginLimit.filter(module__in=data.keys(), load_status=True).all()
#     for limit in limits:
#         await cache_data.set_key(limit.module, limit)
