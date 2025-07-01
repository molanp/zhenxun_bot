from nonebot.adapters import Event
from nonebot_plugin_uninfo import Uninfo

from zhenxun.models.group_console import GroupConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.cache import Cache
from zhenxun.utils.common_utils import CommonUtils
from zhenxun.utils.enum import BlockType, CacheType
from zhenxun.utils.utils import get_entity_ids

from .exception import IsSuperuserException, SkipPluginException
from .utils import freq, is_poke, send_message


class GroupCheck:
    def __init__(
        self, plugin: PluginInfo, group_id: str, session: Uninfo, is_poke: bool
    ) -> None:
        self.group_id = group_id
        self.session = session
        self.is_poke = is_poke
        self.plugin = plugin

    async def __get_data(self):
        cache = Cache[GroupConsole](CacheType.GROUPS)
        return await cache.get(self.group_id)

    async def check(self):
        await self.check_superuser_block(self.plugin)

    async def check_superuser_block(self, plugin: PluginInfo):
        """超级用户禁用群组插件检测

        参数:
            plugin: PluginInfo

        异常:
            IgnoredException: 忽略插件
        """
        group = await self.__get_data()
        if group and CommonUtils.format(plugin.module) in group.superuser_block_plugin:
            if freq.is_send_limit_message(plugin, group.group_id, self.is_poke):
                await send_message(
                    self.session, "超级管理员禁用了该群此功能...", self.group_id
                )
            raise SkipPluginException(
                f"{plugin.name}({plugin.module}) 超级管理员禁用了该群此功能..."
            )
        await self.check_normal_block(self.plugin)

    async def check_normal_block(self, plugin: PluginInfo):
        """群组插件状态

        参数:
            plugin: PluginInfo

        异常:
            IgnoredException: 忽略插件
        """
        group = await self.__get_data()
        if group and CommonUtils.format(plugin.module) in group.block_plugin:
            if freq.is_send_limit_message(plugin, self.group_id, self.is_poke):
                await send_message(self.session, "该群未开启此功能...", self.group_id)
            raise SkipPluginException(f"{plugin.name}({plugin.module}) 未开启此功能...")
        await self.check_global_block(self.plugin)

    async def check_global_block(self, plugin: PluginInfo):
        """全局禁用插件检测

        参数:
            plugin: PluginInfo

        异常:
            IgnoredException: 忽略插件
        """
        if plugin.block_type == BlockType.GROUP:
            """全局群组禁用"""
            if freq.is_send_limit_message(plugin, self.group_id, self.is_poke):
                await send_message(
                    self.session, "该功能在群组中已被禁用...", self.group_id
                )
            raise SkipPluginException(
                f"{plugin.name}({plugin.module}) 该插件在群组中已被禁用..."
            )


class PluginCheck:
    def __init__(self, group_id: str | None, session: Uninfo, is_poke: bool):
        self.session = session
        self.is_poke = is_poke
        self.group_id = group_id

    async def check_user(self, plugin: PluginInfo):
        """全局私聊禁用检测

        参数:
            plugin: PluginInfo

        异常:
            IgnoredException: 忽略插件
        """
        if plugin.block_type == BlockType.PRIVATE:
            if freq.is_send_limit_message(plugin, self.session.user.id, self.is_poke):
                await send_message(self.session, "该功能在私聊中已被禁用...")
            raise SkipPluginException(
                f"{plugin.name}({plugin.module}) 该插件在私聊中已被禁用..."
            )

    async def check_global(self, plugin: PluginInfo):
        """全局状态

        参数:
            plugin: PluginInfo

        异常:
            IgnoredException: 忽略插件
        """
        if plugin.status or plugin.block_type != BlockType.ALL:
            return
        """全局状态"""
        cache = Cache[GroupConsole](CacheType.GROUPS)
        if self.group_id and (group := await cache.get(self.group_id)):
            if group.is_super:
                raise IsSuperuserException()
        sid = self.group_id or self.session.user.id
        if freq.is_send_limit_message(plugin, sid, self.is_poke):
            await send_message(self.session, "全局未开启此功能...", sid)
        raise SkipPluginException(f"{plugin.name}({plugin.module}) 全局未开启此功能...")


async def auth_plugin(plugin: PluginInfo, session: Uninfo, event: Event):
    """插件状态

    参数:
        plugin: PluginInfo
        session: Uninfo
        event: Event
    """
    entity = get_entity_ids(session)
    is_poke_event = is_poke(event)
    user_check = PluginCheck(entity.group_id, session, is_poke_event)
    if entity.group_id:
        group_check = GroupCheck(plugin, entity.group_id, session, is_poke_event)
        await group_check.check()
    else:
        await user_check.check_user(plugin)
    await user_check.check_global(plugin)
