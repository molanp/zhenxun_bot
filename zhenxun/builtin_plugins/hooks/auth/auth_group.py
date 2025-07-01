from nonebot_plugin_alconna import UniMsg

from zhenxun.models.group_console import GroupConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.cache import Cache
from zhenxun.utils.enum import CacheType
from zhenxun.utils.utils import EntityIDs

from .config import SwitchEnum
from .exception import SkipPluginException


async def auth_group(plugin: PluginInfo, entity: EntityIDs, message: UniMsg):
    """群黑名单检测 群总开关检测

    参数:
        plugin: PluginInfo
        entity: EntityIDs
        message: UniMsg
    """
    if not entity.group_id:
        return
    text = message.extract_plain_text()
    group = await Cache[GroupConsole](CacheType.GROUPS).get(entity.group_id)
    if not group:
        raise SkipPluginException("群组信息不存在...")
    if group.level < 0:
        raise SkipPluginException("群组黑名单, 目标群组群权限权限-1...")
    if text.strip() != SwitchEnum.ENABLE and not group.status:
        raise SkipPluginException("群组休眠状态...")
    if plugin.level > group.level:
        raise SkipPluginException(
            f"{plugin.name}({plugin.module}) 群等级限制，"
            f"该功能需要的群等级: {plugin.level}..."
        )
