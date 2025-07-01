from nonebot_plugin_alconna import At
from nonebot_plugin_uninfo import Uninfo

from zhenxun.models.level_user import LevelUser
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.cache import Cache
from zhenxun.utils.enum import CacheType
from zhenxun.utils.utils import get_entity_ids

from .exception import SkipPluginException
from .utils import send_message


async def auth_admin(plugin: PluginInfo, session: Uninfo):
    """管理员命令 个人权限

    参数:
        plugin: PluginInfo
        session: Uninfo
    """
    if not plugin.admin_level:
        return
    entity = get_entity_ids(session)
    cache = Cache[list[LevelUser]](CacheType.LEVEL)
    user_list = await cache.get(session.user.id) or []
    if entity.group_id:
        user_list += await cache.get(session.user.id, entity.group_id) or []
        if user_list:
            user = max(user_list, key=lambda x: x.user_level)
            user_level = user.user_level
        else:
            user_level = 0
        if user_level < plugin.admin_level:
            await send_message(
                session,
                [
                    At(flag="user", target=session.user.id),
                    f"你的权限不足喔，该功能需要的权限等级: {plugin.admin_level}",
                ],
                entity.user_id,
            )
            raise SkipPluginException(
                f"{plugin.name}({plugin.module}) 管理员权限不足..."
            )
    elif user_list:
        user = max(user_list, key=lambda x: x.user_level)
        if user.user_level < plugin.admin_level:
            await send_message(
                session,
                f"你的权限不足喔，该功能需要的权限等级: {plugin.admin_level}",
            )
        raise SkipPluginException(f"{plugin.name}({plugin.module}) 管理员权限不足...")
