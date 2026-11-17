import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import random
from typing import cast

import httpx
import nonebot
from nonebot.adapters import Bot
from nonebot.utils import is_coroutine_callable
from nonebot_plugin_alconna import SupportScope
from nonebot_plugin_alconna.uniseg import Receipt, Target, UniMessage
from nonebot_plugin_uninfo import SceneType, Uninfo, get_interface
from pydantic import BaseModel

from zhenxun.configs.config import BotConfig
from zhenxun.models.friend_user import FriendUser
from zhenxun.models.group_console import GroupConsole
from zhenxun.services.cache.cache_containers import CacheDict
from zhenxun.services.log import logger
from zhenxun.services.network_proxy import ManagedAsyncClient
from zhenxun.services.platform_identity import native_id
from zhenxun.utils.exception import AllURIsFailedError, NotFindSuperuser
from zhenxun.utils.http_utils import AsyncHttpx
from zhenxun.utils.message import MessageUtils

driver = nonebot.get_driver()

_ADMIN_ROLE_LEVEL = 10
"""uninfo 中管理员角色的等级，群主为 100"""
_ROLE_RETRY_EXPIRE = 30
_GROUP_ROLE_CACHE = CacheDict[int]("PLATFORM_GROUP_ROLE", expire=300, max_items=4096)


def _adapter_name(bot: Bot) -> str:
    adapter = getattr(bot, "adapter", None)
    if adapter is None:
        return ""
    get_name = getattr(adapter, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name()).lower()
        except Exception:
            return ""
    return adapter.__class__.__name__.lower()


def _scope_name(scope: object) -> str:
    """Normalize uninfo/alconna scope values without changing legacy platform."""
    if scope is None:
        return ""
    raw = getattr(scope, "name", None) or getattr(scope, "value", scope)
    text = str(raw or "").strip().lower()
    if not text:
        text = str(getattr(scope, "name", "") or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    compact = "".join(ch for ch in text if ch.isalnum())
    if compact == "qqclient":
        return "qq_client"
    if compact == "qqapi":
        return "qq_api"
    return text


_VERIFIED_SCOPE_ADAPTERS = {"qq_client": "onebot_v11", "qq_api": "qq"}


def _adapter_key(adapter: object) -> str:
    text = str(getattr(adapter, "value", adapter) or "").strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def _qualified_scope(scope: str, adapter: object) -> str:
    """为非验证适配器上报的 QQ 作用域加上适配器名

    参数:
        scope: uninfo 上报的作用域
        adapter: 适配器名称

    返回:
        str: qq_client 只属于 OneBot V11、qq_api 只属于 QQ 官方适配器，
            其他适配器返回 `作用域_适配器名`
    """
    expected = _VERIFIED_SCOPE_ADAPTERS.get(scope)
    name = _adapter_key(adapter)
    if expected is None or name == expected:
        return scope
    return f"{scope}_{name or 'unknown'}"


class UserData(BaseModel):
    name: str
    """昵称"""
    card: str | None = None
    """名片/备注"""
    user_id: str
    """用户id"""
    group_id: str | None = None
    """群组id"""
    channel_id: str | None = None
    """频道id"""
    role: str | None = None
    """角色"""
    avatar_url: str | None = None
    """头像url"""
    join_time: int | None = None
    """加入时间"""


class PlatformUtils:
    @classmethod
    def storage_bot_id(cls, scope: str, bot_id: str) -> str:
        """按平台作用域生成持久化使用的 Bot id

        参数:
            scope: 平台作用域
            bot_id: 适配器中的 Bot id

        返回:
            str: qq_client 为原始 id，其余平台带作用域前缀
        """
        bot_id = str(bot_id)
        return bot_id if scope == "qq_client" else f"{scope}:{bot_id}"

    @classmethod
    def get_storage_bot_id(cls, bot: Bot | Uninfo) -> str:
        """Return a persistence-safe bot ID without changing adapter routing IDs."""
        return cls.storage_bot_id(cls.get_platform_scope(bot), str(bot.self_id))

    @classmethod
    def is_superuser(cls, bot: Bot, user_id: str | None) -> bool:
        """判断用户是否为该 Bot 所在平台的超级用户

        参数:
            bot: Bot
            user_id: 用户id

        返回:
            bool: 是否在 get_superusers 的结果中
        """
        return bool(user_id) and str(user_id) in cls.get_superusers(bot)

    @classmethod
    async def is_group_admin(
        cls, bot: Bot, session: Uninfo, user_id: str | None = None
    ) -> bool:
        """判断用户是否为当前群聊或频道的群主、管理员

        优先使用事件携带的群角色，缺失时查询一次群成员信息并缓存

        参数:
            bot: Bot
            session: Uninfo
            user_id: 用户id，默认为事件发送者

        返回:
            bool: 是否为群主或管理员
        """
        scene = session.group or session.guild
        if scene is None:
            return False
        user_id = str(user_id or session.user.id)
        role = None
        if session.member and session.member.id == user_id:
            role = session.member.role
        if role is not None:
            return role.level >= _ADMIN_ROLE_LEVEL
        key = f"{bot.self_id}:{scene.type}:{scene.id}:{user_id}"
        if key in _GROUP_ROLE_CACHE:
            return _GROUP_ROLE_CACHE[key] >= _ADMIN_ROLE_LEVEL
        level = 0
        try:
            if interface := get_interface(bot):
                member = await interface.get_member(scene.type, scene.id, user_id)
                if member and member.role:
                    level = member.role.level
        except Exception as e:
            logger.debug(
                "查询群成员角色失败", "权限", session=user_id, group_id=scene.id, e=e
            )
            _GROUP_ROLE_CACHE.set(key, level, expire=_ROLE_RETRY_EXPIRE)
            return False
        _GROUP_ROLE_CACHE[key] = level
        return level >= _ADMIN_ROLE_LEVEL

    @classmethod
    def qq_client_bots(cls) -> list[Bot]:
        """在线的 OneBot 协议端 Bot，按账号排序

        返回:
            list[Bot]: Bot 列表
        """
        return sorted(
            (
                bot
                for bot in nonebot.get_bots().values()
                if cls.get_platform_scope(bot) == "qq_client"
            ),
            key=lambda bot: str(bot.self_id),
        )

    @classmethod
    def _resolve_unique_qq_client_bot(cls, log_cmd: str | None = None) -> Bot | None:
        bots = list(nonebot.get_bots().values())
        if not bots:
            logger.warning("当前没有可用的 OneBot 协议端 Bot，已跳过。", log_cmd)
            return None
        qq_client_bots = [
            bot for bot in bots if cls.get_platform_scope(bot) == "qq_client"
        ]
        if len(qq_client_bots) == 1:
            bot = qq_client_bots[0]
            if len(bots) > 1:
                logger.warning(
                    f"多 Bot 在线且未指定 Bot，自动选择 OneBot {bot.self_id}。",
                    log_cmd,
                )
            return bot
        if not qq_client_bots:
            logger.warning("未找到 OneBot 协议端 Bot，已跳过。", log_cmd)
        else:
            logger.warning(
                "存在多个 OneBot 协议端 Bot，无法安全选择，已跳过。", log_cmd
            )
        return None

    @classmethod
    def resolve_bot(
        cls,
        bot_id: str | None = None,
        platform_scope: str | None = None,
        log_cmd: str | None = None,
    ) -> Bot | None:
        """Resolve a bot without randomly selecting the QQ official adapter.

        Background jobs that require OneBot APIs should pass
        ``platform_scope="qq_client"``.  When no scope is supplied and multiple
        bots are online, a single OneBot client is preferred; otherwise the
        ambiguous selection is skipped.
        """
        if bot_id:
            try:
                bot = nonebot.get_bot(bot_id)
            except KeyError:
                logger.warning(f"Bot:{bot_id} 对象未连接或不存在", log_cmd)
                return None
            if platform_scope and cls.get_platform_scope(bot) != platform_scope:
                logger.warning(f"Bot:{bot_id} 平台作用域不匹配，已跳过。", log_cmd)
                return None
            return bot

        bots = list(nonebot.get_bots().values())
        if platform_scope:
            bots = [
                bot for bot in bots if cls.get_platform_scope(bot) == platform_scope
            ]
        if not bots:
            logger.warning("当前没有匹配的 Bot，已跳过。", log_cmd)
            return None
        if len(bots) == 1:
            return bots[0]
        if platform_scope is None:
            return cls._resolve_unique_qq_client_bot(log_cmd)
        logger.warning("存在多个匹配的 Bot，无法安全选择，已跳过。", log_cmd)
        return None

    @classmethod
    def is_qbot(cls, session: Uninfo | Bot) -> bool:
        """判断bot是否为qq官bot

        参数:
            session: Uninfo

        返回:
            bool: 是否为官bot
        """
        if isinstance(session, Bot):
            if cls.get_platform_scope(session) == "qq_api":
                return True
            return bool(BotConfig.get_qbot_uid(session.self_id))
        if cls.get_platform_scope(session) == "qq_api":
            return True
        if BotConfig.get_qbot_uid(session.self_id):
            return True
        return getattr(session, "scope", None) == SupportScope.qq_api

    @classmethod
    async def ban_user(cls, bot: Bot, user_id: str, group_id: str, duration: int):
        """禁言

        参数:
            bot: Bot
            user_id: 用户id
            group_id: 群组id
            duration: 禁言时长(分钟)
        """
        if cls.get_platform_scope(bot) == "qq_client":
            await bot.set_group_ban(
                group_id=int(group_id),
                user_id=int(user_id),
                duration=duration * 60,
            )

    @classmethod
    def get_superusers(cls, bot: Bot) -> list[str]:
        """获取该 Bot 所在平台的超级用户id

        参数:
            bot: Bot

        返回:
            list[str]: PLATFORM_SUPERUSERS 对应平台的 id，加上 SUPERUSERS 中
                `适配器:id` 形式的 id；QQ 平台另含 SUPERUSERS 中的裸 id
        """
        platform = cls.get_platform(bot)
        adapter = _adapter_name(bot).split(maxsplit=1)
        prefix = f"{adapter[0]}:" if adapter else None
        result = []
        for item in BotConfig.get_superuser(platform):
            if (user_id := str(item).strip()) and user_id not in result:
                result.append(user_id)
        for item in bot.config.superusers:
            item = str(item).strip()
            if ":" not in item:
                if platform != "qq":
                    continue
                user_id = item
            elif prefix and item.startswith(prefix):
                user_id = item[len(prefix) :]
            else:
                continue
            if user_id and user_id not in result:
                result.append(user_id)
        return result

    @classmethod
    async def send_superuser(
        cls,
        bot: Bot | None,
        message: UniMessage | str,
        superuser_id: str | None = None,
    ) -> list[tuple[str, Receipt]]:
        """发送消息给超级用户

        参数:
            bot: Bot，未传入或为 QQ 官方 Bot 时使用唯一在线的 OneBot
            message: 消息
            superuser_id: 指定超级用户id.

        异常:
            NotFindSuperuser: 未找到超级用户id

        返回:
            Receipt | None: Receipt
        """
        if not bot or cls.get_platform_scope(bot) == "qq_api":
            # QQ 官方 Bot 不能主动私聊，改由 OneBot 发送
            bot = cls._resolve_unique_qq_client_bot("PlatformUtils:send_superuser")
            if bot is None:
                return []
        superuser_ids = []
        if superuser_id:
            superuser_ids.append(superuser_id)
        elif not (superuser_ids := cls.get_superusers(bot)):
            raise NotFindSuperuser()
        if isinstance(message, str):
            message = MessageUtils.build_message(message)
        result = []
        for superuser_id in superuser_ids:
            try:
                result.append(
                    (
                        superuser_id,
                        await cls.send_message(bot, superuser_id, None, message),
                    )
                )
            except Exception as e:
                logger.error(
                    "发送消息给超级用户失败",
                    "PlatformUtils:send_superuser",
                    target=superuser_id,
                    e=e,
                )
        return result

    @classmethod
    async def iter_group_members(
        cls,
        bot: Bot,
        group_id: str,
        *,
        scene: SceneType = SceneType.GROUP,
        parent_scene_id: str | None = None,
    ) -> AsyncIterator[UserData]:
        """Enumerate only on demand; errors never masquerade as an empty roster."""
        from nonebot.adapters.onebot.v11 import Bot as OneBotBot

        try:
            from nonebot.adapters.qq import Bot as QQBot
        except ModuleNotFoundError:
            QQBot = None

        if QQBot is not None and isinstance(bot, QQBot):
            if scene == SceneType.GROUP:
                cursor = None
                seen = set()
                while True:
                    page = await bot.post_group_members(
                        group_id=str(group_id), limit=100, start_index=cursor
                    )
                    next_cursor = page.next_index
                    if next_cursor is not None and (
                        next_cursor == cursor or next_cursor in seen
                    ):
                        raise RuntimeError("qq member pagination cursor repeated")
                    for member in page.members:
                        yield UserData(
                            name="",
                            user_id=member.member_openid,
                            group_id=group_id,
                            join_time=int(member.join_timestamp.timestamp()),
                        )
                    if next_cursor is None:
                        return
                    seen.add(next_cursor)
                    cursor = next_cursor
            elif scene == SceneType.GUILD or scene >= SceneType.CHANNEL_TEXT:
                guild_id = group_id if scene == SceneType.GUILD else parent_scene_id
                if not guild_id:
                    raise ValueError("channel member query requires parent_scene_id")
                after = None
                seen = set()
                while True:
                    members = await bot.get_members(
                        guild_id=guild_id, after=after, limit=100
                    )
                    if not members:
                        return
                    cursor = members[-1].user.id
                    if not cursor or cursor == after or cursor in seen:
                        raise RuntimeError("qq guild member pagination cursor repeated")
                    for member in members:
                        yield UserData(
                            name=member.user.username or "",
                            card=member.nick,
                            user_id=member.user.id,
                            group_id=guild_id,
                            channel_id=group_id if parent_scene_id else None,
                            avatar_url=member.user.avatar,
                            join_time=int(member.joined_at.timestamp())
                            if member.joined_at
                            else None,
                        )
                    seen.add(cursor)
                    after = cursor
            else:
                raise ValueError("private scenes do not have a member roster")
            return
        if isinstance(bot, OneBotBot) and scene == SceneType.GROUP:
            for member in await bot.get_group_member_list(group_id=int(group_id)):
                yield UserData(
                    name=member.get("nickname") or "",
                    card=member.get("card"),
                    user_id=str(member["user_id"]),
                    group_id=group_id,
                    role=member.get("role"),
                    join_time=member.get("join_time"),
                )
            return
        interface = get_interface(bot)
        if interface is None:
            raise NotImplementedError("member query unsupported by adapter")
        # The Interface list wrapper has no parent_scene_id argument and hides
        # NotImplementedError. Use the registered iterator to preserve failures.
        async for member in interface.fetcher.query_members(
            bot, scene, parent_scene_id or group_id
        ):
            yield UserData(
                name=member.user.name or "",
                card=member.nick,
                user_id=member.user.id,
                group_id=parent_scene_id or group_id,
                channel_id=group_id if parent_scene_id else None,
                role=member.role.id if member.role else None,
                avatar_url=member.user.avatar,
                join_time=int(member.joined_at.timestamp())
                if member.joined_at
                else None,
            )

    @classmethod
    async def get_group_member_list(
        cls,
        bot: Bot,
        group_id: str,
        *,
        scene: SceneType = SceneType.GROUP,
        parent_scene_id: str | None = None,
    ) -> list[UserData]:
        return [
            member
            async for member in cls.iter_group_members(
                bot, group_id, scene=scene, parent_scene_id=parent_scene_id
            )
        ]

    @classmethod
    async def get_user(
        cls,
        bot: Bot,
        user_id: str,
        group_id: str | None = None,
        channel_id: str | None = None,
    ) -> UserData | None:
        """获取用户信息

        参数:
            bot: Bot
            user_id: 用户id
            group_id: 群组id.
            channel_id: 频道id.

        返回:
            UserData | None: 用户数据
        """
        if not (interface := get_interface(bot)):
            return None
        member = None
        user = None
        if channel_id:
            member = await interface.get_member(
                SceneType.CHANNEL_TEXT, channel_id, user_id
            )
            if member:
                user = member.user
        elif group_id:
            member = await interface.get_member(SceneType.GROUP, group_id, user_id)
            if member:
                user = member.user
        else:
            user = await interface.get_user(user_id)
        if not user:
            return None
        return (
            UserData(
                name=user.name or "",
                card=member.nick,
                user_id=user.id,
                group_id=group_id,
                channel_id=channel_id,
                role=member.role.id if member.role else None,
                join_time=(
                    int(member.joined_at.timestamp()) if member.joined_at else None
                ),
            )
            if member
            else UserData(
                name=user.name or "",
                user_id=user.id,
                group_id=group_id,
                channel_id=channel_id,
            )
        )

    @classmethod
    async def get_user_avatar(
        cls, user_id: str, platform: str, appid: str | None = None
    ) -> bytes | None:
        """快捷获取用户头像

        参数:
            user_id: 用户id
            platform: 平台
        """
        url = None
        if platform == "qq":
            if user_id.isdigit():
                url = f"http://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            else:
                url = f"https://q.qlogo.cn/qqapp/{appid}/{user_id}/640"
        if not url:
            return None
        try:
            return await AsyncHttpx.get_content(url)
        except (
            httpx.HTTPError,
            AllURIsFailedError,
            TimeoutError,
            asyncio.TimeoutError,
        ):
            logger.debug("头像暂不可用，使用展示占位", "Util", target=user_id)
            return None

    @classmethod
    def get_user_avatar_url(
        cls, user_id: str, platform: str, appid: str | None = None
    ) -> str | None:
        """快捷获取用户头像url

        参数:
            user_id: 用户id
            platform: 平台
        """
        if platform != "qq":
            return None
        if user_id.isdigit():
            return f"http://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
        else:
            return f"https://q.qlogo.cn/qqapp/{appid}/{user_id}/640"

    @classmethod
    async def get_group_avatar(cls, gid: str, platform: str) -> bytes | None:
        """快捷获取用群头像

        参数:
            gid: 群组id
            platform: 平台
        """
        if platform == "qq":
            url = f"http://p.qlogo.cn/gh/{gid}/{gid}/640/"
            async with ManagedAsyncClient() as client:
                for _ in range(3):
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        return response.content
                    except (httpx.HTTPError, TimeoutError, asyncio.TimeoutError):
                        logger.debug(
                            "群头像暂不可用", "Util", target=gid, platform=platform
                        )
        return None

    @classmethod
    async def send_message(
        cls,
        bot: Bot,
        user_id: str | None,
        group_id: str | None,
        message: str | UniMessage,
        official_context: object | None = None,
    ) -> Receipt | None:
        """发送消息

        QQ 官方 Bot 指定的群与事件所在群不同或不在事件中时向该群发送主动消息，
        未指定群时回复到事件所在会话，不在事件中时向用户发送主动单聊消息。

        参数:
            bot: Bot
            user_id: 用户id
            group_id: 群组id或频道id
            message: 消息文本
            official_context: QQ 官方事件上下文，默认取当前事件

        返回:
            Receipt | None: 是否发送成功
        """
        send_message = (
            MessageUtils.build_message(message) if isinstance(message, str) else message
        )
        if cls.get_platform_scope(bot) == "qq_api":
            from zhenxun.adapters.qq_official.bot import (
                ZhenxunQQBot,
                send_official_message,
            )

            return await send_official_message(
                cast(ZhenxunQQBot, bot),
                send_message,
                group_id,
                official_context,
                user_id,
            )
        if target := cls.get_target(user_id=user_id, group_id=group_id, bot=bot):
            return await send_message.send(target=target, bot=bot)
        return None

    @classmethod
    async def update_group(cls, bot: Bot) -> int:
        """更新群组信息，只写入群名与人数有变化的群，群缓存在提交后从数据库刷新

        参数:
            bot: Bot

        返回:
            int: 新增个数
        """
        from tortoise.transactions import in_transaction

        create_list = []
        update_list: list[tuple[int, dict]] = []
        group_list, platform = await cls.get_group_list(bot)
        if group_list:
            db_group = {
                (group.group_id, group.channel_id): group
                for group in await GroupConsole.all()
            }
            for group in group_list:
                group.platform = platform
                current = db_group.get((group.group_id, group.channel_id))
                if current is None:
                    create_list.append(group)
                    logger.debug(
                        "群聊信息更新成功",
                        "更新群信息",
                        target=f"{group.group_id}:{group.channel_id}",
                    )
                    continue
                values = {
                    "group_name": group.group_name,
                    "max_member_count": group.max_member_count,
                    "member_count": group.member_count,
                }
                if any(getattr(current, k) != v for k, v in values.items()):
                    update_list.append((current.id, values))
        if create_list or update_list:
            async with in_transaction() as connection:
                if create_list:
                    await GroupConsole.bulk_create(create_list, 10, using_db=connection)
                for group_id, values in update_list:
                    await (
                        GroupConsole.filter(id=group_id)
                        .using_db(connection)
                        .update(**values)
                    )
        return len(create_list)

    @classmethod
    def get_platform(cls, t: Bot | Uninfo) -> str:
        """获取平台

        参数:
            bot: Bot

        返回:
            str | None: 平台
        """
        if isinstance(t, Bot):
            if interface := get_interface(t):
                info = interface.basic_info()
                platform = info["scope"].lower()
                return "qq" if platform.startswith("qq") else platform
            adapter_name = _adapter_name(t)
            if "onebot" in adapter_name or adapter_name == "qq":
                return "qq"
            return adapter_name or "unknown"
        else:
            platform = t.basic["scope"].lower()
            return "qq" if platform.startswith("qq") else platform

    @classmethod
    def get_platform_scope(cls, t: Bot | Uninfo | object) -> str:
        """Resolve a verified protocol domain, never infer QQ from a name fragment."""
        from nonebot.adapters.onebot.v11 import Bot as OneBotBot

        try:
            from nonebot.adapters.qq import Bot as QQBot
        except ModuleNotFoundError:
            QQBot = None

        if QQBot is not None and isinstance(t, QQBot):
            return "qq_api"
        if isinstance(t, OneBotBot):
            return "qq_client"
        if isinstance(t, Bot):
            if interface := get_interface(t):
                scope = _scope_name(interface.basic_info().get("scope"))
                if scope:
                    return _qualified_scope(scope, _adapter_name(t))
            return "unknown"
        scope = _scope_name(getattr(t, "scope", "") or "")
        adapter = getattr(t, "adapter", None)
        basic = getattr(t, "basic", None)
        if not scope and isinstance(basic, dict):
            scope = _scope_name(basic.get("scope"))
            adapter = basic.get("adapter")
        if scope and adapter:
            return _qualified_scope(scope, adapter)
        return scope or "unknown"

    @classmethod
    def is_forward_merge_supported(cls, t: Bot | Uninfo) -> bool:
        """是否支持转发消息

        参数:
            t: bot | Uninfo

        返回:
            bool: 是否支持转发消息
        """
        if not isinstance(t, Bot):
            return t.basic["scope"] == SupportScope.qq_client
        if interface := get_interface(t):
            info = interface.basic_info()
            return info["scope"] == SupportScope.qq_client
        return False

    @classmethod
    async def get_group_list(
        cls, bot: Bot, only_group: bool = False
    ) -> tuple[list[GroupConsole], str]:
        """获取群组列表

        参数:
            bot: Bot
            only_group: 是否只获取群组（不获取channel）

        返回:
            tuple[list[GroupConsole], str]: 群组列表, 平台
        """
        if not (interface := get_interface(bot)):
            return [], ""
        platform = cls.get_platform(bot)
        result_list = []
        scenes = await interface.get_scenes(SceneType.GROUP)
        for scene in scenes:
            group_id = scene.id
            result_list.append(
                GroupConsole(
                    group_id=scene.id,
                    group_name=scene.name,
                )
            )
            if not only_group and platform != "qq":
                if channel_list := await interface.get_scenes(parent_scene_id=group_id):
                    result_list.extend(
                        GroupConsole(
                            group_id=scene.id,
                            group_name=channel.name,
                            channel_id=channel.id,
                        )
                        for channel in channel_list
                    )
        return result_list, platform

    @classmethod
    async def update_friend(cls, bot: Bot) -> int:
        """更新好友信息

        参数:
            bot: Bot

        返回:
            int: 更新个数
        """
        if cls.get_platform_scope(bot) == "qq_api":
            logger.warning("QQ 官方适配器不支持旧好友同步，已跳过。", "更新好友信息")
            return 0
        create_list = []
        friend_list, platform = await cls.get_friend_list(bot)
        if friend_list:
            user_id_list = await FriendUser.all().values_list("user_id", flat=True)
            for friend in friend_list:
                friend.platform = platform
                if friend.user_id not in user_id_list:
                    create_list.append(friend)
        if create_list:
            await FriendUser.bulk_create(create_list, 10)
        return len(create_list)

    @classmethod
    async def get_friend_list(cls, bot: Bot) -> tuple[list[FriendUser], str]:
        """获取好友列表

        参数:
            bot: Bot

        返回:
            list[FriendUser]: 好友列表
        """
        if cls.get_platform_scope(bot) == "qq_api":
            logger.warning(
                "QQ 官方适配器不支持旧好友列表查询，已返回空列表。", "好友列表"
            )
            return [], cls.get_platform(bot)
        if interface := get_interface(bot):
            user_list = await interface.get_users()
            return [
                FriendUser(user_id=u.id, user_name=u.name) for u in user_list
            ], cls.get_platform(bot)
        return [], ""

    @classmethod
    def get_target(
        cls,
        *,
        user_id: str | None = None,
        group_id: str | None = None,
        channel_id: str | None = None,
        bot: Bot | None = None,
    ) -> Target | None:
        """获取发送Target

        参数:
            user_id: 用户id
            group_id: 频道id或群组id
            channel_id: 频道id
            bot: 发送所用的 Bot，传入时把持久化的 id 还原为该 Bot 平台中的 id

        返回:
            Target | None: 对应平台Target，id 不属于该 Bot 的平台时为 None
        """
        if bot is not None:
            scope = cls.get_platform_scope(bot)
            if group_id:
                group_id = native_id(
                    group_id, scope, group=True, bot_id=str(bot.self_id)
                )
                if group_id is None:
                    return None
            elif user_id:
                user_id = native_id(user_id, scope)
        target = None
        if group_id and channel_id:
            target = Target(channel_id, parent_id=group_id, channel=True)
        elif group_id:
            target = Target(group_id)
        elif user_id:
            target = Target(user_id, private=True)
        return target


class BroadcastEngine:
    def __init__(
        self,
        message: str | UniMessage,
        bot: Bot | list[Bot] | None = None,
        bot_id: str | set[str] | None = None,
        ignore_group: list[str] | None = None,
        check_func: Callable[[Bot, str], Awaitable] | None = None,
        log_cmd: str | None = None,
        platform: str | None = None,
        *,
        report: dict | None = None,
    ):
        """广播引擎

        参数:
        message: 广播消息内容
        bot: 指定bot对象.
        bot_id: 指定bot id.
        ignore_group: 忽略群聊列表.
        check_func: 发送前对群聊检测方法，判断是否发送.
        log_cmd: 日志标记.
        platform: 指定平台.
        report: 接收广播计数与原因摘要.

        异常:
            ValueError: 没有可用的Bot对象
        """
        self.message = MessageUtils.build_message(message)
        self.ignore_group = list(ignore_group or [])
        self.sent: set[str] = set()
        self.check_func = check_func
        self.log_cmd = log_cmd
        self.platform = platform
        self.bot_list = []
        self.count = 0
        self.report = report if report is not None else {}
        self.report.update(
            examined=0, skipped=0, failed=0, processed=0, reason_counts={}
        )
        if bot:
            self.bot_list = [bot] if isinstance(bot, Bot) else bot
        if isinstance(bot_id, str):
            bot_id = {bot_id}
        if bot_id:
            for i in bot_id:
                try:
                    self.bot_list.append(nonebot.get_bot(i))
                except KeyError:
                    logger.warning(f"Bot:{i} 对象未连接或不存在", log_cmd)
        if not self.bot_list:
            # 未指定时由全部 OneBot 协议端发送，同一群只发送一次
            self.bot_list = PlatformUtils.qq_client_bots()
            if not self.bot_list:
                logger.warning("未找到 OneBot 协议端 Bot，已跳过。", log_cmd)

    async def call_check(self, bot: Bot, group_id: str) -> bool:
        """运行发送检测函数

        参数:
            bot: Bot
            group_id: 群组id

        返回:
            bool: 是否发送
        """
        if not self.check_func:
            return True
        if is_coroutine_callable(self.check_func):
            is_run = await self.check_func(bot, group_id)
        else:
            is_run = self.check_func(bot, group_id)
        return cast(bool, is_run)

    def _skip(self, reason: str) -> None:
        """记录未发送群聊的原因。"""
        self.report["skipped"] += 1
        reasons = self.report["reason_counts"]
        reasons[reason] = reasons.get(reason, 0) + 1

    async def __send_message(self, bot: Bot, group: GroupConsole) -> bool:
        """群组发送消息

        参数:
            bot: Bot
            group: GroupConsole

        返回:
            bool: 是否取得成功发送回执
        """
        key = f"{group.group_id}:{group.channel_id}"
        if not await self.call_check(bot, group.group_id):
            logger.debug(
                "广播方法检测运行方法为 False, 已跳过该群组...",
                self.log_cmd,
                group_id=group.group_id,
            )
            self._skip("policy_rejected")
            return False
        if PlatformUtils.get_platform_scope(bot) == "qq_api":
            sending = PlatformUtils.send_message(
                bot, None, group.group_id, self.message
            )
        elif target := PlatformUtils.get_target(
            group_id=group.group_id, channel_id=group.channel_id, bot=bot
        ):
            sending = MessageUtils.build_message(self.message).send(target, bot)
        else:
            logger.warning("广播消息获取Target失败...", self.log_cmd, target=key)
            self._skip("no_target")
            return False
        self.sent.add(key)
        if await sending is None:
            self._skip("send_unconfirmed")
            return False
        logger.debug("广播消息发送成功...", self.log_cmd, target=key)
        return True

    @classmethod
    async def _group_list(cls, bot: Bot) -> list[GroupConsole]:
        """获取广播的群组

        参数:
            bot: Bot

        返回:
            list[GroupConsole]: 群组列表，QQ 官方 Bot 为已开启主动消息的群
        """
        if PlatformUtils.get_platform_scope(bot) == "qq_api":
            from zhenxun.adapters.qq_official.proactive import proactive_groups

            return [
                GroupConsole(group_id=group_id)
                for group_id in await proactive_groups(str(bot.self_id))
            ]
        group_list, _ = await PlatformUtils.get_group_list(bot)
        return group_list

    async def broadcast(self) -> int:
        """广播消息

        返回:
            int: 成功发送次数
        """
        reasons = self.report["reason_counts"]
        if not self.bot_list:
            reasons["no_bot"] = 1
        for bot in self.bot_list:
            if self.platform and self.platform != PlatformUtils.get_platform(bot):
                reasons["platform_mismatch"] = reasons.get("platform_mismatch", 0) + 1
                continue
            group_list = await self._group_list(bot)
            if not group_list:
                reasons["empty_groups"] = reasons.get("empty_groups", 0) + 1
                continue
            for group in group_list:
                self.report["examined"] += 1
                if (
                    group.group_id in self.ignore_group
                    or group.channel_id in self.ignore_group
                ):
                    self._skip("ignored_group")
                    continue
                if f"{group.group_id}:{group.channel_id}" in self.sent:
                    self._skip("duplicate")
                    continue
                try:
                    sent = await self.__send_message(bot, group)
                except Exception as e:
                    self.report["failed"] += 1
                    reasons["send_failed"] = reasons.get("send_failed", 0) + 1
                    logger.warning(
                        "广播消息发送失败", self.log_cmd, target=group.group_id, e=e
                    )
                    continue
                if sent:
                    self.count += 1
                    self.report["processed"] = self.count
                    await asyncio.sleep(random.randint(1, 3))
        return self.count


async def broadcast_group(
    message: str | UniMessage,
    bot: Bot | list[Bot] | None = None,
    bot_id: str | set[str] | None = None,
    ignore_group: list[str] | None = None,
    check_func: Callable[[Bot, str], Awaitable] | None = None,
    log_cmd: str | None = None,
    platform: str | None = None,
    *,
    report: dict | None = None,
) -> int:
    """获取所有Bot或指定Bot对象广播群聊

    参数:
        message: 广播消息内容
        bot: 指定bot对象.
        bot_id: 指定bot id.
        ignore_group: 忽略群聊列表.
        check_func: 发送前对群聊检测方法，判断是否发送.
        log_cmd: 日志标记.
        platform: 指定平台
        report: 接收群聊检查、跳过、失败与成功发送数及原因摘要

    返回:
        int: 成功发送次数
    """
    if not message.strip():
        raise ValueError("群聊广播消息不能为空...")
    return await BroadcastEngine(
        message=message,
        bot=bot,
        bot_id=bot_id,
        ignore_group=ignore_group,
        check_func=check_func,
        log_cmd=log_cmd,
        platform=platform,
        report=report,
    ).broadcast()
