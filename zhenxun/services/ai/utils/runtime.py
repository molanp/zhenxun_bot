import asyncio
import inspect
import time
from typing import Any


class ContextUtils:
    """
    从底层依赖容器 (deps) 中提取运行环境信息的纯静态工具类。
    """

    @staticmethod
    def extract_user_id(deps: Any) -> str | None:
        """从依赖容器中提取当前用户的 ID"""
        if not deps:
            return None
        if hasattr(deps, "user_id") and getattr(deps, "user_id") is not None:
            return str(getattr(deps, "user_id"))
        event = getattr(deps, "event", None)
        if event:
            try:
                return str(event.get_user_id())
            except Exception:
                return (
                    str(getattr(event, "user_id", ""))
                    or str(getattr(event, "sender_id", ""))
                    or None
                )
        return None

    @staticmethod
    def extract_group_id(deps: Any) -> str | None:
        """从依赖容器中提取当前群聊或频道的 ID

        依次使用 group_id、QQ 官方群聊的 group_openid、QQ 频道消息的
        channel_id（频道私信视为私聊）。

        参数:
            deps: 依赖容器

        返回:
            str | None: 群聊或频道 ID，私聊为 None
        """
        if not deps:
            return None
        if getattr(deps, "group_id", None) is not None:
            return str(getattr(deps, "group_id"))
        event = getattr(deps, "event", None)
        if event is None:
            return None
        for key in ("group_id", "group_openid"):
            if value := getattr(event, key, None):
                return str(value)
        if channel_id := getattr(event, "channel_id", None):
            from nonebot.adapters.qq import DirectMessageCreateEvent

            if not isinstance(event, DirectMessageCreateEvent):
                return str(channel_id)
        return None

    @staticmethod
    def extract_platform(deps: Any) -> str:
        """从依赖容器的 Bot 实例中提取当前聊天平台名称"""
        if not deps:
            return "unknown"
        if hasattr(deps, "platform") and getattr(deps, "platform") is not None:
            return str(getattr(deps, "platform"))
        bot = getattr(deps, "bot", None)
        if bot:
            from zhenxun.utils.platform import PlatformUtils

            return PlatformUtils.get_platform(bot)
        return "unknown"

    @staticmethod
    def default_session_id(deps: Any) -> str:
        """按平台、Bot、群聊与用户生成默认会话 ID

        参数:
            deps: 依赖容器

        返回:
            str: 形如“平台:Bot:群号或 private:用户”的会话 ID
        """
        bot = getattr(deps, "bot", None)
        return ":".join(
            [
                ContextUtils.extract_platform(deps),
                str(getattr(bot, "self_id", "") or "unknown"),
                ContextUtils.extract_group_id(deps) or "private",
                ContextUtils.extract_user_id(deps) or "unknown",
            ]
        )


class PermissionUtils:
    """运行时权限校验通用工具类"""

    @staticmethod
    def bind_caller(context: Any, source: str, management_check=None) -> None:
        """从受信入口绑定身份，子运行继承身份而不重新解释模型输入。"""
        from zhenxun.services.ai.run.context import CallerIdentity
        from zhenxun.utils.platform import PlatformUtils

        if context.run.caller is not None:
            if context.run.caller.source != source:
                raise ValueError("不能改变运行来源")
            return
        bot = context.get_bot()
        context.run.caller = CallerIdentity(
            source=source,
            platform=PlatformUtils.get_platform_scope(bot) if bot else "webui",
            bot_id=str(bot.self_id) if bot else "",
            user_id=context.get_user_id(),
            group_id=context.get_group_id(),
        )
        context.run.management_check = management_check

    @staticmethod
    def ensure_current(context: Any) -> None:
        """在执行与投递安全点检查取消、截止时间和根会话修订。"""
        if token := context.run.cancellation_token:
            token.raise_if_cancelled()
        if context.run.connection_id and (bot := context.get_bot()):
            if vars(bot).get("_zhenxun_connection_id") != context.run.connection_id:
                raise asyncio.CancelledError("来源连接已变更")
        if deadline := context.run.budget.get("deadline"):
            if time.monotonic() >= deadline:
                raise asyncio.TimeoutError("本轮执行超时")
        if context.run.root_revision is not None:
            from zhenxun.services.ai.run.session import session_manager

            if context.run.root_revision != session_manager.revision(
                context.run.root_session_id or ""
            ):
                raise asyncio.CancelledError("会话已重置")

    @staticmethod
    def _deny(context: Any, reason: str) -> bool:
        context.state["_permission_reason"] = reason
        return False

    @staticmethod
    async def check_superuser(context: Any) -> bool:
        """按可信来源和当前授权校验超管，不复用旧事件角色。"""
        caller = context.run.caller
        if caller is None or caller.source not in {"qq", "task", "webui"}:
            return PermissionUtils._deny(context, "当前来源没有管理授权")
        if caller.source == "webui" or (
            caller.source == "task" and context.run.management_check is not None
        ):
            check = context.run.management_check
            try:
                allowed = check() if check else False
                if inspect.isawaitable(allowed):
                    allowed = await allowed
                if allowed:
                    return True
            except Exception:
                return PermissionUtils._deny(context, "管理会话授权暂不可确认")
            return PermissionUtils._deny(context, "管理会话已失效或未授权")
        bot = context.get_bot()
        try:
            from zhenxun.utils.platform import PlatformUtils

            if (
                bot
                and str(bot.self_id) == caller.bot_id
                and PlatformUtils.get_platform_scope(bot) == caller.platform
                and PlatformUtils.is_superuser(bot, caller.user_id)
            ):
                return True
        except Exception:
            return PermissionUtils._deny(context, "权限读取失败")
        return PermissionUtils._deny(context, "当前调用者不是超级用户")

    @staticmethod
    async def check_admin_level(context: Any, min_level: int) -> bool:
        """校验当前用户的管理权限

        参数:
            context: 运行上下文
            min_level: 权限等级，大于 0 时要求超级用户或当前群的群主、管理员

        返回:
            bool: 是否满足
        """
        if min_level <= 0 or await PermissionUtils.check_superuser(context):
            return True
        bot = context.get_bot()
        caller = context.run.caller
        if (
            not bot
            or caller is None
            or caller.source not in {"qq", "task"}
            or not caller.group_id
            or not caller.user_id
        ):
            return False
        from nonebot_plugin_uninfo import SceneType, get_interface, get_session

        from zhenxun.utils.platform import PlatformUtils

        if (
            str(bot.self_id) != caller.bot_id
            or PlatformUtils.get_platform_scope(bot) != caller.platform
        ):
            return PermissionUtils._deny(context, "调用者与当前连接不匹配")
        try:
            scene_type, scene_id = SceneType.GROUP, caller.group_id
            if event := context.get_event():
                session = await get_session(bot, event)
                scene = (session.group or session.guild) if session else None
                if scene is None or str(scene.id) != caller.group_id:
                    return PermissionUtils._deny(context, "群身份无法确认")
                scene_type, scene_id = scene.type, scene.id
            interface = get_interface(bot)
            member = (
                await interface.get_member(scene_type, scene_id, caller.user_id)
                if interface
                else None
            )
            if member and member.role and member.role.level >= 10:
                return True
        except Exception:
            return PermissionUtils._deny(context, "当前群管理权限暂不可确认")
        return PermissionUtils._deny(context, "当前调用者不是本群管理员")
