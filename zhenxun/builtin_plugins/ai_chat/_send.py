from collections import deque
import time
from typing import Any, ClassVar

from nonebot.adapters import Bot
from nonebot.exception import ActionFailed, ApiNotAvailable
from nonebot_plugin_alconna import At, Reply, Segment, Text, UniMessage

from zhenxun.configs.config import BotConfig, Config
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ResolvedToolPayload, ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.platform import PlatformUtils

from ._buffer import ChatBuffer, ChatLine
from ._data_source import MODULE
from ._reply import clean_markdown
from ._safety import Safety, prune
from ._tools import GroupHistory

SEND_MAX_MESSAGES = 5
"""send_message 单次最多条数"""
SEND_MAX_CHARS = 500
"""send_message 每条最多字数"""
_POKE_METHODS = {
    "group": [
        ("group_poke", ("group_id", "user_id")),
        ("send_poke", ("group_id", "user_id")),
        ("send_group_msg", ("group_id", "message")),
    ],
    "private": [
        ("friend_poke", ("user_id",)),
        ("send_poke", ("user_id",)),
        ("send_private_msg", ("user_id", "message")),
    ],
}
"""各协议端戳一戳接口，依次尝试"""


class Scene:
    """发送类工具的当前会话：群聊或私聊"""

    def __init__(self, bot: Bot, group_id: str | None, user_id: str | None):
        self.bot = bot
        self.group_id = group_id
        self.user_id = user_id

    @property
    def key(self) -> tuple[str, str]:
        """限速与接口记忆使用的会话键"""
        return (
            self.bot.self_id,
            f"g{self.group_id}" if self.group_id else f"p{self.user_id}",
        )

    @classmethod
    def of(cls, context: RunContext) -> "Scene | None":
        """从运行上下文取当前会话，不在会话中或是发往私聊的主动任务时返回 None

        参数:
            context: 运行上下文

        返回:
            Scene | None: 会话
        """
        bot, group_id = context.get_bot(), context.get_group_id()
        user_id = context.get_user_id()
        if bot is None or not (group_id or user_id):
            return None
        if not group_id and getattr(context.deps, "schedule_id", None) is not None:
            return None
        return cls(bot, group_id, None if group_id else user_id)

    async def is_member(self, user_id: str) -> bool:
        """判断用户是否为当前群成员（仅 OneBot V11 群聊）

        参数:
            user_id: QQ 号

        返回:
            bool: 是否为本群成员
        """
        if not self.group_id or not user_id.isdigit() or not self.group_id.isdigit():
            return False
        try:
            await self.bot.call_api(
                "get_group_member_info",
                group_id=int(self.group_id),
                user_id=int(user_id),
            )
        except Exception:
            return False
        return True

    def note(self, text: str) -> None:
        """在群聊缓冲中记一条 Bot 的动作

        参数:
            text: 动作描述
        """
        if self.group_id:
            ChatBuffer.record(
                self.bot.self_id,
                self.group_id,
                ChatLine(
                    None,
                    self.bot.self_id,
                    BotConfig.self_nickname or "Bot",
                    text,
                    time.time(),
                    is_bot=True,
                ),
            )


class SendQuota:
    """发送类工具（发消息、戳一戳、表情）在同一会话内共用的次数限制，
    也用于收藏表情等其他按会话限速的操作"""

    _times: ClassVar[dict[tuple[str, str], deque[float]]] = {}
    _swept: ClassVar[float] = 0
    """上次清理过期会话的时间"""

    @classmethod
    def take(
        cls,
        key: tuple[str, str],
        count: int = 1,
        per_minute: int | None = None,
        per_hour: int | None = None,
        unit: str = "次",
    ) -> str | None:
        """占用发送次数，超出每分钟或每小时上限时返回原因，原因中提示模型不要等待重试；
        每小时最多清理一次超过一小时没有发送的会话

        参数:
            key: 会话键
            count: 次数
            per_minute: 每分钟上限，为空使用发送工具配置，0 为不限
            per_hour: 每小时上限，为空使用发送工具配置，0 为不限
            unit: 原因中的计数单位

        返回:
            str | None: 超限原因，未超限为 None
        """
        if per_minute is None:
            per_minute = int(Config.get_config(MODULE, "SEND_PER_MINUTE") or 0)
        if per_hour is None:
            per_hour = int(Config.get_config(MODULE, "SEND_PER_HOUR") or 0)
        now = time.time()
        if now - cls._swept >= 3600:
            cls._swept = now
            prune(cls._times, now, 3600)
        times = cls._times.setdefault(key, deque())
        while times and now - times[0] >= 3600:
            times.popleft()
        if per_minute and sum(now - t < 60 for t in times) + count > per_minute:
            return (
                f"已达到每分钟 {per_minute} {unit}的上限，本轮不要等待或重试，"
                "直接回复，告诉对方稍后再来。"
            )
        if per_hour and len(times) + count > per_hour:
            return (
                f"已达到每小时 {per_hour} {unit}的上限，本轮不要等待或重试，"
                "直接回复，告诉对方稍后再来。"
            )
        times.extend([now] * count)
        return None

    @classmethod
    def refund(cls, key: tuple[str, str], count: int) -> None:
        """退还未实际发送的次数，退完后丢弃该会话的记录

        参数:
            key: 会话键
            count: 次数
        """
        times = cls._times.get(key)
        for _ in range(min(count, len(times or ()))):
            times.pop()  # type: ignore
        if times is not None and not times:
            del cls._times[key]


class QQSendToolkit(BaseToolkit):
    """AI 聊天的发送工具：运行中发言、戳一戳，只能发到当前会话"""

    _poke_methods: ClassVar[dict[tuple[str, str], str]] = {}

    default_instructions = (
        "## QQ 发送工具\n"
        "需要先说一句（比如“我查一下”）或想连发几条时用 `send_message`，"
        "第一条可以引用消息（reply_to）或 @ 一位本群成员（at）；"
        "`poke` 戳一戳某人。发送只会到当前会话，次数有限，不要刷屏。"
        "用 `send_message` 说过的话不要在最终回复里重复；要说的都已经用 "
        "`send_message` 发完时，最终回复只输出 [SILENT]。"
        "戳一戳、收藏表情等操作本身不算回复，做完仍要正常回一句。"
    )

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        if not Config.get_config(MODULE, "SEND_TOOLS", True):
            return ResolvedToolPayload()
        return await super().resolve(context)

    @tool(
        name="send_message",
        description=(
            "立刻向当前会话发送 1 到 5 条文字消息（每条最多 500 字），"
            "按长度间隔发送。reply_to：第一条引用的消息 id（仅群聊）；"
            "at：第一条 @ 的本群成员 QQ 号。"
        ),
    )
    async def send_message(
        self,
        context: RunContext,
        messages: list[str],
        reply_to: str = "",
        at: str = "",
    ) -> ToolResult:
        scene = Scene.of(context)
        if scene is None:
            return ToolResult(output="当前不在会话中。").as_error()
        texts = [clean_markdown(m) for m in messages if m and m.strip()]
        if not texts or len(texts) > SEND_MAX_MESSAGES:
            return ToolResult(output="一次发送 1 到 5 条消息。").as_error()
        if any(len(m) > SEND_MAX_CHARS for m in texts):
            return ToolResult(output="每条消息最多 500 字。").as_error()
        checked = [Safety.filter_output(m) for m in texts]
        if any(blocked for _, blocked in checked):
            return ToolResult(
                output="消息包含安全词或疑似密钥，未发送，不要再尝试发送这些内容。"
            ).as_error(False)
        texts = [m for m, _ in checked]
        reply_to, at = reply_to.strip().lstrip("#"), at.strip()
        if at and not (at.isdigit() and await scene.is_member(at)):
            return ToolResult(output="只能 @ 本群成员的 QQ 号。").as_error()
        if reply_to and (
            not scene.group_id
            or await GroupHistory.message(scene.bot, scene.group_id, reply_to) is None
        ):
            return ToolResult(output="只能引用当前群里的消息。").as_error()
        if reason := SendQuota.take(scene.key, len(texts)):
            return ToolResult(output=reason).as_error(False)
        from ._reply import DeliveryResult, _deliver, delivery_target

        async def send(message, first):
            segments: list[Segment] = []
            if first and reply_to:
                segments.append(Reply(reply_to))
            if first and at:
                segments += [At("user", at), Text(" ")]
            return await PlatformUtils.send_message(
                scene.bot, scene.user_id, scene.group_id, UniMessage(segments) + message
            )

        result = DeliveryResult()
        try:
            result = await _deliver(
                "\n".join(texts),
                scene.bot,
                send,
                result=result,
                context=context,
                origin="tool",
                chunks=[(UniMessage(Text(text)), text) for text in texts],
                target=delivery_target(scene.bot, scene.group_id, scene.user_id),
                request_key=(tuple(texts), reply_to, at),
            )
        finally:
            SendQuota.refund(scene.key, max(0, len(texts) - result.attempted))
        if result.status != "sent":
            return ToolResult(output=result.error or "消息未全部送达。").as_error(
                False, uncertain=result.uncertain
            )
        return ToolResult(
            output=f"已发送 {len(result.parts)} 条。", reused=result.reused
        )

    @tool(
        name="poke",
        description="戳一戳当前群的一位成员或私聊对象；user_id 为空时戳当前说话的人。",
    )
    async def poke(self, context: RunContext, user_id: str = "") -> ToolResult:
        scene = Scene.of(context)
        if scene is None:
            return ToolResult(output="当前不在会话中。").as_error()
        target = user_id.strip() or context.get_user_id() or ""
        if scene.group_id:
            if not await scene.is_member(target):
                return ToolResult(output="只能戳本群成员。").as_error()
        elif target != scene.user_id:
            return ToolResult(output="私聊中只能戳对方。").as_error()
        if not target.isdigit():
            return ToolResult(output="用户 id 应为 QQ 号。").as_error()
        if reason := SendQuota.take(scene.key):
            return ToolResult(output=reason).as_error(False)
        kind = "group" if scene.group_id else "private"
        values: dict[str, Any] = {
            "group_id": int(scene.group_id or 0),
            "user_id": int(target),
            "message": [{"type": "poke", "data": {"qq": target}}],
        }
        methods = _POKE_METHODS[kind]
        known = self._poke_methods.get((scene.bot.self_id, kind))
        methods = sorted(methods, key=lambda m: m[0] != known)
        for api, keys in methods:
            try:
                await scene.bot.call_api(api, **{k: values[k] for k in keys})
            except (ActionFailed, ApiNotAvailable, NotImplementedError) as exc:
                logger.debug(f"戳一戳接口 {api} 不可用: {type(exc).__name__}", MODULE)
                continue
            except Exception as exc:
                logger.debug(f"戳一戳结果未知: {type(exc).__name__}", MODULE)
                return ToolResult(
                    output="戳一戳未确认完成，本轮不要再次执行。"
                ).as_error(False, uncertain=True)
            self._poke_methods[(scene.bot.self_id, kind)] = api
            scene.note(f"[戳了戳 {target}]")
            return ToolResult(output="已戳了一下。")
        SendQuota.refund(scene.key, 1)
        return ToolResult(output="协议端不支持戳一戳。").as_error(False)
