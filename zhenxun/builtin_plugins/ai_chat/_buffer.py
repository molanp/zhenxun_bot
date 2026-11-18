"""群聊上下文：最近群消息的内存缓冲、消息摘要与空 @ 等待"""

import asyncio
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any, ClassVar
import uuid

from nonebot.adapters import Event
from nonebot_plugin_alconna import At, Other, Reply, Text, UniMessage

from zhenxun.configs.config import BotConfig
from zhenxun.services.ai.core.templates import wrap_untrusted

AI_SEND: ContextVar[bool] = ContextVar("ai_chat_send", default=False)

_PLACEHOLDERS = {
    "image": "图片",
    "emoji": "表情",
    "voice": "语音",
    "audio": "语音",
    "video": "视频",
    "file": "文件",
    "reference": "合并转发",
    "hyper": "卡片",
    "at_all": "@全体成员",
}


@dataclass
class ChatLine:
    """群聊缓冲中的一条消息"""

    message_id: str | None
    user_id: str
    name: str
    text: str
    at: float
    is_bot: bool = False
    is_ai: bool = False
    sequence: int = 0


@dataclass(frozen=True)
class BackgroundSnapshot:
    """本轮实际读取的背景锚点，只有对应历史结算后才能确认。"""

    claim: str
    key: tuple[str, str]
    run_id: str
    revision: int
    epoch: int
    anchor: int
    sequences: tuple[int, ...]
    text: str | None
    truncated: bool = False


def sender_name(event: Event, user_id: str) -> str:
    """消息发送者的群名片或昵称，取不到时用用户id；去掉换行、半角括号与 #，
    最长 32 字，避免伪造发送者头

    参数:
        event: 事件
        user_id: 用户id

    返回:
        str: 名称
    """
    sender = getattr(event, "sender", None)
    name = getattr(sender, "card", None) or getattr(sender, "nickname", None) or ""
    name = " ".join(str(name).split()).translate(_NAME_TABLE)[:32].strip()
    return name or user_id


_NAME_TABLE = str.maketrans({"(": "（", ")": "）", "#": "＃", ":": "："})


def summarize(message: UniMessage) -> str:
    """把消息转为供群聊背景使用的简短文本，媒体用占位表示

    参数:
        message: 消息

    返回:
        str: 文本
    """
    parts = []
    for seg in message:
        if isinstance(seg, Text):
            parts.append(seg.text)
        elif isinstance(seg, At):
            parts.append(f"@{seg.target}")
        elif isinstance(seg, Reply):
            parts.append(f"[回复 #{seg.id}] ")
        elif isinstance(seg, Other) and seg.origin.type == "mface":
            summary = str(seg.origin.data.get("summary") or "").strip("[]")
            parts.append(f"[表情:{summary}]" if summary else "[表情]")
        else:
            parts.append(f"[{_PLACEHOLDERS.get(seg.type, seg.type)}]")
    return "".join(parts).strip()


class ChatBuffer:
    """每个群最近消息的内存环形缓冲（含 Bot 自己的发言）"""

    _lines: ClassVar[dict[tuple[str, str], deque[ChatLine]]] = {}
    _cursors: ClassVar[dict[tuple[str, str], int]] = {}
    _sequence: ClassVar[int] = 0
    _confirmed: ClassVar[dict[tuple[str, str], set[int]]] = {}
    _epochs: ClassVar[dict[tuple[str, str], int]] = {}
    _evicted: ClassVar[dict[tuple[str, str], int]] = {}
    MAX_LINES = 50

    @classmethod
    def record(cls, bot_id: str, group_id: str, line: ChatLine) -> None:
        """记录一条群消息

        参数:
            bot_id: Bot id
            group_id: 群号
            line: 消息
        """
        if not line.text:
            return
        key = (bot_id, group_id)
        if key not in cls._lines:
            cls._lines[key] = deque(maxlen=cls.MAX_LINES)
        cls._sequence += 1
        line.sequence = cls._sequence
        if len(cls._lines[key]) == cls.MAX_LINES:
            cls._evicted[key] = cls._lines[key][0].sequence
        cls._lines[key].append(line)

    @classmethod
    def record_bot_send(cls, bot_id: str, data: dict[str, Any], result: Any) -> None:
        """记录 Bot 发出的群消息（OneBot 的 send_msg / send_group_msg，
        QQ 官方的群消息由调用方整理为同样的参数）

        参数:
            bot_id: Bot id
            data: 接口参数
            result: 接口返回
        """
        group_id = data.get("group_id")
        if not group_id:
            return
        message = data.get("message", "")
        if hasattr(message, "extract_plain_text"):
            text = message.extract_plain_text()
        elif isinstance(message, list):
            text = "".join(
                str(seg.get("data", {}).get("text", ""))
                if seg.get("type") == "text"
                else "[图片]"
                if seg.get("type") == "image"
                else ""
                for seg in message
                if isinstance(seg, dict)
            )
        else:
            text = str(message)
        message_id = result.get("message_id") if isinstance(result, dict) else None
        nickname = BotConfig.self_nickname or "Bot"
        cls.record(
            bot_id,
            str(group_id),
            ChatLine(
                str(message_id) if message_id else None,
                bot_id,
                nickname,
                text.strip(),
                time.time(),
                is_bot=True,
                is_ai=AI_SEND.get(),
            ),
        )

    @classmethod
    def reset(cls, bot_id: str, group_id: str) -> None:
        """群聊会话不再看到此刻之前的群消息

        参数:
            bot_id: Bot id
            group_id: 群号
        """
        cls._cursors[(bot_id, group_id)] = cls._sequence
        key = (bot_id, group_id)
        cls._confirmed.pop(key, None)
        cls._epochs[key] = cls._epochs.get(key, 0) + 1

    @staticmethod
    def _format(lines: list[ChatLine]) -> str:
        """把消息排成带时间、发送者和消息id的文本，不是当天的消息标出日期

        参数:
            lines: 消息

        返回:
            str: 文本
        """
        today = time.strftime("%Y%m%d")

        def stamp(at: float) -> str:
            local = time.localtime(at)
            same_day = time.strftime("%Y%m%d", local) == today
            return time.strftime("%H:%M" if same_day else "%m-%d %H:%M", local)

        def speaker(line):
            return (
                ("你（其他功能）" if not line.is_ai else "你")
                if line.is_bot
                else line.name
            )

        return "\n".join(
            f"[{stamp(line.at)}] "
            f"{speaker(line)}({line.user_id})"
            f"{f' #{line.message_id}' if line.message_id else ''}: {line.text}"
            for line in lines
        )

    @classmethod
    def background(
        cls,
        bot_id: str,
        group_id: str,
        exclude_id: str | None,
        size: int,
        *,
        anchor: int | None = None,
    ) -> str | None:
        """生成最近群聊记录的背景提示（含 Bot 的发言），聊天内容按不可信内容包裹

        参数:
            bot_id: Bot id
            group_id: 群号
            exclude_id: 排除的消息id（当前消息）
            size: 条数

        返回:
            str | None: 背景提示
        """
        lines = [
            line
            for line in cls._lines.get((bot_id, group_id), ())
            if (not exclude_id or line.message_id != exclude_id)
            and (anchor is None or line.sequence < anchor)
        ][-size:]
        if size <= 0 or not lines:
            return None
        return "群里最近的聊天记录：\n" + wrap_untrusted(
            cls._format(lines), "group_chat"
        )

    @classmethod
    def anchor(
        cls,
        bot_id: str,
        group_id: str,
        message_id: str | None,
        fallback: int | None = None,
    ) -> int:
        """取得已接纳消息的稳定边界，失去消息时不读取后来的消息。"""
        if message_id is not None:
            for line in reversed(cls._lines.get((bot_id, group_id), ())):
                if line.message_id == message_id:
                    return line.sequence
        return fallback if fallback is not None else cls._sequence

    @classmethod
    def snapshot(
        cls,
        bot_id: str,
        group_id: str,
        anchor: int,
        size: int,
        run_id: str,
        revision: int,
        excluded: frozenset[int] = frozenset(),
    ) -> BackgroundSnapshot:
        """按消息边界读取未消费背景，不改变消费位置。"""
        key = (bot_id, group_id)
        since = cls._cursors.get(key, 0)
        buffered = list(cls._lines.get(key, ()))
        confirmed = cls._confirmed.get(key, set())
        lines = [
            line
            for line in buffered
            if not line.is_ai
            and since < line.sequence < anchor
            and line.sequence not in confirmed
            and line.sequence not in excluded
        ]
        selected = lines[: max(0, size)]
        text = (
            "你上次发言之后群里的新消息：\n"
            + wrap_untrusted(cls._format(selected), "group_chat")
            if selected
            else None
        )
        return BackgroundSnapshot(
            uuid.uuid4().hex,
            key,
            run_id,
            revision,
            cls._epochs.get(key, 0),
            anchor,
            tuple(line.sequence for line in selected),
            text,
            len(selected) < len(lines) or cls._evicted.get(key, 0) > since,
        )

    @classmethod
    def confirm(cls, snapshot: BackgroundSnapshot, run_id: str, revision: int) -> bool:
        """幂等确认本轮采用的锚点，保留尚未消费区段。"""
        key = snapshot.key
        if (
            snapshot.run_id != run_id
            or snapshot.revision != revision
            or snapshot.epoch != cls._epochs.get(key, 0)
        ):
            return False
        confirmed = cls._confirmed.setdefault(key, set())
        confirmed.update((*snapshot.sequences, snapshot.anchor))
        cursor = cls._cursors.get(key, 0)
        lines = cls._lines.get(key, ())
        for line in lines:
            if line.sequence <= cursor:
                continue
            if not line.is_ai and line.sequence not in confirmed:
                break
            cursor = line.sequence
        cls._cursors[key] = cursor
        live = {line.sequence for line in lines}
        cls._confirmed[key] = {seq for seq in confirmed if seq > cursor and seq in live}
        return True


class AtWaiter:
    """只 @ 了 Bot 没说内容时，等该用户的下一条消息一起处理"""

    _futures: ClassVar[dict[tuple[str, str, str], asyncio.Future]] = {}
    WAIT_SECONDS = 15

    @classmethod
    def pending(cls, bot_id: str, group_id: str, user_id: str) -> bool:
        """判断当前用户是否有尚未收到补问内容的空 @。"""
        future = cls._futures.get((bot_id, group_id, user_id))
        return future is not None and not future.done()

    @classmethod
    def feed(
        cls,
        bot_id: str,
        group_id: str,
        user_id: str,
        message: UniMessage,
        message_id: str | None,
    ) -> bool:
        """把消息交给正在等待该用户的空 @

        参数:
            bot_id: Bot id
            group_id: 群号
            user_id: 用户id
            message: 消息
            message_id: 消息id

        返回:
            bool: 是否被等待中的空 @ 接收
        """
        future = cls._futures.get((bot_id, group_id, user_id))
        if future is None or future.done():
            return False
        future.set_result((message, message_id))
        return True

    @classmethod
    async def wait(
        cls, bot_id: str, group_id: str, user_id: str
    ) -> tuple[UniMessage, str | None] | None:
        """等待该用户在群里的下一条消息，最多 WAIT_SECONDS 秒

        参数:
            bot_id: Bot id
            group_id: 群号
            user_id: 用户id

        返回:
            tuple[UniMessage, str | None] | None: 消息与消息id，超时或已在等待时为空
        """
        key = (bot_id, group_id, user_id)
        if key in cls._futures:
            return None
        future = asyncio.get_running_loop().create_future()
        cls._futures[key] = future
        try:
            return await asyncio.wait_for(future, cls.WAIT_SECONDS)
        except asyncio.TimeoutError:
            return None
        finally:
            cls._futures.pop(key, None)
