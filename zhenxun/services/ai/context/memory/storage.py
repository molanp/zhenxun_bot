from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
import datetime
from pathlib import Path
import time
from typing import Any, cast

from tortoise.timezone import now

from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    AudioPart,
    FilePart,
    ImagePart,
    LLMContentPart,
    LLMMessage,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
    VideoPart,
)
from zhenxun.utils.pydantic_compat import TypeAdapter, model_copy, model_dump
from zhenxun.utils.time_utils import TimeUtils

from .models import BaseChatContext, HistorySettlement


def _settlement_receipt(messages, run_id, revision):
    """仅返回本轮已结算可见用户输入的背景收据。"""
    matched = [
        m
        for m in messages
        if (m.metadata or {}).get("chat_run") == run_id
        and (m.metadata or {}).get("chat_revision") == revision
    ]
    claims = frozenset(
        m.metadata["background_claim"]
        for m in matched
        if m.role == "user"
        and m.metadata.get("chat_settled")
        and m.metadata.get("chat_visible")
        and m.metadata.get("background_claim")
    )
    return HistorySettlement("confirmed" if matched else "unconfirmed", claims)


_MEDIA_NAMES = {
    ImagePart: "图片",
    AudioPart: "语音",
    VideoPart: "视频",
    FilePart: "文件",
}
_CONTENT_PART = TypeAdapter(LLMContentPart)


def is_summary(message: LLMMessage) -> bool:
    """消息是否为对话摘要"""
    return bool(message.metadata and message.metadata.get("is_summary"))


def after_latest_summary(messages: list[LLMMessage]) -> list[LLMMessage]:
    """取最新的摘要及其覆盖范围之后的消息，并去掉开头可能缺少配对的工具调用

    摘要元数据中的 covers_until 为被覆盖的最后一条消息的时间，
    没有该字段时视为覆盖摘要之前的全部消息。

    参数:
        messages: 按时间排列的历史消息

    返回:
        list[LLMMessage]: [最新摘要, 之后的消息...]，没有摘要时为全部消息
    """
    summaries = [m for m in messages if is_summary(m)]
    if not summaries:
        return _drop_partial_tools(messages)
    latest = max(summaries, key=lambda m: m.created_at)
    until = (latest.metadata or {}).get("covers_until", latest.created_at)
    return [
        latest,
        *_drop_partial_tools(
            [m for m in messages if not is_summary(m) and m.created_at > until]
        ),
    ]


def _drop_partial_tools(messages: list[LLMMessage]) -> list[LLMMessage]:
    """去掉开头的工具调用与工具结果：历史被截断后它们可能缺少配对，
    各协议都会拒绝没有对应调用的工具结果"""
    start = 0
    while start < len(messages) and (
        messages[start].role == "tool" or messages[start].tool_calls
    ):
        start += 1
    return messages[start:] if start else messages


def _row_to_message(row: Any) -> LLMMessage:
    """将数据库记录转换为消息"""
    kwargs = {
        "content": DBMessageSerializer.deserialize_content(row.content),
        "metadata": row.metadata if isinstance(row.metadata, dict) else None,
        "created_at": row.created_at.timestamp() if row.created_at else time.time(),
    }
    message_class = {
        "system": SystemMessage,
        "user": UserMessage,
        "assistant": AssistantMessage,
        "tool": ToolMessage,
    }.get(row.role)
    if message_class is None:
        return LLMMessage(role=row.role, **kwargs)
    return cast(LLMMessage, message_class(**kwargs))


class DBMessageSerializer:
    """将 LLMMessage 与数据库 JSON 格式进行序列化/反序列化的帮助类"""

    @staticmethod
    def deserialize_content(content_raw: Any) -> list[LLMContentPart]:
        """反序列化数据库中的 JSON 数据为 LLMMessage 消息内容部件列表"""
        content_parts: list[LLMContentPart] = []
        if isinstance(content_raw, list):
            for p in content_raw:
                if isinstance(p, dict):
                    for k in list(p.keys()):
                        if k.startswith("_is_b64_"):
                            orig_k = k[8:]
                            if orig_k in p and isinstance(p[orig_k], str):
                                p[orig_k] = base64.b64decode(p[orig_k])
                            p.pop(k, None)
                    content_parts.append(_CONTENT_PART.validate_python(p))
        elif isinstance(content_raw, str):
            content_parts.append(TextPart(text=content_raw))
        return content_parts

    @staticmethod
    def serialize_content(
        content_payload: list[LLMContentPart] | str,
    ) -> list[dict[str, Any]]:
        """将消息内容序列化为可存入数据库的 JSON

        图片、语音、视频已转为文字的保存文字，其余媒体只保存 http 地址，
        没有地址的替换为占位文字。

        参数:
            content_payload: 消息内容

        返回:
            list[dict[str, Any]]: JSON 内容
        """
        if isinstance(content_payload, str):
            return [{"type": "text", "text": content_payload}]
        elif isinstance(content_payload, list):
            processed_content = []
            for p in content_payload:
                if name := _MEDIA_NAMES.get(type(p)):
                    url = getattr(p, "url", None) or ""
                    text = (p.metadata or {}).get("text")
                    processed_content.append(
                        {"type": "text", "text": str(text)}
                        if text
                        else model_dump(p, exclude_none=True, exclude={"raw", "path"})
                        if url.startswith(("http://", "https://"))
                        else {"type": "text", "text": f"[{name}]"}
                    )
                    continue
                p_dump = (
                    model_dump(p, exclude_none=True)
                    if hasattr(p, "model_dump")
                    else (p.copy() if isinstance(p, dict) else p)
                )
                if isinstance(p_dump, dict):
                    for k, v in list(p_dump.items()):
                        if isinstance(v, bytes):
                            p_dump[k] = base64.b64encode(v).decode("utf-8")
                            p_dump[f"_is_b64_{k}"] = True
                        elif isinstance(v, Path):
                            p_dump[k] = str(v)
                processed_content.append(p_dump)
            return (
                processed_content
                if processed_content
                else [
                    {"type": "text", "text": "[仅包含思维链或工具调度，无实质文本输出]"}
                ]
            )
        return []


def visible_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """读取已结算对话，只把平台确认的正文作为最终回复。"""
    out = []
    for msg in messages:
        meta = msg.metadata or {}
        if meta.get("chat_run"):
            if not meta.get("chat_settled") or not meta.get("chat_visible"):
                continue
            if msg.role == "assistant" and not meta.get("chat_final"):
                msg = model_copy(
                    msg,
                    update={
                        "content": [
                            p for p in msg.content if getattr(p, "type", "") != "text"
                        ]
                    },
                )
                if not msg.content:
                    continue
        out.append(msg)
    return out


def _settled_message(msg: LLMMessage, run_id: str, revision: int, delivery: dict):
    meta = msg.metadata or {}
    if meta.get("chat_run") != run_id or meta.get("chat_revision") != revision:
        return None
    meta = {
        **meta,
        "chat_settled": True,
        "chat_visible": bool(delivery.get("text")),
        "delivery": delivery.get("status", "unknown"),
    }
    content = msg.content
    if meta.get("chat_final"):
        content = [TextPart(text=delivery.get("text", ""))]
        meta.pop("token_cost", None)
    return model_copy(msg, update={"metadata": meta, "content": content})


def _delivered_fallback(messages, run_id, revision, delivery):
    """生成失败后若兜底已经送达，为该轮补入确认正文。"""
    related = [m for m in messages if (m.metadata or {}).get("chat_run") == run_id]
    if (
        not delivery.get("text")
        or not related
        or any((m.metadata or {}).get("chat_final") for m in related)
    ):
        return None
    return AssistantMessage(
        content=[TextPart(text=delivery["text"])],
        metadata={
            **(related[0].metadata or {}),
            "chat_final": True,
            "chat_settled": True,
            "chat_visible": True,
            "chat_revision": revision,
            "delivery": delivery.get("status", "unknown"),
        },
    )


class InMemoryChatContext(BaseChatContext):
    """基于内存的聊天上下文存储后端"""

    def __init__(self):
        """初始化内存聊天上下文"""
        self._messages: dict[str, list[LLMMessage]] = {}

    async def get_messages(self, session_id: str) -> list[LLMMessage]:
        """获取指定会话的所有短期历史消息"""
        return visible_messages(list(self._messages.get(session_id, [])))

    async def add_messages(self, session_id: str, messages: list[LLMMessage]) -> None:
        """向指定会话中追加历史消息，消息时间保持严格递增"""
        stored = self._messages.setdefault(session_id, [])
        for message in messages:
            if stored and message.created_at <= stored[-1].created_at:
                message = model_copy(
                    message, update={"created_at": stored[-1].created_at + 1e-6}
                )
            stored.append(message)

    async def settle_run(self, session_id, run_id, revision, delivery, started_at):
        """按运行身份结算内存中的对话。"""
        from zhenxun.services.ai.run.session import session_manager

        if revision != session_manager.revision(session_id):
            return HistorySettlement("stale")
        self._messages[session_id] = [
            _settled_message(m, run_id, revision, delivery) or m
            for m in self._messages.get(session_id, [])
        ]
        if fallback := _delivered_fallback(
            self._messages[session_id], run_id, revision, delivery
        ):
            await self.add_messages(session_id, [fallback])
        return _settlement_receipt(self._messages[session_id], run_id, revision)

    async def clear(self, session_id: str) -> None:
        """清空指定会话的全部历史消息"""
        self._messages.pop(session_id, None)


HISTORY_RETAIN: list[Callable[[], Awaitable[dict[str, int]]]] = []
"""返回各会话已处理到的消息id的回调，清理历史时保留其后的消息；由需要继续读取
历史的插件注册"""


class DatabaseChatContext(BaseChatContext):
    """默认的短期记忆后端：按消息逐条追加到 ai_chat_message 表，
    数据库未就绪时退回内存"""

    def __init__(self):
        """初始化数据库短期记忆后端"""
        self._memory = InMemoryChatContext()

    @staticmethod
    def _model() -> Any:
        from zhenxun.models.ai_chat_message import AIChatMessage

        return AIChatMessage

    @staticmethod
    def _ready() -> bool:
        from zhenxun.services.db_context import database_ready

        return database_ready()

    async def get_messages(self, session_id: str) -> list[LLMMessage]:
        """获取最新摘要及其之后的消息"""
        if not self._ready():
            return await self._memory.get_messages(session_id)
        model = self._model()
        query = model.filter(session_id=session_id, is_summary=False)
        summary = (
            await model.filter(session_id=session_id, is_summary=True)
            .order_by("-id")
            .first()
        )
        if summary:
            until = (summary.metadata or {}).get("covers_until")
            if isinstance(until, int | float):
                gap = summary.created_at.timestamp() - until
                bound = summary.created_at - datetime.timedelta(seconds=gap + 1)
                query = query.filter(created_at__gte=bound)
        rows = list(reversed(await query.order_by("-id").limit(1000)))
        messages = [_row_to_message(row) for row in rows]
        if summary:
            messages.insert(0, _row_to_message(summary))
        return after_latest_summary(_drop_partial_tools(visible_messages(messages)))

    async def add_messages(self, session_id: str, messages: list[LLMMessage]) -> None:
        """逐条追加消息"""
        if not self._ready():
            return await self._memory.add_messages(session_id, messages)
        if not messages:
            return
        model = self._model()
        base_time = now()
        last = await model.filter(session_id=session_id).order_by("-id").first()
        if last and last.created_at >= base_time:
            base_time = last.created_at + datetime.timedelta(microseconds=1)
        await model.bulk_create(
            [
                model(
                    session_id=session_id,
                    role=msg.role,
                    content=DBMessageSerializer.serialize_content(msg.content),
                    metadata=msg.metadata,
                    is_summary=is_summary(msg),
                    created_at=base_time + datetime.timedelta(microseconds=i),
                )
                for i, msg in enumerate(messages)
            ]
        )

    async def settle_run(self, session_id, run_id, revision, delivery, started_at):
        """定向结算本轮记录，旧会话修订不能改写新历史。"""
        from tortoise.timezone import now as orm_now
        from tortoise.transactions import in_transaction

        from zhenxun.services.ai.run.session import session_manager

        if revision != session_manager.revision(session_id):
            return HistorySettlement("stale")
        if not self._ready():
            return await self._memory.settle_run(
                session_id, run_id, revision, delivery, started_at
            )
        since = datetime.datetime.fromtimestamp(started_at - 1, orm_now().tzinfo)
        async with in_transaction() as connection:
            rows = (
                await self._model()
                .filter(session_id=session_id, created_at__gte=since)
                .using_db(connection)
            )
            for row in rows:
                if revision != session_manager.revision(session_id):
                    raise asyncio.CancelledError("会话已重置")
                if msg := _settled_message(
                    _row_to_message(row), run_id, revision, delivery
                ):
                    row.content = DBMessageSerializer.serialize_content(msg.content)
                    row.metadata = msg.metadata
                    await row.save(
                        using_db=connection, update_fields=["content", "metadata"]
                    )

            fallback = _delivered_fallback(
                [_row_to_message(row) for row in rows], run_id, revision, delivery
            )
            if fallback and revision == session_manager.revision(session_id):
                await self._model().bulk_create(
                    [
                        self._model()(
                            session_id=session_id,
                            role="assistant",
                            content=DBMessageSerializer.serialize_content(
                                fallback.content
                            ),
                            metadata=fallback.metadata,
                            is_summary=False,
                            created_at=now(),
                        )
                    ],
                    using_db=connection,
                )
        return _settlement_receipt(
            [_row_to_message(row) for row in rows], run_id, revision
        )

    async def clear(self, session_id: str) -> None:
        """删除会话的全部消息"""
        await self._memory.clear(session_id)
        if self._ready():
            await self._model().filter(session_id=session_id).delete()

    @classmethod
    async def cleanup(cls, retention_days: int, max_messages: int) -> int:
        """清理超过保留天数的消息，并让每个会话只保留最近的若干条

        参数:
            retention_days: 保留天数，0 表示不限制
            max_messages: 每个会话保留的条数，0 表示不限制

        返回:
            int: 删除的消息条数
        """
        if not cls._ready():
            return 0
        model = cls._model()
        deleted = 0
        keep: dict[str, int] = {}
        for retain in HISTORY_RETAIN:
            keep.update(await retain())
        sessions = await model.all().distinct().values_list("session_id", flat=True)
        for session_id in sessions:
            query = model.filter(session_id=session_id)
            if session_id in keep:
                query = query.filter(id__lte=keep[session_id])
            if retention_days > 0:
                expire = TimeUtils.now() - datetime.timedelta(days=retention_days)
                deleted += await query.filter(created_at__lt=expire).delete()
            if max_messages > 0:
                boundary = (
                    await model.filter(session_id=session_id)
                    .order_by("-id")
                    .offset(max_messages)
                    .limit(1)
                    .values_list("id", flat=True)
                )
                if boundary:
                    deleted += await query.filter(id__lte=boundary[0]).delete()
        return deleted


__all__ = [
    "DatabaseChatContext",
    "InMemoryChatContext",
]
