"""MessageBuilder 的补充消息段处理器：合并转发、卡片、文件、表情及其他平台消息段"""

import asyncio
from collections import OrderedDict
from contextvars import ContextVar
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from nonebot.matcher import current_bot, current_event
from nonebot_plugin_alconna.uniseg import (
    Emoji,
    File,
    Hyper,
    Other,
    Reference,
    Reply,
    UniMessage,
    Voice,
)

from zhenxun.services.ai.core.messages import (
    AudioPart,
    ImagePart,
    LLMContentPart,
    TextPart,
    VideoPart,
)
from zhenxun.services.ai.utils.logger import log_core as logger

from .message_builder import MessageBuilder, _process_audio_seg

FORWARD_MAX_DEPTH = 6
"""合并转发最多展开的嵌套层数"""
FORWARD_MAX_CALLS = 8
"""单条消息内最多调用 get_forward_msg 的次数"""
FORWARD_MAX_NODES = 60
"""每个合并转发最多展开的消息条数"""
FORWARD_MAX_MEDIA = 9
"""每个合并转发内保留为媒体的图片、语音、视频数量，超出部分改为占位文字"""

QQ_VOICE_CACHE_SIZE = 512
"""记住的 QQ 官方语音附件数量"""
_qq_voices: OrderedDict[str, tuple[str, str]] = OrderedDict()

_forward_depth: ContextVar[int] = ContextVar("ai_forward_depth", default=0)
_forward_calls: ContextVar[list[int] | None] = ContextVar(
    "ai_forward_calls", default=None
)
_XML_BRIEF = re.compile(r'brief="([^"]*)"')
_XML_TITLE = re.compile(r"<title>([^<]*)</title>")
_MEDIA_NAMES = {ImagePart: "图片", AudioPart: "语音", VideoPart: "视频"}
_OTHER_NAMES = {
    "dice": "骰子",
    "rps": "猜拳",
    "poke": "戳一戳",
    "shake": "窗口抖动",
    "contact": "推荐联系人",
    "music": "音乐分享",
    "location": "位置",
}


def _forward_nodes(data: Any) -> list[Any]:
    """从 get_forward_msg 的返回值中取出消息节点

    参数:
        data: 协议端返回值

    返回:
        list[Any]: 待校验的消息节点列表
    """
    if isinstance(data, dict):
        if isinstance(data.get("messages"), list):
            return data["messages"]
        inner = data.get("data")
        if isinstance(inner, dict) and isinstance(inner.get("message"), list):
            return inner["message"]
    if isinstance(data, list):
        return data
    return []


def _limit_media(parts: list[LLMContentPart]) -> list[LLMContentPart]:
    """把超出数量上限的媒体片段替换为占位文字

    参数:
        parts: 内容片段

    返回:
        list[LLMContentPart]: 处理后的内容片段
    """
    result: list[LLMContentPart] = []
    kept = 0
    for part in parts:
        name = _MEDIA_NAMES.get(type(part))
        if name and kept >= FORWARD_MAX_MEDIA:
            result.append(TextPart(text=f"[{name}]"))
            continue
        if name:
            kept += 1
        result.append(part)
    return result


async def expand_forward(forward_id: str) -> list[LLMContentPart]:
    """通过协议端展开合并转发，转为“发送者: 内容”的逐条记录

    超过嵌套层数、调用次数上限或协议端不支持时只返回占位文字。
    格式异常的节点保留占位文字，不影响其他节点的解析。

    参数:
        forward_id: 合并转发 ID

    返回:
        list[LLMContentPart]: 内容片段
    """
    placeholder: list[LLMContentPart] = [TextPart(text="[合并转发]")]
    bot = current_bot.get(None)
    if bot is None or bot.adapter.get_name() != "OneBot V11":
        return placeholder
    depth = _forward_depth.get()
    calls = _forward_calls.get()
    calls_token = None
    if calls is None:
        calls = [FORWARD_MAX_CALLS]
        calls_token = _forward_calls.set(calls)
    depth_token = _forward_depth.set(depth + 1)
    try:
        if depth >= FORWARD_MAX_DEPTH or calls[0] <= 0:
            return placeholder
        calls[0] -= 1
        try:
            nodes = _forward_nodes(await bot.call_api("get_forward_msg", id=forward_id))
        except Exception as e:
            logger.debug(f"获取合并转发 {forward_id} 失败: {e}")
            return placeholder
        if not nodes:
            return placeholder

        from nonebot.adapters.onebot.v11 import Message, MessageSegment

        parts: list[LLMContentPart] = [TextPart(text="[合并转发]\n")]
        for node in nodes[:FORWARD_MAX_NODES]:
            if not isinstance(node, dict):
                parts.append(TextPart(text="[无法解析的消息]\n"))
                continue
            sender = node.get("sender")
            if not isinstance(sender, dict):
                sender = {}
            name = (
                sender.get("card")
                or sender.get("nickname")
                or str(sender.get("user_id") or "未知")
            )
            raw = node.get("message") or node.get("content") or ""
            try:
                if isinstance(raw, list):
                    raw = [
                        MessageSegment(s["type"], s["data"])
                        if (
                            isinstance(s, dict)
                            and isinstance(s.get("type"), str)
                            and isinstance(s.get("data"), dict)
                        )
                        else MessageSegment.text("[无法解析的消息片段]")
                        for s in raw
                    ]
                elif not isinstance(raw, str):
                    raw = "[无法解析的消息]"
                uni_msg = UniMessage.of(Message(raw)).exclude(Reply)
            except (AttributeError, KeyError, TypeError, ValueError):
                parts.append(TextPart(text=f"{name}: [无法解析的消息]\n"))
                continue
            node_parts = await MessageBuilder.unimsg_to_llm_parts(uni_msg)
            parts.append(TextPart(text=f"{name}: "))
            parts.extend(node_parts or [TextPart(text="[空消息]")])
            parts.append(TextPart(text="\n"))
        if len(nodes) > FORWARD_MAX_NODES:
            rest = len(nodes) - FORWARD_MAX_NODES
            parts.append(TextPart(text=f"……另有 {rest} 条未展开\n"))
        parts.append(TextPart(text="[合并转发结束]"))
        return _limit_media(parts)
    finally:
        _forward_depth.reset(depth_token)
        if calls_token is not None:
            _forward_calls.reset(calls_token)


def _card_text(content: dict) -> str:
    """从 QQ 卡片消息的 JSON 中提取标题、描述和链接

    参数:
        content: 卡片 JSON

    返回:
        str: 卡片摘要
    """
    texts: list[str] = []
    link = ""
    meta = content.get("meta")
    if isinstance(meta, dict):
        for item in meta.values():
            if not isinstance(item, dict):
                continue
            for key in ("title", "desc", "tag"):
                value = item.get(key)
                if isinstance(value, str) and value and value not in texts:
                    texts.append(value)
            for key in ("qqdocurl", "jumpUrl", "url"):
                value = item.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    link = link or value
    if not texts and isinstance(content.get("prompt"), str):
        texts.append(content["prompt"])
    summary = " - ".join(texts) or "卡片消息"
    return f"[卡片: {summary}]" + (f" {link}" if link else "")


@MessageBuilder.register_segment_handler(Reference, scope="global")
async def _handle_reference(seg: Reference) -> list[LLMContentPart]:
    if not seg.id:
        return [TextPart(text="[合并转发]")]
    return await expand_forward(seg.id)


@MessageBuilder.register_segment_handler(Hyper, scope="global")
async def _handle_hyper(seg: Hyper) -> list[LLMContentPart] | TextPart:
    if seg.format == "xml":
        raw = seg.raw or ""
        match = _XML_BRIEF.search(raw) or _XML_TITLE.search(raw)
        return TextPart(text=f"[卡片: {match[1] if match else 'XML 消息'}]")
    content = seg.content
    if content is None and seg.raw:
        try:
            content = json.loads(seg.raw)
        except ValueError:
            content = None
    if not isinstance(content, dict):
        return TextPart(text="[卡片消息]")
    detail = (content.get("meta") or {}).get("detail") or {}
    if content.get("app") == "com.tencent.multimsg" and detail.get("resid"):
        return await expand_forward(str(detail["resid"]))
    return TextPart(text=_card_text(content))


def remember_qq_voices(data: object) -> None:
    """记下 QQ 官方事件中语音附件的 wav 地址与平台识别文字，按附件地址索引

    适配器把语音附件转为文件消息段并丢弃这两个字段，消息转换时据此还原为语音。

    参数:
        data: QQ 官方事件数据
    """
    attachments = data.get("attachments") if isinstance(data, dict) else None
    for item in attachments or ():
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if item.get("content_type") != "voice" and not item.get("voice_wav_url"):
            continue
        url = str(item["url"])
        url = url if urlparse(url).hostname else f"https://{url}"
        _qq_voices[url] = (
            str(item.get("voice_wav_url") or ""),
            str(item.get("asr_refer_text") or "").strip(),
        )
        _qq_voices.move_to_end(url)
        while len(_qq_voices) > QQ_VOICE_CACHE_SIZE:
            _qq_voices.popitem(last=False)


def is_voice(seg: object) -> bool:
    """消息段是否为语音，包括 QQ 官方被转为文件段的语音附件

    参数:
        seg: 消息段

    返回:
        bool: 是否为语音
    """
    if isinstance(seg, Voice):
        return True
    return isinstance(seg, File) and bool(seg.url) and seg.url in _qq_voices


def _direct_url(seg: File) -> str | None:
    """文件消息段自带的下载地址；OneBot V11 只取原始消息段的 url，
    uniseg 会把只有文件名的 file 字段补成“https://文件名”

    参数:
        seg: 文件消息段

    返回:
        str | None: 下载地址
    """
    bot = current_bot.get(None)
    if bot is not None and bot.adapter.get_name() == "OneBot V11":
        url = (getattr(seg.origin, "data", None) or {}).get("url")
    else:
        url = seg.url
    url = str(url or "")
    return url if url.startswith("http") else None


async def _file_url(seg: File) -> str | None:
    """文件的下载地址，OneBot V11 没有地址时按文件 id 查询群文件或私聊文件

    参数:
        seg: 文件消息段

    返回:
        str | None: 下载地址
    """
    if url := _direct_url(seg):
        return url
    bot, event = current_bot.get(None), current_event.get(None)
    if bot is None or not seg.id or bot.adapter.get_name() != "OneBot V11":
        return None
    group_id = getattr(event, "group_id", None)
    try:
        if group_id:
            result = await bot.call_api(
                "get_group_file_url", group_id=int(group_id), file_id=seg.id
            )
        else:
            result = await bot.call_api("get_private_file_url", file_id=seg.id)
    except Exception as e:
        logger.debug(f"获取文件 {seg.name} 的地址失败: {e}")
        return None
    url = result.get("url") if isinstance(result, dict) else None
    return str(url) if url else None


async def _read_file(seg: File) -> TextPart | None:
    """file_max_chars 不为 0 时下载并解析文件，截断后按不可信内容交给模型

    参数:
        seg: 文件消息段

    返回:
        TextPart | None: 文件内容，未开启、类型不支持或读取失败时为空
    """
    from zhenxun.services.ai.config import get_llm_config
    from zhenxun.services.ai.core.templates import wrap_untrusted
    from zhenxun.services.ai.utils.doc_parse import (
        DOC_TYPES,
        PLAIN_TYPES,
        TEXT_TYPES,
        parse,
    )
    from zhenxun.services.ai.utils.utils import MediaFetchError, fetch_media

    limit = get_llm_config().context_settings.file_max_chars
    name = seg.name or ""
    ext = Path(name).suffix.lower()
    if not limit or ext not in TEXT_TYPES | DOC_TYPES | PLAIN_TYPES:
        return None
    data = seg.raw if isinstance(seg.raw, bytes) else None
    try:
        if data is None:
            if not (url := await _file_url(seg)):
                return None
            data = await fetch_media(url, allow_private=True)
        text = (await asyncio.to_thread(parse, data, ext)).text.strip()
    except (MediaFetchError, ValueError) as e:
        return TextPart(text=f"[文件: {name}，未读取：{e}]")
    except Exception as e:
        logger.debug(f"解析文件 {name} 失败: {e}")
        return None
    if not text:
        return None
    if len(text) > limit:
        text = f"{text[:limit]}\n……（后面还有 {len(text) - limit} 字未读取）"
    return TextPart(text=f"[文件: {name}]\n" + wrap_untrusted(text, "file"))


@MessageBuilder.register_segment_handler(File, scope="global")
async def _handle_file(seg: File) -> LLMContentPart | None:
    if seg.url and (voice := _qq_voices.get(seg.url)):
        wav_url, text = voice
        if text:
            return TextPart(text=f"[语音] {text}")
        return await _process_audio_seg(
            Voice(url=wav_url, mimetype="audio/wav") if wav_url else Voice(url=seg.url)
        )
    if part := await _read_file(seg):
        return part
    link = _direct_url(seg) or ""
    return TextPart(text=f"[文件: {seg.name}]" + (f" {link}" if link else ""))


@MessageBuilder.register_segment_handler(Emoji, scope="global")
async def _handle_emoji(seg: Emoji) -> TextPart:
    return TextPart(text=f"[表情: {seg.name}]" if seg.name else "[表情]")


@MessageBuilder.register_segment_handler(Other, scope="global")
async def _handle_other(seg: Other) -> TextPart | None:
    seg_type = getattr(seg.origin, "type", "")
    data = getattr(seg.origin, "data", None) or {}
    if seg_type == "mface":
        summary = str(data.get("summary") or "").strip("[]")
        return TextPart(text=f"[表情包: {summary}]" if summary else "[表情包]")
    if seg_type == "share":
        title = data.get("title") or "链接"
        return TextPart(text=f"[分享: {title}] {data.get('url') or ''}".rstrip())
    if seg_type == "markdown":
        return TextPart(text=str(data.get("content") or "[Markdown 消息]"))
    if seg_type == "location":
        place = data.get("title") or data.get("content") or ""
        return TextPart(text=f"[位置: {place}]" if place else "[位置]")
    if name := _OTHER_NAMES.get(seg_type):
        return TextPart(text=f"[{name}]")
    return TextPart(text=f"[{seg_type}消息]") if seg_type else None
