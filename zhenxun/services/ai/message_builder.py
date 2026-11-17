import asyncio
import base64
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, TypeVar, cast

from nonebot.adapters import Message as PlatformMessage
from nonebot.matcher import current_bot, current_event, current_matcher
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_alconna.uniseg import (
    At,
    AtAll,
    Audio,
    Image,
    Reply,
    Segment,
    Text,
    Video,
    Voice,
)
from nonebot_plugin_alconna.uniseg.tools import image_fetch, reply_fetch

from zhenxun.services.ai.core.messages import (
    AudioPart,
    BaseContentPart,
    EmbedBatch,
    EmbedPayload,
    FilePart,
    ImagePart,
    LLMContentPart,
    LLMMessage,
    PromptInput,
    SystemMessage,
    TextPart,
    UserContentUnion,
    VideoPart,
)
from zhenxun.services.ai.core.options import LLMEmbeddingConfig
from zhenxun.services.ai.run.context import get_current_run_context
from zhenxun.services.ai.utils.logger import log_core as logger
from zhenxun.services.ai.utils.utils import (
    MEDIA_MAX_BYTES,
    MediaFetchError,
    fetch_media,
)
from zhenxun.utils.pydantic_compat import TypeAdapter, model_dump
from zhenxun.utils.utils import infer_plugin_namespace

S = TypeVar("S", bound=Segment)
SegmentHandler = Callable[
    [Any], Awaitable[LLMContentPart | list[LLMContentPart] | None]
]
"""消息段处理器：把一个 Uniseg 消息段转换为内容片段"""
_USER_CONTENT = TypeAdapter(UserContentUnion)


class MessageBuilder:
    """
    平台消息与 LLM 内部结构转换的 Builder 门面。
    实现 Nonebot/Alconna 生态与底层 LLM 核心（types/messages）的绝对解耦。
    """

    _SEGMENT_HANDLERS: ClassVar[dict[type[Segment], dict[str, SegmentHandler]]] = {}

    @classmethod
    def register_segment_handler(cls, seg_type: type[S], scope: str | None = None):
        """装饰器：注册 Uniseg 消息段的处理器

        参数:
            seg_type: 消息段类型
            scope: 生效的插件命名空间，为空时为调用方插件，"global" 为全局
        """
        ns = scope if scope is not None else infer_plugin_namespace()

        def decorator(
            func: Callable[
                [S], Awaitable[LLMContentPart | list[LLMContentPart] | None]
            ],
        ):
            cls._SEGMENT_HANDLERS.setdefault(seg_type, {})[ns] = func
            return func

        return decorator

    @classmethod
    async def _transform_to_content_part(cls, item: Any) -> UserContentUnion:
        """将任意输入项转换为符合 LLM 规范的内容部件"""
        if isinstance(item, BaseContentPart):
            return _USER_CONTENT.validate_python(model_dump(item))
        if isinstance(item, str):
            return TextPart(text=item)
        raise TypeError(f"不支持的输入类型用于构建 ContentPart: {type(item)}")

    @classmethod
    async def unimsg_to_llm_parts(
        cls,
        message: UniMessage,
        namespace: str | None = None,
        allowed_modalities: set[str] | None = None,
    ) -> list[UserContentUnion]:
        """将 UniMessage 消息解析并转换为 LLM 内容部件列表

        参数:
            message: 消息
            namespace: 插件命名空间，优先使用该命名空间注册的消息段处理器
            allowed_modalities: 允许的模态，为空时不限制

        返回:
            list[UserContentUnion]: 内容片段
        """
        namespace = namespace or infer_plugin_namespace()
        parts: list[UserContentUnion] = []
        for seg in message:
            if allowed_modalities is not None:
                if isinstance(seg, Image) and "image" not in allowed_modalities:
                    continue
                if isinstance(seg, Audio | Voice) and "audio" not in allowed_modalities:
                    continue
                if isinstance(seg, Video) and "video" not in allowed_modalities:
                    continue

            handlers = cls._SEGMENT_HANDLERS.get(type(seg), {})
            if handler := handlers.get(namespace) or handlers.get("global"):
                try:
                    part = await handler(seg)

                    if part:
                        if isinstance(part, list):
                            parts.extend(cast(list[UserContentUnion], part))
                        else:
                            parts.append(cast(UserContentUnion, part))
                except Exception as e:
                    from zhenxun.services.ai.run.context import interrupt_transport

                    interrupt_transport(e, "media_input")
                    logger.warning(f"处理消息段 {seg} 失败: {e}", "LLMUtils")
        merged_parts: list[UserContentUnion] = []
        for part in parts:
            if (
                isinstance(part, TextPart)
                and merged_parts
                and isinstance(merged_parts[-1], TextPart)
            ):
                merged_parts[-1].text += part.text
            else:
                merged_parts.append(part)
        return merged_parts

    @classmethod
    async def reply_to_parts(
        cls,
        reply: Reply,
        namespace: str | None = None,
        allowed_modalities: set[str] | None = None,
    ) -> list[LLMContentPart] | None:
        """把引用消息转为内容片段，标出被引用者并与当前消息分隔；
        引用段不带原消息时向适配器获取

        参数:
            reply: 引用消息段
            namespace: 插件命名空间
            allowed_modalities: 允许的模态，为空时不限制

        返回:
            list[LLMContentPart] | None: 内容片段，取不到原消息时为空
        """
        if reply.msg is None:
            reply = await _fetch_reply(reply) or reply
        if not reply.msg:
            return None
        if isinstance(reply.msg, PlatformMessage):
            uni_msg = UniMessage.of(reply.msg)
        else:
            uni_msg = UniMessage.text(str(reply.msg))
        parts = await cls.unimsg_to_llm_parts(
            uni_msg.exclude(Reply),
            namespace=namespace,
            allowed_modalities=allowed_modalities,
        )
        if not parts:
            return None
        sender = getattr(reply.origin, "sender", None)
        user_id = getattr(sender, "user_id", None)
        name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
        who = f" {name or user_id}({user_id}) 的消息" if user_id else "消息"
        return [
            TextPart(text=f"[引用{who}]\n"),
            *cast(list[LLMContentPart], parts),
            TextPart(text="\n[引用结束]\n"),
        ]

    @classmethod
    async def normalize_to_llm_messages(
        cls,
        message: PromptInput,
        instruction: str | None = None,
        namespace: str | None = None,
        allowed_modalities: set[str] | None = None,
    ) -> list[LLMMessage]:
        """将任意类型的提示输入标准化为统一的 LLM 消息历史列表"""
        namespace = namespace or infer_plugin_namespace()
        messages = []
        if instruction:
            messages.append(SystemMessage(content=[TextPart(text=instruction)]))

        if isinstance(message, UniMessage):
            parts = await cls.unimsg_to_llm_parts(
                message, namespace=namespace, allowed_modalities=allowed_modalities
            )
            converted_msgs = [LLMMessage.user(parts)]
        elif isinstance(message, LLMMessage):
            converted_msgs = [message]
        elif isinstance(message, list) and all(
            isinstance(m, LLMMessage) for m in message
        ):
            converted_msgs = cast(list[LLMMessage], list(message))
        elif isinstance(message, str):
            converted_msgs = [LLMMessage.user(message)]
        elif isinstance(message, list):
            parts = [await cls._transform_to_content_part(item) for item in message]
            converted_msgs = [LLMMessage.user(parts)]
        else:
            raise TypeError(f"不支持的消息类型: {type(message)}")

        messages.extend(converted_msgs)
        return messages

    @classmethod
    def message_to_unimessage(cls, message: PlatformMessage) -> UniMessage:
        """将平台原生 Message 转换为统一的 UniMessage"""
        return UniMessage.of(message)

    @classmethod
    async def _extract_parts_for_embed(
        cls,
        item: Any,
        namespace: str | None = None,
        config: LLMEmbeddingConfig | None = None,
    ) -> list[LLMContentPart]:
        """为 Embed 向量化提取纯粹的内容片段，忽略杂项"""
        namespace = namespace or infer_plugin_namespace()
        allowed_modalities = {"text"}
        if config:
            if config.multimodal is True:
                allowed_modalities = None
            elif isinstance(config.multimodal, list):
                allowed_modalities = set(config.multimodal)
                allowed_modalities.add("text")
            elif config.multimodal is False:
                allowed_modalities = {"text"}

        messages = await cls.normalize_to_llm_messages(
            item, namespace=namespace, allowed_modalities=allowed_modalities
        )
        parts = []
        for msg in messages:
            for part in msg.content:
                if isinstance(
                    part, TextPart | ImagePart | AudioPart | VideoPart | FilePart
                ):
                    parts.append(part)
        return parts

    @classmethod
    async def normalize_to_embed_batch(
        cls,
        inputs: Any,
        namespace: str | None = None,
        config: LLMEmbeddingConfig | None = None,
    ) -> "EmbedBatch":
        """将任意输入标准化为嵌入向量批处理对象"""
        namespace = namespace or infer_plugin_namespace()
        if isinstance(inputs, list) and not isinstance(inputs, UniMessage):
            if not inputs:
                return EmbedBatch(payloads=[])
            if isinstance(inputs[0], BaseContentPart):
                return EmbedBatch(payloads=[EmbedPayload(parts=inputs)])

            batch = EmbedBatch(payloads=[])
            for item in inputs:
                parts = await cls._extract_parts_for_embed(item, namespace, config)
                if parts:
                    batch.payloads.append(EmbedPayload(parts=parts))
                else:
                    fallback_parts: list[LLMContentPart] = [TextPart(text=" ")]
                    batch.payloads.append(EmbedPayload(parts=fallback_parts))
            return batch

        else:
            parts = await cls._extract_parts_for_embed(inputs, namespace, config)
            if not parts:
                fallback_parts: list[LLMContentPart] = [TextPart(text=" ")]
                parts = fallback_parts
            return EmbedBatch(payloads=[EmbedPayload(parts=parts)])


@MessageBuilder.register_segment_handler(Text, scope="global")
async def _handle_text(seg: Text) -> TextPart | None:
    return TextPart(text=seg.text) if seg.text.strip() else None


def _extract_media_kwargs(seg: Segment, default_mime: str) -> dict | None:
    """提取媒体 Segment 的公共属性字典，消除冗余解析"""
    mime_type = getattr(seg, "mimetype", None) or default_mime

    raw_data = getattr(seg, "raw", None)
    if raw_data:
        return {
            "raw": raw_data if isinstance(raw_data, bytes) else raw_data.read(),
            "mime_type": mime_type,
        }

    path_data = getattr(seg, "path", None)
    if path_data is not None:
        return {"path": Path(str(path_data)), "mime_type": mime_type}

    url_data = getattr(seg, "url", None)
    if url_data:
        return {"url": url_data, "mime_type": mime_type}

    return None


@MessageBuilder.register_segment_handler(Image, scope="global")
async def _handle_image(
    seg: Image,
) -> ImagePart | TextPart | list[LLMContentPart] | None:
    """把图片消息段转为图片片段：协议端只给地址时先下载，按 image_max_side 压缩后
    交给模型（消息段保留原图），动图拼成 3×3 并附说明

    参数:
        seg: 图片消息段

    返回:
        ImagePart | TextPart | list[LLMContentPart] | None: 图片片段，
            超过大小上限时为占位文字
    """
    if not seg.raw and not getattr(seg, "path", None):
        try:
            bot = current_bot.get(None)
            event = current_event.get(None)
            matcher = current_matcher.get(None)

            if bot and event and matcher:
                logger.debug("MessageBuilder 正在底层静默拉取图片实体...")
                raw_bytes = await image_fetch(event, bot, matcher.state, seg)
                if raw_bytes:
                    seg.raw = raw_bytes
        except Exception as e:
            logger.debug(f"底层静默水合下载图片失败: {e}")

        if too_large := await _download_segment(seg, "图片"):
            return too_large

    kwargs = _extract_media_kwargs(seg, "image/png")
    if not kwargs:
        return None
    from zhenxun.services.ai.config import get_llm_config
    from zhenxun.services.ai.utils.media import compress_image

    if "raw" not in kwargs and "path" not in kwargs:
        return ImagePart(**kwargs)
    try:
        data = kwargs.get("raw") or await asyncio.to_thread(kwargs["path"].read_bytes)
    except OSError as e:
        logger.debug(f"读取图片失败: {e}")
        return ImagePart(**kwargs)
    max_side = get_llm_config().context_settings.image_max_side
    if max_side and (result := await asyncio.to_thread(compress_image, data, max_side)):
        raw, mime, animated = result
        part = ImagePart(raw=raw, mime_type=mime)
        if animated:
            return [TextPart(text="[动图，按时间顺序抽取 9 帧拼成 3×3]"), part]
        return part
    return ImagePart(raw=data, mime_type=kwargs["mime_type"])


async def _download_segment(
    seg: Image | Audio | Voice | Video, label: str
) -> TextPart | None:
    """把协议端提供的媒体地址下载为字节，存回消息段

    参数:
        seg: 媒体消息段
        label: 媒体名称，用于占位文字

    返回:
        TextPart | None: 超过大小上限时的占位文字；其余情况为 None，
            下载失败时保留原地址
    """
    url = seg.url
    if seg.raw or getattr(seg, "path", None) or not url:
        return None
    try:
        seg.raw = await fetch_media(url, allow_private=True)
        seg.url = None
    except MediaFetchError as e:
        logger.debug(f"下载{label} {url[:80]} 失败: {e}")
        if e.too_large:
            limit = MEDIA_MAX_BYTES // 1024 // 1024
            return TextPart(text=f"[{label}: 超过 {limit}MB，未读取]")
    return None


def sniff_audio_mime(data: bytes) -> str | None:
    """按文件头识别音频格式

    参数:
        data: 音频数据（至少包含文件头）

    返回:
        str | None: MIME 类型，无法识别时为 None
    """
    head = data[:16]
    if head.startswith((b"\x02#!SILK", b"#!SILK")):
        return "audio/silk"
    if head.startswith(b"#!AMR"):
        return "audio/amr"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"ID3") or (
        len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
    ):
        return "audio/mp3"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head[4:8] == b"ftyp":
        return "audio/mp4"
    return None


async def _record_as_wav(file_id: str) -> bytes | None:
    """请求 OneBot V11 协议端把语音转为 wav

    参数:
        file_id: 语音消息段中的 file 字段

    返回:
        bytes | None: wav 数据，协议端不支持或转换失败时为 None
    """
    bot = current_bot.get(None)
    if bot is None or bot.adapter.get_name() != "OneBot V11":
        return None
    try:
        result = await bot.call_api("get_record", file=file_id, out_format="wav")
    except Exception as e:
        logger.debug(f"协议端转换语音 {file_id} 失败: {e}")
        return None
    if not isinstance(result, dict):
        return None
    data = None
    if result.get("base64"):
        data = base64.b64decode(str(result["base64"]).removeprefix("base64://"))
    elif result.get("file") and Path(str(result["file"])).is_file():
        data = Path(str(result["file"])).read_bytes()
    return data if data and sniff_audio_mime(data) == "audio/wav" else None


async def _process_audio_seg(seg: Audio | Voice) -> AudioPart | TextPart | None:
    if too_large := await _download_segment(seg, "语音"):
        return too_large
    raw = seg.raw.getvalue() if isinstance(seg.raw, BytesIO) else seg.raw
    if raw:
        mime = sniff_audio_mime(raw)
        if mime in ("audio/silk", "audio/amr"):
            wav = await _record_as_wav(seg.id) if seg.id else None
            if wav is None:
                from zhenxun.services.ai.utils.media import voice_to_wav

                wav = await voice_to_wav(raw, mime)
            if wav:
                raw, mime = wav, "audio/wav"
        seg.raw = raw
        seg.mimetype = mime or seg.mimetype
    kwargs = _extract_media_kwargs(seg, "audio/mp3")
    return AudioPart(**kwargs) if kwargs else None


@MessageBuilder.register_segment_handler(Audio, scope="global")
async def _handle_audio(seg: Audio) -> AudioPart | TextPart | None:
    return await _process_audio_seg(seg)


@MessageBuilder.register_segment_handler(Voice, scope="global")
async def _handle_voice(seg: Voice) -> AudioPart | TextPart | None:
    return await _process_audio_seg(seg)


@MessageBuilder.register_segment_handler(Video, scope="global")
async def _handle_video(seg: Video) -> VideoPart | TextPart | None:
    if too_large := await _download_segment(seg, "视频"):
        return too_large
    kwargs = _extract_media_kwargs(seg, "video/mp4")
    return VideoPart(**kwargs) if kwargs else None


async def _fetch_reply(reply: Reply) -> Reply | None:
    """获取引用段对应的原消息：是当前事件的引用时由适配器提供，
    OneBot V11 下其他消息按消息 id 查询

    参数:
        reply: 不带原消息的引用段

    返回:
        Reply | None: 带原消息的引用段，取不到时为空
    """
    bot, event = current_bot.get(None), current_event.get(None)
    if not bot or not event:
        return None
    fetched = await reply_fetch(event, bot)
    if fetched is not None and str(fetched.id) == str(reply.id):
        return fetched
    if (
        bot.adapter.get_name() != "OneBot V11"
        or not str(reply.id).lstrip("-").isdigit()
    ):
        return None
    from nonebot.adapters.onebot.v11 import Message as V11Message
    from nonebot.adapters.onebot.v11 import MessageSegment as V11Segment

    data = await bot.call_api("get_msg", message_id=int(reply.id))
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, list):
        message = V11Message(
            V11Segment(str(s.get("type")), s.get("data") or {})
            for s in message
            if isinstance(s, dict)
        )
    elif isinstance(message, str):
        message = V11Message(message)
    else:
        return None
    sender = data.get("sender")
    sender = {str(k): v for k, v in sender.items()} if isinstance(sender, dict) else {}
    return Reply(
        str(reply.id), message, SimpleNamespace(sender=SimpleNamespace(**sender))
    )


@MessageBuilder.register_segment_handler(Reply, scope="global")
async def _handle_reply(seg: Reply) -> list[LLMContentPart] | None:
    ctx = get_current_run_context()
    ns = getattr(ctx.session, "namespace", "global") if ctx else "global"
    try:
        return await MessageBuilder.reply_to_parts(seg, namespace=ns)
    except Exception as e:
        logger.debug(f"读取引用消息失败: {e}")
        return None


@MessageBuilder.register_segment_handler(AtAll, scope="global")
async def _handle_at_all(seg: AtAll) -> TextPart:
    return TextPart(text="[@全体成员] ")


@MessageBuilder.register_segment_handler(At, scope="global")
async def _handle_at(seg: At) -> TextPart:
    if seg.display:
        return TextPart(text=f"[@{seg.display}] ")
    target_id = str(seg.target)
    try:
        nickname = await _resolve_at_name(target_id)
    except Exception as e:
        logger.debug(f"解析 @{target_id} 的昵称失败: {e}")
        nickname = None
    return TextPart(text=f"[@{nickname or target_id}] ")


async def _resolve_at_name(target_id: str) -> str | None:
    """解析 @ 对象的显示名称

    依次使用事件发送者信息、Bot 自身昵称、群成员名称缓存，
    都没有时才调用协议端接口查询。

    参数:
        target_id: 被 @ 的用户 ID

    返回:
        str | None: 显示名称，无法获取时为 None
    """
    from zhenxun.configs.config import BotConfig
    from zhenxun.services.hot_query_cache import get_member_name

    bot = current_bot.get(None)
    event = current_event.get(None)
    group_id = getattr(event, "group_id", None)

    if event is not None and str(getattr(event, "user_id", "")) == target_id:
        sender = getattr(event, "sender", None)
        name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
        if name:
            return name
    if bot is not None and str(bot.self_id) == target_id and BotConfig.self_nickname:
        return BotConfig.self_nickname

    name = await get_member_name(target_id, group_id)
    if name and name != target_id:
        return name

    if bot is None or not target_id.isdigit():
        return None
    if group_id and hasattr(bot, "get_group_member_info"):
        info = await bot.get_group_member_info(
            group_id=group_id, user_id=int(target_id)
        )
        return info.get("card") or info.get("nickname")
    if hasattr(bot, "get_stranger_info"):
        info = await bot.get_stranger_info(user_id=int(target_id))
        return info.get("nickname")
    return None


from . import message_segments  # noqa: F401
