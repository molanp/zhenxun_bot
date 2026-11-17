"""
LLM 服务的高级 API 接口 - 便捷函数入口 (无状态)
"""

import asyncio
from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
    LLMException,
    ModelRetry,
    NetworkTimeoutException,
    get_user_friendly_error_message,
)
from zhenxun.services.ai.core.messages import (
    AudioPart,
    AudioResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImagePart,
    ImageRequest,
    ImageResponse,
    LLMMessage,
    PromptInput,
    RerankRequest,
    RerankResult,
    SpeechRequest,
    TextPart,
    TranscriptionRequest,
    UsageInfo,
    UserMessage,
    VideoPart,
)
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.core.options import (
    GenerationConfig,
    LLMEmbeddingConfig,
    OutputFormatConfig,
    TTSConfig,
)
from zhenxun.services.ai.core.templates import wrap_untrusted
from zhenxun.services.ai.utils.logger import log_llm as logger
from zhenxun.utils.pydantic_compat import model_copy

from .client import Route, execute, unmet_needs
from .manager import get_default_model, resolve_routes
from .protocols import lift_tool_media

T = TypeVar("T", bound=BaseModel)

_CAPTION_PROMPT = (
    "请客观、详细地描述这张图片的内容，包括其中出现的文字。只输出描述本身。"
)
_VIDEO_PROMPT = (
    "下面是从一段 {duration} 秒的视频中按时间顺序截取的 {count} 帧画面。"
    "请客观描述视频的内容和画面中的文字，只输出描述本身。"
)
_media_text_cache: OrderedDict[str, str] = OrderedDict()
MediaPart = ImagePart | AudioPart | VideoPart


async def _media_key(part: MediaPart) -> str:
    """媒体片段的缓存键：http 地址原样使用，其余取内容摘要

    参数:
        part: 媒体片段

    返回:
        str: 缓存键
    """
    if part.url and not part.url.startswith("data:"):
        return part.url
    if part.url:
        return hashlib.sha1(part.url.encode()).hexdigest()
    return hashlib.sha1(await part.get_raw_bytes()).hexdigest()


async def _video_request(part: VideoPart) -> ChatRequest | str | None:
    """把视频截成几帧交给视觉模型描述；没有 ffmpeg 或无法解析时只给出元信息

    参数:
        part: 视频片段

    返回:
        ChatRequest | str | None: 描述请求，或元信息文字；读取失败时为 None
    """
    from zhenxun.services.ai.utils.media import video_frames

    if not (raw := await part.get_raw_bytes()):
        return None
    frames, duration = await video_frames(raw)
    if not frames:
        length = f"，时长约 {duration:.0f} 秒" if duration else ""
        return f"一段视频（{len(raw) / 1024 / 1024:.1f}MB{length}），无法查看画面"
    prompt = _VIDEO_PROMPT.format(duration=f"{duration:.0f}", count=len(frames))
    images = [ImagePart(raw=frame, mime_type="image/jpeg") for frame in frames]
    return ChatRequest(
        messages=[UserMessage(content=[TextPart(text=prompt), *images])],
        timeout=120,
    )


async def _media_to_text(
    part: MediaPart, model: str | None, token: CancellationToken | None
) -> str | None:
    """用视觉模型描述图片或视频截帧，或用语音识别模型转写语音，结果按地址或内容缓存

    参数:
        part: 图片、语音或视频片段
        model: 视觉模型或语音识别模型
        token: 取消令牌

    返回:
        str | None: 描述或转写文字，失败或为空时为 None
    """
    is_image = isinstance(part, ImagePart)
    kind = "image" if is_image else "video" if isinstance(part, VideoPart) else "audio"
    converter = model or f"deepseek-tools:{get_llm_config().deepseek_tools.provider}"
    key = f"{kind}:{converter}:{await _media_key(part)}"
    if key in _media_text_cache:
        _media_text_cache.move_to_end(key)
        return _media_text_cache[key]
    try:
        if isinstance(part, VideoPart):
            request = await _video_request(part)
            if not isinstance(request, ChatRequest):
                return request
            text = (await invoke("chat", request, model, token)).text
        elif is_image and model is None:
            from .deepseek_tools import ocr_file, ocr_text

            result = await ocr_file(
                await part.get_raw_bytes(), "image.png", token=token
            )
            text = ocr_text(result)
        elif is_image:
            request = ChatRequest(
                messages=[UserMessage(content=[TextPart(text=_CAPTION_PROMPT), part])],
                timeout=120,
            )
            text = (await invoke("chat", request, model, token)).text
        else:
            request = TranscriptionRequest(audio=part, timeout=120)
            text = (await invoke("stt", request, model, token)).text
    except LLMException as e:
        label = {"image": "图片", "video": "视频", "audio": "语音"}[kind]
        logger.warning(f"模型 {model} 转换{label}失败: {e}")
        return None
    if not (text := text.strip()):
        return None
    _media_text_cache[key] = text
    if len(_media_text_cache) > 256:
        _media_text_cache.popitem(last=False)
    return text


def _converter(
    modality: str, routes: list[Route]
) -> tuple[type[MediaPart], str | None, str] | None:
    """取对话线路不支持的媒体所对应的转换模型

    参数:
        modality: image、audio 或 video
        routes: 对话线路

    返回:
        tuple | None: (片段类型, 转换模型, 标签)，任一线路支持该模态或
            未配置可用的转换模型时为 None
    """
    if any(not unmet_needs(route, {modality}) for route in routes):
        return None
    part_type, task, label = {
        "image": (ImagePart, "vision", "图片描述"),
        "audio": (AudioPart, "stt", "语音转写"),
        "video": (VideoPart, "vision", "视频描述"),
    }[modality]
    from zhenxun.services.ai.run.context import get_current_run_context

    context = get_current_run_context()
    scene = context.state.get("_scene_configuration", {}) if context else {}
    model = scene.get("effective", {}).get("stt") if task == "stt" else None
    if not (model := model or get_default_model(task)):
        settings = get_llm_config().deepseek_tools
        if modality == "audio" and settings.asr_enabled:
            return part_type, None, label
        if modality == "image" and settings.ocr_enabled:
            return part_type, None, "图片文字识别（OCR，不含场景描述）"
        return None
    route_task = "chat" if task == "vision" else task
    try:
        candidates = resolve_routes(model, route_task)
    except LLMException as e:
        if task == "stt" and scene.get("overrides", {}).get("stt"):
            raise
        logger.warning(f"{label}模型 {model} 不可用: {e}")
        return None
    if task == "vision" and all(unmet_needs(route, {"image"}) for route in candidates):
        logger.warning(f"视觉模型 {model} 不支持图片输入，已跳过{label}")
        return None
    return part_type, model, label


async def _describe_media(
    request: Any, routes: list[Route], token: CancellationToken | None
) -> Any:
    """对话线路都不支持图片、语音或视频输入时，用视觉模型 (default_models.vision)
    描述图片和视频截帧、用语音识别模型 (default_models.stt) 转写语音，替换为文字

    转换结果同时记在原片段的 metadata["text"] 中，供短期记忆保存；转换失败或
    未配置转换模型的媒体保持原样，由协议层替换为占位文字。

    参数:
        request: 对话请求
        routes: 对话线路
        token: 取消令牌

    返回:
        Any: 处理后的请求，无需处理时为原请求
    """
    if not isinstance(request, ChatRequest):
        return request
    converters = [
        c for m in ("image", "audio", "video") if (c := _converter(m, routes))
    ]
    targets = {
        id(part): (part, model, label)
        for part_type, model, label in converters
        for message in request.messages
        for part in message.content
        if isinstance(part, part_type)
    }
    if not targets:
        return request
    results = await asyncio.gather(
        *(_media_to_text(part, model, token) for part, model, _ in targets.values())
    )
    texts = {
        key: wrap_untrusted(text, targets[key][2])
        for key, text in zip(targets, results)
        if text
    }
    if not texts:
        return request
    for key, text in texts.items():
        part = targets[key][0]
        part.metadata = {**(part.metadata or {}), "text": text}
    messages = []
    for message in request.messages:
        if not any(id(part) in texts for part in message.content):
            messages.append(message)
            continue
        content = [
            TextPart(text=texts[id(part)]) if id(part) in texts else part
            for part in message.content
        ]
        messages.append(model_copy(message, update={"content": content}))
    logger.debug(f"已将 {len(texts)} 个媒体片段转换为文字")
    return model_copy(request, update={"messages": messages})


_EMBED_TASKS = {
    "query": "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
    "similarity": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


async def invoke(
    task: str,
    request: Any,
    model: str | None = None,
    token: CancellationToken | None = None,
) -> Any:
    """解析模型线路并执行请求

    参数:
        task: 任务类型
        request: 统一请求对象
        model: 模型全名或路由组名，为空时使用任务默认模型
        token: 取消令牌

    返回:
        Any: 统一响应对象
    """
    if (
        task == "stt"
        and not model
        and not get_default_model("stt")
        and get_llm_config().deepseek_tools.asr_enabled
    ):
        from .deepseek_tools import transcribe

        pending = transcribe(request.audio, token=token)
        try:
            return (
                await asyncio.wait_for(pending, request.timeout)
                if request.timeout
                else await pending
            )
        except asyncio.TimeoutError as exc:
            from zhenxun.services.ai.run.context import interrupt_transport

            interrupt_transport(exc, "deepseek_asr")
            raise NetworkTimeoutException("DeepSeek ASR 请求超时，未自动重发") from exc
    routes = resolve_routes(model, task)
    if task == "chat":
        messages = lift_tool_media(request.messages)
        if messages is not request.messages:
            request = model_copy(request, update={"messages": messages})
        request = await _describe_media(request, routes, token)
    if not request.timeout:
        return await execute(task, routes, request, token)
    try:
        return await asyncio.wait_for(
            execute(task, routes, request, token), request.timeout
        )
    except asyncio.TimeoutError as e:
        raise NetworkTimeoutException(f"请求超过 {request.timeout} 秒未完成") from e


def _wrap_error(action: str, error: Exception) -> LLMException:
    friendly = get_user_friendly_error_message(error)
    logger.error(f"{action}失败: {error} | 建议: {friendly}", e=error)
    return LLMException(f"{action}失败: {friendly}", cause=error)


async def chat(
    message: PromptInput | list[LLMMessage],
    *,
    model: str | None = None,
    instruction: str | None = None,
    config: GenerationConfig | None = None,
    timeout: float | None = None,
) -> ChatResponse:
    """
    无状态的聊天对话便捷函数，单次执行后立即销毁上下文。

    示例:
        response = await chat("你好", model="OpenAI/gpt-4o", instruction="你是一个助手")
        print(response.text)

    参数:
        message: 用户输入的消息内容，支持多种格式。
        model: 模型全名或路由组名，为空时使用默认模型。
        instruction: 系统指令，用于指导AI的行为和回复风格。
        config: 生成配置。
        timeout: 本次请求的超时时间（秒）。

    返回:
        ChatResponse: 包含AI回复内容、使用信息和工具调用等的完整响应对象。
    """
    from zhenxun.services.ai.message_builder import MessageBuilder

    try:
        messages = await MessageBuilder.normalize_to_llm_messages(
            message, instruction=instruction
        )
    except Exception as e:
        raise _wrap_error("消息构建", e).with_traceback(None) from None
    return await generate(messages, model=model, config=config, timeout=timeout)


@overload
async def embed(
    input_batch: PromptInput,
    *,
    model: str | None = None,
    task: Literal[
        "general", "query", "document", "similarity", "classification", "clustering"
    ] = "general",
    dimensions: int | None = None,
    multimodal: bool | list[str] = False,
    config: LLMEmbeddingConfig | None = None,
) -> EmbeddingResponse: ...


@overload
async def embed(
    input_batch: list[PromptInput],
    *,
    model: str | None = None,
    task: Literal[
        "general", "query", "document", "similarity", "classification", "clustering"
    ] = "general",
    dimensions: int | None = None,
    multimodal: bool | list[str] = False,
    config: LLMEmbeddingConfig | None = None,
) -> EmbeddingResponse: ...


async def embed(
    input_batch: PromptInput | list[PromptInput],
    *,
    model: str | None = None,
    task: Literal[
        "general", "query", "document", "similarity", "classification", "clustering"
    ] = "general",
    dimensions: int | None = None,
    multimodal: bool | list[str] = False,
    config: LLMEmbeddingConfig | None = None,
) -> EmbeddingResponse:
    """
    无状态的向量嵌入便捷函数，支持文本批量与图文多模态融合。

    参数:
        input_batch: 要生成嵌入的内容。传入单条视为单向量；传入多条视为批量。
        model: 嵌入模型全名或路由组名，为空时使用默认模型。
        task: 生成意图 (query 检索词 / document 目标文档 / similarity 相似度 等)。
        dimensions: 要求输出的向量维度。
        multimodal: 是否开启多模态嵌入，默认纯文本。
        config: 嵌入配置。

    返回:
        EmbeddingResponse: 包含向量和 Token 消耗统计的响应对象。
    """
    from zhenxun.services.ai.message_builder import MessageBuilder

    final_config = config or LLMEmbeddingConfig()
    if multimodal is not False:
        final_config.multimodal = multimodal
    if dimensions is not None:
        final_config.output_dimensionality = dimensions
    if task != "general":
        final_config.task_type = _EMBED_TASKS[task]
    try:
        batch = await MessageBuilder.normalize_to_embed_batch(
            input_batch, config=final_config
        )
        if not batch.payloads:
            return EmbeddingResponse(
                embeddings=[], usage=UsageInfo(), model_name=str(model)
            )
        request = EmbeddingRequest(batch=batch, config=final_config)
        return await invoke("embedding", request, model)
    except LLMException as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("文本嵌入", e).with_traceback(None) from None


async def rerank(
    query: str,
    documents: list[str | dict[str, str]],
    top_n: int = 3,
    *,
    model: str | None = None,
) -> list[RerankResult]:
    """
    无状态的文本重排便捷函数。

    参数:
        query: 用户查询问题
        documents: 候选文档列表 (纯文本或含 text 字段的字典)
        top_n: 返回匹配度最高的前 n 个文档
        model: 重排模型全名或路由组名

    返回:
        list[RerankResult]: 按相关度从高到低排列的结果
    """
    try:
        request = RerankRequest(query=query, documents=documents, top_n=top_n)
        response = await invoke("rerank", request, model)
        return response.results
    except LLMException as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("文档重排", e).with_traceback(None) from None


async def generate_structured(
    message: PromptInput | list[LLMMessage],
    response_model: type[T],
    *,
    model: str | None = None,
    config: GenerationConfig | None = None,
    max_retries: int | None = None,
    error_prompt_template: str | None = None,
    instruction: str | None = None,
    timeout: float | None = None,
) -> T:
    """
    请求大模型生成结构化数据，并自动验证/解析为指定的 Pydantic 模型。

    示例:
        class UserInfo(BaseModel):
            name: str
        info = await generate_structured("提取张三的信息", response_model=UserInfo)

    参数:
        message: 输入的消息内容，支持纯文本、UniMessage、消息对象列表等。
        response_model: 目标结构化输出的 Pydantic 模型类。
        model: 模型全名或路由组名，为空时使用默认模型。
        config: 生成配置。
        max_retries: 解析失败时的最大自我修正次数，为空时使用全局配置。
        error_prompt_template: 校验失败时引导模型自我修正的提示词模板。
        instruction: 系统指令。
        timeout: 本次请求的超时时间（秒）。

    返回:
        T: 解析验证通过后的模型实例。
    """
    from zhenxun.services.ai.capabilities.builtin import ReflexionCapability
    from zhenxun.services.ai.core.engine.structured_parser import (
        BaseOutputProcessor,
    )
    from zhenxun.services.ai.message_builder import MessageBuilder

    try:
        processor = BaseOutputProcessor(
            response_model=response_model, error_template=error_prompt_template
        )
        structured = GenerationConfig(
            output=OutputFormatConfig(
                response_format="json", response_schema=processor.get_json_schema()
            )
        )
        messages = await MessageBuilder.normalize_to_llm_messages(
            message if message is not None else [], instruction=instruction
        )
        if max_retries is None:
            max_retries = get_llm_config().client_settings.structured_retries
        response = await generate(
            messages,
            model=model,
            config=structured.merge_with(config),
            timeout=timeout,
            extra={
                "output_processor": processor,
                "max_retries": max_retries,
                "__sys_capabilities": [ReflexionCapability()],
            },
        )
    except (LLMException, ModelRetry, ControlFlowExit) as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("生成结构化响应", e).with_traceback(None) from None
    if response.parsed_obj is None:
        raise LLMException("结构化输出失败：未得到解析后的对象。")
    return response.parsed_obj


async def generate(
    messages: list[LLMMessage],
    *,
    model: str | None = None,
    config: GenerationConfig | None = None,
    timeout: float | None = None,
    extra: dict[str, Any] | None = None,
    token: CancellationToken | None = None,
) -> ChatResponse:
    """
    直接传入消息列表生成响应，一般业务插件推荐使用 `chat`。

    参数:
        messages: 完整的消息历史列表。
        model: 模型全名或路由组名，为空时使用默认模型。
        config: 生成配置，与提供商和模型参数合并后生效。
        timeout: 本次请求的超时时间（秒）。
        extra: 透传给能力钩子的额外数据。
        token: 取消令牌。

    返回:
        ChatResponse: 包含AI回复内容、使用信息和工具调用等的完整响应对象。
    """
    request = ChatRequest(
        messages=messages, config=config, timeout=timeout, extra=extra or {}
    )
    capabilities = request.extra.pop("__sys_capabilities", [])
    run_context = request.extra.pop("run_context", None)
    try:
        if not capabilities:
            return await invoke("chat", request, model, token)

        from zhenxun.services.ai.capabilities.base import CombinedCapability
        from zhenxun.services.ai.core.models import LLMContext
        from zhenxun.services.ai.run.context import RunContext

        async def handler(ctx: LLMContext[ChatRequest, ChatResponse]) -> ChatResponse:
            return await invoke("chat", ctx.request, model, token)

        return await CombinedCapability(capabilities).wrap_model_request(
            run_context or RunContext(), LLMContext(request=request), handler
        )
    except (LLMException, ModelRetry, ControlFlowExit) as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("生成响应", e).with_traceback(None) from None


@overload
async def create_image(
    prompt: PromptInput,
    *,
    images: None = None,
    model: str | None = None,
    config: GenerationConfig | None = None,
) -> ImageResponse:
    """根据文本提示生成一张新图片。"""
    ...


@overload
async def create_image(
    prompt: PromptInput,
    *,
    images: list[Path | bytes | str] | Path | bytes | str,
    model: str | None = None,
    config: GenerationConfig | None = None,
) -> ImageResponse:
    """在给定图片的基础上，根据文本提示进行编辑或重新生成。"""
    ...


async def create_image(
    prompt: PromptInput,
    *,
    images: list[Path | bytes | str] | Path | bytes | str | None = None,
    model: str | None = None,
    config: GenerationConfig | None = None,
) -> ImageResponse:
    """生成或编辑图片

    提示词中的文字作为图片描述，其中的图片与 images 一起作为参考图
    （提示词中的在前）。没有参考图时生成新图片，有参考图时进行编辑。

    参数:
        prompt: 提示词，支持文本、UniMessage、LLMMessage 或消息片段列表
        images: 额外的参考图（字节、路径、URL 或 data URI）
        model: 图片模型全名或路由组名，为空时使用 image 默认模型
        config: 生成配置

    返回:
        ImageResponse: 生成的图片

    示例:
        res = await create_image("画一只猫", model="OpenAI/gpt-image-1")
        img_bytes = res.images[0]
    """
    from zhenxun.services.ai.message_builder import MessageBuilder

    extra = images if isinstance(images, list) else [images] if images else []
    try:
        messages = await MessageBuilder.normalize_to_llm_messages(prompt)
        parts = [
            part
            for message in messages
            if isinstance(message.content, list)
            for part in message.content
        ]
        text = "\n".join(m.extract_text for m in messages if m.extract_text)
        image_list = [p for p in parts if isinstance(p, ImagePart)] + list(extra)
        request = ImageRequest(
            prompt=text, images=image_list or None, config=config or GenerationConfig()
        )
        return await invoke("image", request, model)
    except LLMException as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("图片生成", e).with_traceback(None) from None


async def create_speech(
    text: str,
    voice: str | None = None,
    *,
    model: str | None = None,
    config: TTSConfig | None = None,
) -> AudioResponse:
    """
    通用文本转语音便捷函数。

    参数:
        text: 待合成的文本内容。
        voice: 音色，为空时使用模型的默认音色。
        model: 语音模型全名或路由组名。
        config: 语音合成配置。

    示例:
        res = await create_speech("你好，世界", voice="alloy", model="OpenAI/tts-1")
        Path("out.mp3").write_bytes(res.audio_bytes)
    """
    if not text:
        raise LLMException("TTS 输入文本不能为空")
    try:
        if (
            not model
            and not get_default_model("tts")
            and get_llm_config().deepseek_tools.tts_enabled
        ):
            from .deepseek_tools import synthesize

            return await synthesize(text, voice, config=config)
        request = SpeechRequest(input_text=text, voice=voice, config=config)
        return await invoke("tts", request, model)
    except LLMException as e:
        raise e.with_traceback(None) from None
    except Exception as e:
        raise _wrap_error("语音生成", e).with_traceback(None) from None
