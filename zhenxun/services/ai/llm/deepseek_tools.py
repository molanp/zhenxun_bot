"""DeepSeek 独立工具 API：固定官方地址、单次请求与有界媒体处理。"""

import asyncio
import io
import json
import re
from urllib.parse import urlsplit
import wave

import httpx

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.exceptions import (
    ConfigurationException,
    InvalidRequestException,
    LLMException,
    ResponseParseException,
)
from zhenxun.services.ai.core.messages import AudioResponse, TranscriptionResponse
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.core.options import TTSConfig
from zhenxun.services.network_proxy import ManagedAsyncClient

API_ROOT = "https://api.deepseek.com/v1/tools"
_OPERATIONS = {"search", "fetch", "ocr", "asr", "tts"}
_RATES = {8000, 16000, 22050, 24000, 32000, 44100, 48000}
_STATUS = {
    400: "参数或文件格式不被接受，也可能命中内容审核",
    401: "API Key 无效",
    402: "账户余额不足",
    403: "没有接口权限",
    408: "上传超时",
    413: "文件过大",
    429: "请求频率或并发超限",
    500: "服务内部错误",
    503: "服务暂不可用",
    504: "处理超时",
}


def api_key() -> str:
    """读取指定的官方服务商密钥，不读取 DSH 登录凭据或中转站密钥。"""
    config = get_llm_config()
    name = config.deepseek_tools.provider.strip()
    for provider in config.providers:
        if name and provider.name != name:
            continue
        url = urlsplit(provider.base_url)
        if (
            url.scheme == "https"
            and url.hostname == "api.deepseek.com"
            and not url.username
            and not url.password
            and url.port in (None, 443)
        ):
            if provider.api_keys:
                return provider.api_keys[0]
    raise ConfigurationException(
        "请先配置官方 DeepSeek 服务商及 API Key，并检查工具服务商名称"
    )


async def request(
    operation: str,
    *,
    keys: list[str] | None = None,
    token: CancellationToken | None = None,
    **kwargs,
):
    """调用固定工具端点，不重试上传、切换密钥或跟随重定向。"""
    if operation not in _OPERATIONS:
        raise InvalidRequestException("未知的 DeepSeek 工具接口")
    if token:
        token.raise_if_cancelled()
    key = next((k for k in keys or [] if k), None) if keys is not None else api_key()
    if not key:
        raise ConfigurationException("DeepSeek 工具接口需要 API Key")
    timeout = get_llm_config().deepseek_tools.timeout

    async def send():
        async with ManagedAsyncClient(
            timeout=timeout, follow_redirects=False
        ) as client:
            async with client.stream(
                "POST",
                f"{API_ROOT}/{operation}",
                headers={"Authorization": f"Bearer {key}"},
                **kwargs,
            ) as response:
                if not 200 <= response.status_code < 300:
                    if response.status_code >= 500:
                        from zhenxun.services.ai.run.context import interrupt_network

                        interrupt_network(f"deepseek_{operation}")
                    reason = _STATUS.get(response.status_code, "请求未成功，未自动重发")
                    raise LLMException(
                        f"DeepSeek {operation}: HTTP {response.status_code}，{reason}"
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 32 * 1024 * 1024:
                        raise ResponseParseException(
                            "DeepSeek 工具响应超过 32 MiB，未自动重发"
                        )
                if operation == "tts" and "json" not in response.headers.get(
                    "content-type", ""
                ):
                    if not body:
                        raise ResponseParseException("DeepSeek 未返回音频")
                    return bytes(body)
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError) as exc:
                    raise ResponseParseException(
                        "DeepSeek 工具响应不是有效 JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ResponseParseException("DeepSeek 工具响应结构无效")
                if error := payload.get("error"):
                    code = error.get("code") if isinstance(error, dict) else None
                    code = (
                        code
                        if isinstance(code, str)
                        and re.fullmatch(r"[a-zA-Z0-9_]{1,80}", code)
                        else "unknown"
                    )
                    raise LLMException(
                        f"DeepSeek {operation} 处理失败（{code}），未自动重发"
                    )
                if operation == "tts":
                    raise ResponseParseException("DeepSeek TTS 返回的不是音频")
                return payload

    future = asyncio.create_task(send())
    if token:
        token.link_future(future)
    try:
        return await asyncio.wait_for(future, timeout)
    except (httpx.TransportError, asyncio.TimeoutError) as exc:
        from zhenxun.services.ai.run.context import interrupt_transport

        interrupt_transport(exc, f"deepseek_{operation}")
        raise LLMException(
            f"DeepSeek {operation} 通信失败，执行结果未确认，未自动重发"
        ) from exc


async def search(query: str, count: int, *, keys: list[str] | None = None) -> dict:
    """搜索文字网页，校验查询长度并保留完整查询。"""
    if not isinstance(query, str) or not query.strip() or len(query) > 128:
        raise InvalidRequestException("DeepSeek 搜索词须为 1～128 个字符，不能全为空白")
    if type(count) is not int or not 1 <= count <= 50:
        raise InvalidRequestException("DeepSeek 搜索条数须为 1～50 的整数")
    result = await request(
        "search",
        keys=keys,
        json={"query": query, "count": count, "need_content": False},
    )
    if not isinstance(result.get("results"), list) or any(
        not isinstance(x, dict) for x in result["results"]
    ):
        raise ResponseParseException("DeepSeek 搜索结果结构无效")
    return result


async def fetch(url: str, *, keys: list[str] | None = None) -> dict:
    """请求官方抓取网页，不允许带登录信息的 URL。"""
    parsed = urlsplit(url)
    if (
        len(url) > 8192
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise InvalidRequestException(
            "网页地址须为不含用户名或密码的 HTTP/HTTPS URL，最长 8192 字符"
        )
    result = await request("fetch", keys=keys, json={"url": url})
    if not isinstance(result.get("content"), str):
        raise ResponseParseException("DeepSeek 网页正文结构无效")
    return result


async def ocr_file(
    data: bytes,
    filename: str,
    *,
    purpose: str = "text_recognition",
    token: CancellationToken | None = None,
) -> dict:
    """识别文档或图片，保留逐页完成状态与原始用量。"""
    if purpose not in {"text_recognition", "rich_layout_recognition"}:
        raise InvalidRequestException("未知的 OCR 识别用途")
    if not data or len(data) > 100 * 1024 * 1024:
        raise InvalidRequestException("OCR 文件不能为空或超过 100 MiB")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or any(c in name for c in "\r\n\x00"):
        raise InvalidRequestException("OCR 文件名无效")
    result = await request(
        "ocr",
        token=token,
        data={"purpose": purpose},
        files={"file": (name, data, "application/octet-stream")},
    )
    pages = result.get("pages")
    if (
        not isinstance(pages, list)
        or not pages
        or result.get("page_count") != len(pages)
        or any(
            not isinstance(p, dict)
            or p.get("index") != i
            or not isinstance(p.get("text"), str)
            or p.get("finish_reason") not in {"stop", "length"}
            for i, p in enumerate(pages)
        )
    ):
        raise ResponseParseException("DeepSeek OCR 页码、页数或识别结果无效")
    return result


def ocr_text(result: dict) -> str:
    """将 OCR 页结果转换为文本，明确标记被截断的页面。"""
    texts = []
    for page in result["pages"]:
        text = re.sub(r"<｜box｜>.*?<｜/box｜>", "", page["text"], flags=re.S)
        text = text.replace("<｜ref｜>", "").replace("<｜/ref｜>", "")
        suffix = "（该页识别不完整）" if page["finish_reason"] == "length" else ""
        texts.append(f"[第 {page['index'] + 1} 页{suffix}]\n{text.strip()}")
    return "\n\n".join(texts)


def _wav(data: bytes) -> tuple[bytes, float] | None:
    try:
        with wave.open(io.BytesIO(data)) as reader:
            if (
                reader.getsampwidth() != 2
                or reader.getnchannels() not in (1, 2)
                or reader.getframerate() not in _RATES
            ):
                return None
            rate, channels = reader.getframerate(), reader.getnchannels()
            declared = reader.getnframes()
            frames = reader.readframes(declared)
            streamed = declared in {
                0xFFFFFFFF // (channels * 2),
                0x7FFFFFFF // (channels * 2),
            }
            if len(frames) % (channels * 2) or (
                not streamed and len(frames) != declared * channels * 2
            ):
                return None
            duration = len(frames) / (rate * channels * 2)
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(frames)
        return output.getvalue(), duration
    except (wave.Error, EOFError):
        return None


async def transcribe(
    audio, *, token: CancellationToken | None = None
) -> TranscriptionResponse:
    """转写语音，上传前转换并检查大小、采样率及两分钟时长上限。"""
    data = await audio.get_raw_bytes()
    if not data or len(data) > 8 * 1024 * 1024:
        raise InvalidRequestException("DeepSeek ASR 音频不能为空或超过 8 MiB")
    normalized = _wav(data)
    if normalized is None:
        from zhenxun.services.ai.utils.media import voice_to_wav

        converted = await voice_to_wav(data, audio.mime_type or "audio/mpeg")
        normalized = _wav(converted) if converted else None
    if normalized is None:
        raise InvalidRequestException(
            "音频无法转为 16 位 PCM WAV，请安装 ffmpeg 或发送 WAV"
        )
    data, duration = normalized
    if duration <= 0 or duration > 120 or len(data) > 8 * 1024 * 1024:
        raise InvalidRequestException(
            "DeepSeek ASR 仅接受不超过 120 秒、8 MiB 的有效音频"
        )
    result = await request(
        "asr", token=token, files={"file": ("audio.wav", data, "audio/wav")}
    )
    if not isinstance(result.get("text"), str):
        raise ResponseParseException("DeepSeek ASR 响应缺少转写文本")
    return TranscriptionResponse(
        text=result["text"].strip(), raw_response=result, model_name="DeepSeek ASR"
    )


async def synthesize(
    text: str,
    voice: str | None = None,
    *,
    config: TTSConfig | None = None,
    token: CancellationToken | None = None,
) -> AudioResponse:
    """合成 WAV 或 PCM 音频，不把工具按字符计费伪装为模型 token 用量。"""
    settings = get_llm_config().deepseek_tools
    voice = voice or settings.voice
    if (
        not text.strip()
        or sum(2 if "\u4e00" <= c <= "\u9fff" else 1 for c in text) > 10000
    ):
        raise InvalidRequestException(
            "DeepSeek TTS 文本须为 1～10000 个计费字符（汉字计 2）"
        )
    if voice not in {"stella", "tide"}:
        raise InvalidRequestException("DeepSeek TTS 音色仅支持 stella 或 tide")
    audio_format = "wav"
    if config:
        if config.speed != 1:
            raise InvalidRequestException("DeepSeek TTS 不支持调节语速")
        if "response_format" in config.model_fields_set:
            audio_format = config.response_format
        if set(config.params) - {"filter_markdown"}:
            raise InvalidRequestException("DeepSeek TTS 仅支持额外参数 filter_markdown")
    if audio_format not in {"wav", "pcm"}:
        raise InvalidRequestException("DeepSeek TTS 仅支持 WAV 或 PCM 输出")
    markdown = config.params.get("filter_markdown", True) if config else True
    if type(markdown) is not bool:
        raise InvalidRequestException("filter_markdown 必须是布尔值")
    body = {
        "text": text,
        "voice_id": voice,
        "format": audio_format,
        "filter_markdown": markdown,
    }
    if len(json.dumps(body, ensure_ascii=False).encode()) > 64 * 1024:
        raise InvalidRequestException("DeepSeek TTS 请求体超过 64 KiB")
    data = await request("tts", token=token, json=body)
    if audio_format == "wav":
        normalized = _wav(data)
        if normalized is None or normalized[1] <= 0:
            raise ResponseParseException("DeepSeek TTS 未返回有效 WAV 音频")
        data = normalized[0]
    elif len(data) % 2:
        raise ResponseParseException("DeepSeek TTS PCM 音频不完整")
    return AudioResponse(
        audio_bytes=data, audio_format=audio_format, model_name="DeepSeek TTS"
    )
