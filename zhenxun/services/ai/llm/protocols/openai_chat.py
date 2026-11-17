"""
OpenAI Chat Completions 协议

适用于 OpenAI 及兼容该格式的厂商 (DeepSeek、GLM、Kimi、通义、豆包、MiMo 等)。
"""

from copy import deepcopy
import re
from typing import Any
import uuid

from zhenxun.services.ai.config.presets import detect_preset
from zhenxun.services.ai.core.exceptions import (
    ConfigurationException,
    ContentFilteredException,
    ResponseParseException,
    UpstreamServerException,
)
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    AudioPart,
    ChatRequest,
    ChatResponse,
    FilePart,
    ImagePart,
    LLMGroundingAttribution,
    LLMGroundingMetadata,
    SystemMessage,
    TextPart,
    ThoughtPart,
    ToolCallPart,
    ToolMessage,
    VideoPart,
)
from zhenxun.services.ai.core.options import GenerationConfig, deep_merge
from zhenxun.services.ai.llm.profile import map_reasoning_effort
from zhenxun.services.ai.utils.logger import log_llm as logger

from . import (
    ChatStream,
    HttpCall,
    deepseek_server_tool,
    dump_arguments,
    ensure_content,
    filter_modalities,
    media_url,
    native_tool_allowed,
    output_image,
    replay_data,
    replay_metadata,
    resolve_tool_choice,
    split_tools,
    stream_error,
    tool_output_text,
)
from .schema import schema_instruction, strict_schema, tool_schema
from .tasks import openai_tasks

PROTOCOL = "openai_chat"

_WEB_SEARCH_TOOLS = {
    "mimo": {
        "type": "web_search",
        "max_keyword": 3,
        "limit": 1,
    },
    "glm": {
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_engine": "search_std",
            "search_result": True,
        },
    },
}
"""以工具形式提供网页搜索的厂商写法；百炼改用请求体参数 enable_search"""

_OPENAI_REASONING = re.compile(r"(?:.*/)?(?:o\d|gpt-5)", re.I)
"""只接受 max_completion_tokens 的 OpenAI 推理模型"""
_AUDIO_FORMATS = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav"}
_THINK_TAG = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((data:image/[\w.+-]+;base64,[\w+/=\s]+)\)")
"""中转站把生成的图片以 Markdown data URI 写在正文中"""


async def _part(part: Any, route) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {
            "type": "image_url",
            "image_url": {"url": await media_url(part, "image/png")},
        }
    if isinstance(part, VideoPart):
        return {
            "type": "video_url",
            "video_url": {"url": await media_url(part, "video/mp4")},
        }
    if isinstance(part, AudioPart):
        if route.provider.preset == "mimo":
            return {
                "type": "input_audio",
                "input_audio": {"data": await media_url(part, "audio/wav")},
            }
        subtype = (part.mime_type or "audio/wav").split("/")[-1]
        return {
            "type": "input_audio",
            "input_audio": {
                "data": await part.get_base64_data(),
                "format": _AUDIO_FORMATS.get(subtype, subtype),
            },
        }
    if isinstance(part, FilePart):
        filename = (part.metadata or {}).get("filename") or (
            part.path.name if part.path else "file"
        )
        return {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": await part.get_data_uri("application/pdf"),
            },
        }
    return None


async def _content(parts: list[Any], route) -> str | list[dict[str, Any]]:
    if all(isinstance(p, TextPart | ThoughtPart) for p in parts):
        return "".join(p.text for p in parts if isinstance(p, TextPart))
    items = [await _part(p, route) for p in parts]
    return [item for item in items if item]


def _assistant_message(message: AssistantMessage) -> dict:
    """编码助手消息，推理内容按收到时的字段原样回传"""
    result: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(p.text for p in message.content if isinstance(p, TextPart)),
    }
    for part in message.content:
        data = (
            replay_data(part, PROTOCOL)
            if isinstance(part, TextPart | ThoughtPart)
            else None
        )
        for key, value in (data or {}).items():
            if key not in result:
                result[key] = value
            elif isinstance(value, list):
                result[key] = [*result[key], *value]
            else:
                result[key] = f"{result[key]}\n{value}"
    if message.tool_calls:
        calls = []
        for call in message.tool_calls:
            item: dict[str, Any] = {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": dump_arguments(call.args),
                },
            }
            if extra := replay_data(call, PROTOCOL):
                item.update(extra)
            calls.append(item)
        result["tool_calls"] = calls
    return result


async def build_messages(request: ChatRequest, route) -> list[dict[str, Any]]:
    """把统一消息编码为 Chat Completions 消息

    参数:
        request: 对话请求
        route: 线路

    返回:
        list[dict[str, Any]]: messages 字段
    """
    messages = filter_modalities(request.messages, route.profile, route.name)
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            result.extend(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": tool_output_text(item.output),
                }
                for item in message.tool_returns
            )
        elif isinstance(message, AssistantMessage):
            item = _assistant_message(message)
            if item["content"] or item.get("tool_calls"):
                result.append(item)
        elif isinstance(message, SystemMessage):
            text = "".join(p.text for p in message.content)
            if text:
                result.append({"role": "system", "content": text})
        else:
            content = await _content(list(message.content), route)
            if content:
                result.append({"role": message.role, "content": content})
    return result


def add_instruction(messages: list[dict[str, Any]], text: str) -> None:
    """把提示追加到首条系统消息，没有系统消息时插入一条"""
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{text}"
    else:
        messages.insert(0, {"role": "system", "content": text})


def apply_reasoning(body: dict[str, Any], config: GenerationConfig, profile) -> None:
    """按模型的推理参数写法设置推理强度

    参数:
        body: 请求体
        config: 生成配置
        profile: 模型能力档案
    """
    param = profile.reasoning_param
    if param == "minimax":
        body["reasoning_split"] = True
        return
    effort = map_reasoning_effort(config.common.reasoning_effort, profile)
    if effort is None:
        return
    enabled = effort != "none"
    if param == "openai":
        body["reasoning_effort"] = effort
    elif param == "openrouter":
        body["reasoning"] = {"effort": effort}
    elif param == "enable_thinking":
        body["enable_thinking"] = enabled
    elif param in ("thinking", "deepseek", "ark"):
        body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if param != "thinking" and enabled:
            body["reasoning_effort"] = effort


def apply_sampling(body: dict[str, Any], config: GenerationConfig, route) -> None:
    """写入通用采样参数，模型不接受采样参数时跳过"""
    common = config.common
    if route.profile.sampling:
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
        ):
            if (value := getattr(common, key)) is not None:
                body[key] = value
    if common.stop is not None:
        body["stop"] = common.stop
    max_tokens = common.max_tokens
    if max_tokens and route.profile.max_output_tokens:
        max_tokens = min(max_tokens, route.profile.max_output_tokens)
    if max_tokens:
        openai = "openai" in (route.provider.preset, detect_preset(route.base_url))
        key = (
            "max_completion_tokens"
            if openai or _OPENAI_REASONING.match(route.model.name)
            else "max_tokens"
        )
        body[key] = max_tokens


def apply_server_tools(body: dict[str, Any], server: list[Any], route) -> None:
    """按厂商写法加入服务端网页搜索，官方接口按域名识别厂商，中转站按预设

    参数:
        body: 请求体
        server: 服务端内置工具
        route: 线路
    """
    vendor = detect_preset(route.base_url) or route.provider.preset
    for tool in server:
        type_id = getattr(tool, "type_id", "")
        allowed = native_tool_allowed(route, type_id)
        if allowed and (payload := deepseek_server_tool(route, type_id)):
            body.setdefault("tools", []).append(payload)
        elif allowed and type_id == "web_search" and vendor == "dashscope":
            body["enable_search"] = True
        elif allowed and type_id == "web_search" and vendor in _WEB_SEARCH_TOOLS:
            body.setdefault("tools", []).append(deepcopy(_WEB_SEARCH_TOOLS[vendor]))
        else:
            logger.warning(
                f"模型 {route.name} 不支持或未确认支持内置工具 {type_id}，已忽略"
            )


async def build_chat(route, request: ChatRequest) -> HttpCall:
    """编码对话请求"""
    config = route.generation_config(request.config)
    profile = route.profile
    messages = await build_messages(request, route)
    body: dict[str, Any] = {"model": route.model.name, "messages": messages}
    apply_sampling(body, config, route)
    apply_reasoning(body, config, profile)

    definitions, server = await split_tools(request.tools)
    kind, allowed, name = resolve_tool_choice(request.tool_choice, config.tools.mode)
    if allowed:
        definitions = [d for d in definitions if d.name in allowed]
    if definitions and kind != "none":
        if not profile.tools:
            raise ConfigurationException(f"模型 {route.name} 不支持函数调用")
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": tool_schema(d.parameters),
                },
            }
            for d in definitions
        ]
        if kind == "function":
            body["tool_choice"] = {"type": "function", "function": {"name": name}}
        elif kind == "required":
            body["tool_choice"] = "required"
    apply_server_tools(body, server, route)

    output = config.output
    schema = output.response_schema
    if output.response_format == "json" or schema:
        if schema and profile.structured_output == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": strict_schema(schema),
                    "strict": True,
                },
            }
        else:
            if profile.structured_output != "none":
                body["response_format"] = {"type": "json_object"}
            add_instruction(
                messages,
                schema_instruction(schema) if schema else "请只输出一个 JSON 对象。",
            )
    if output.response_modalities:
        body["modalities"] = output.response_modalities
        media = config.media
        image_config = {
            "aspect_ratio": media.aspect_ratio,
            "image_size": media.resolution,
        }
        if image_config := {k: v for k, v in image_config.items() if v}:
            body["image_config"] = image_config
    return HttpCall("/chat/completions", json=deep_merge(body, config.params))


def _reasoning_parts(message: dict[str, Any]) -> list[ThoughtPart]:
    details = message.get("reasoning_details")
    if isinstance(details, list) and details:
        text = "\n".join(
            str(d.get("text") or d.get("summary") or "")
            for d in details
            if isinstance(d, dict)
        ).strip()
        return [
            ThoughtPart(
                thought_text=text,
                metadata=replay_metadata(PROTOCOL, {"reasoning_details": details}),
            )
        ]
    text = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(text, str) and text.strip():
        return [
            ThoughtPart(
                thought_text=text,
                metadata=replay_metadata(PROTOCOL, {"reasoning_content": text}),
            )
        ]
    return []


def parse_message(message: dict[str, Any]) -> list[Any]:
    """把 Chat Completions 的 message 解码为内容片段

    参数:
        message: choices[].message

    返回:
        list[Any]: 内容片段
    """
    if message.get("refusal"):
        raise ContentFilteredException(f"模型拒绝回答: {message['refusal']}")
    parts: list[Any] = _reasoning_parts(message)
    content = message.get("content")
    if isinstance(content, str):
        if not parts and (match := _THINK_TAG.match(content)):
            if thought := match.group(1).strip():
                parts.append(ThoughtPart(thought_text=thought))
            content = content[match.end() :]
        for match in _MARKDOWN_IMAGE.finditer(content):
            parts.append(output_image(re.sub(r"\s", "", match.group(1))))
        if (text := _MARKDOWN_IMAGE.sub("", content)).strip():
            parts.append(TextPart(text=text))
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(TextPart(text=item["text"]))
            elif item.get("type") == "image_url":
                parts.append(output_image(item["image_url"]["url"]))
    for image in message.get("images") or []:
        url = (image.get("image_url") or {}).get("url")
        if url:
            parts.append(output_image(url))
    for call in message.get("tool_calls") or []:
        if call.get("type", "function") != "function":
            continue
        function = call.get("function") or {}
        extra = {k: v for k, v in call.items() if k == "extra_content"}
        parts.append(
            ToolCallPart(
                id=call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                tool_name=function.get("name") or "",
                args=function.get("arguments") or "{}",
                metadata=replay_metadata(PROTOCOL, extra) if extra else None,
            )
        )
    if calls := message.get("server_tool_calls"):
        target = next((p for p in parts if isinstance(p, TextPart)), None)
        if target is None:
            target = ThoughtPart(thought_text="")
            parts.append(target)
        target.metadata = replay_metadata(PROTOCOL, {"server_tool_calls": calls})
    return parts


def _body_error(payload: dict[str, Any]) -> str | None:
    """提取状态码为 200 但响应体携带的错误信息

    参数:
        payload: 响应体

    返回:
        str | None: 错误描述，没有错误时为 None
    """
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    base = payload.get("base_resp") or {}
    if base.get("status_code"):
        return f"{base.get('status_msg')} ({base['status_code']})"
    return None


def parse_chat(route, request: ChatRequest, payload: dict[str, Any]) -> ChatResponse:
    """解码对话响应"""
    choices = payload.get("choices")
    if not choices:
        if message := _body_error(payload):
            raise UpstreamServerException(
                f"[{route.name}] 服务端返回错误: {message}",
                details={"response": str(payload)[:500]},
            )
        raise ResponseParseException(
            f"[{route.name}] 响应中没有 choices",
            details={"response": str(payload)[:500]},
        )
    choice = choices[0]
    parts = parse_message(choice.get("message") or {})
    if not parts and choice.get("finish_reason") == "content_filter":
        raise ContentFilteredException(f"[{route.name}] 响应被内容审核拦截")
    ensure_content(
        route,
        parts,
        choice.get("finish_reason") != "length",
        (payload.get("usage") or {}).get("completion_tokens"),
    )
    return ChatResponse(
        content_parts=parts,
        usage_info=payload.get("usage"),
        raw_response=payload,
        grounding_metadata=_grounding(payload),
    )


def _grounding(payload: dict[str, Any]) -> LLMGroundingMetadata | None:
    """提取网页搜索来源：智谱的响应顶层 web_search，以及 message.annotations
    中的 url_citation (MiMo 为平铺字段，OpenAI 写在 url_citation 子对象中)

    参数:
        payload: 响应体

    返回:
        LLMGroundingMetadata | None: 搜索溯源信息，没有搜索时为 None
    """
    sources: dict[str, LLMGroundingAttribution] = {}
    items = payload.get("web_search")
    for item in items if isinstance(items, list) else []:
        url = item.get("link") if isinstance(item, dict) else None
        if url and url not in sources:
            sources[url] = LLMGroundingAttribution(
                title=item.get("title"), uri=url, snippet=item.get("content")
            )
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    for note in message.get("annotations") or []:
        if not isinstance(note, dict) or note.get("type") != "url_citation":
            continue
        cite = note.get("url_citation") or note
        if (url := cite.get("url")) and url not in sources:
            sources[url] = LLMGroundingAttribution(
                title=cite.get("title"), uri=url, snippet=cite.get("summary")
            )
    if not sources:
        return None
    return LLMGroundingMetadata(grounding_attributions=list(sources.values()))


class OpenAIChatStream(ChatStream):
    """Chat Completions 流式累加器"""

    def __init__(self):
        self.payload: dict[str, Any] = {}
        self.message: dict[str, Any] = {"role": "assistant"}
        self.calls: dict[int, dict[str, Any]] = {}
        self.details: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None

    @classmethod
    def prepare(cls, call: HttpCall) -> None:
        if call.json is not None:
            call.json["stream"] = True
            call.json["stream_options"] = {"include_usage": True}
        call.stream = cls

    def _append(self, key: str, value: Any) -> bool:
        if not isinstance(value, str) or not value:
            return False
        self.message[key] = self.message.get(key, "") + value
        return True

    def _merge_detail(self, item: dict[str, Any]) -> str:
        merged = self.details.setdefault(item.get("index", len(self.details)), {})
        for key, value in item.items():
            if key in ("text", "summary", "signature", "data") and isinstance(
                value, str
            ):
                merged[key] = merged.get(key, "") + value
            else:
                merged[key] = value
        return str(item.get("text") or item.get("summary") or "")

    def _merge_call(self, call: dict[str, Any]) -> None:
        merged = self.calls.setdefault(
            call.get("index", len(self.calls)),
            {"type": "function", "function": {"name": "", "arguments": ""}},
        )
        for key in ("id", "type"):
            if call.get(key):
                merged[key] = call[key]
        function = call.get("function") or {}
        merged["function"]["name"] = merged["function"]["name"] or (
            function.get("name") or ""
        )
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = dump_arguments(arguments)
        merged["function"]["arguments"] += arguments or ""
        if extra := call.get("extra_content"):
            merged["extra_content"] = deep_merge(
                merged.get("extra_content") or {}, extra
            )

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, str]]:
        if message := stream_error(data):
            raise UpstreamServerException(
                f"服务端返回错误: {message}", details={"response": str(data)[:500]}
            )
        for key in ("id", "model", "created", "usage", "web_search", "base_resp"):
            if data.get(key):
                self.payload[key] = data[key]
        deltas: list[tuple[str, str]] = []
        for choice in data.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
            delta = choice.get("delta") or {}
            if self._append("content", delta.get("content")):
                deltas.append(("text", delta["content"]))
            reasoning = False
            for key in ("reasoning_content", "reasoning"):
                if self._append(key, delta.get(key)):
                    deltas.append(("thought", delta[key]))
                    reasoning = True
            for item in delta.get("reasoning_details") or []:
                if isinstance(item, dict):
                    text = self._merge_detail(item)
                    if text and not reasoning:
                        deltas.append(("thought", text))
            self._append("refusal", delta.get("refusal"))
            for key in ("images", "annotations"):
                if items := delta.get(key):
                    self.message.setdefault(key, []).extend(items)
            if calls := delta.get("server_tool_calls"):
                self.message["server_tool_calls"] = calls
            for call in delta.get("tool_calls") or []:
                if isinstance(call, dict):
                    self._merge_call(call)
        return deltas

    def result(self) -> dict[str, Any]:
        message = dict(self.message)
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
        if self.details:
            message["reasoning_details"] = [
                self.details[i] for i in sorted(self.details)
            ]
        received = len(message) > 1 or self.finish_reason
        choice = {"index": 0, "message": message, "finish_reason": self.finish_reason}
        return {**self.payload, "choices": [choice] if received else []}


STREAM = OpenAIChatStream
TASKS = {"chat": (build_chat, parse_chat), **openai_tasks()}
