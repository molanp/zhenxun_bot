"""
OpenAI Responses 协议

以无状态方式调用 (store=false)，推理项通过 encrypted_content 回传。
"""

from typing import Any

from zhenxun.services.ai.config.presets import detect_preset
from zhenxun.services.ai.core.exceptions import (
    ConfigurationException,
    ContentFilteredException,
    ResponseParseException,
    UpstreamServerException,
)
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
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
)
from zhenxun.services.ai.core.options import deep_merge
from zhenxun.services.ai.llm.profile import map_reasoning_effort
from zhenxun.services.ai.utils.logger import log_llm as logger

from . import (
    ChatStream,
    HttpCall,
    deepseek_server_tool,
    dump_arguments,
    encodable_inputs,
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

PROTOCOL = "openai_responses"

_SOURCE_VENDORS = (None, "openai", "kimi")
"""文档声明支持 include web_search_call.action.sources 的厂商，None 为中转站"""


async def _user_content(parts: list[Any]) -> list[dict[str, Any]]:
    content = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            url = await media_url(part, "image/png")
            content.append({"type": "input_image", "image_url": url})
        elif isinstance(part, FilePart):
            if part.url and not part.url.startswith("data:"):
                content.append({"type": "input_file", "file_url": part.url})
            else:
                filename = (part.metadata or {}).get("filename") or (
                    part.path.name if part.path else "file.pdf"
                )
                content.append(
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": await part.get_data_uri("application/pdf"),
                    }
                )
    return content


def _assistant_items(message: AssistantMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, ThoughtPart):
            if isinstance(data := replay_data(part, PROTOCOL), dict):
                items.append(data)
        elif isinstance(part, TextPart) and part.text:
            items.append({"role": "assistant", "content": part.text})
        elif isinstance(part, ToolCallPart):
            items.append(
                {
                    "type": "function_call",
                    "call_id": part.id,
                    "name": part.tool_name,
                    "arguments": dump_arguments(part.args),
                }
            )
    return items


async def _build_input(request: ChatRequest, route) -> tuple[str, list[dict]]:
    messages = filter_modalities(
        request.messages,
        route.profile,
        route.name,
        encodable_inputs("openai_responses"),
    )
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            instructions.append("".join(p.text for p in message.content))
        elif isinstance(message, ToolMessage):
            items.extend(
                {
                    "type": "function_call_output",
                    "call_id": item.tool_call_id,
                    "output": tool_output_text(item.output),
                }
                for item in message.tool_returns
            )
        elif isinstance(message, AssistantMessage):
            items.extend(_assistant_items(message))
        elif content := await _user_content(list(message.content)):
            items.append({"role": "user", "content": content})
    return "\n\n".join(t for t in instructions if t), items


def _server_tool(tool: Any) -> dict[str, Any] | None:
    type_id = getattr(tool, "type_id", "")
    if type_id == "web_search":
        return {"type": "web_search"}
    if type_id == "code_execution":
        return {"type": "code_interpreter", "container": {"type": "auto"}}
    if type_id == "image_generation":
        return {"type": "image_generation"}
    if type_id == "computer_use":
        return {
            "type": "computer_use_preview",
            "display_width": tool.display_width_px,
            "display_height": tool.display_height_px,
            "environment": "browser",
        }
    if type_id == "file_search":
        if ids := tool.metadata.get("vector_store_ids"):
            return {"type": "file_search", "vector_store_ids": ids}
    return None


async def build_chat(route, request: ChatRequest) -> HttpCall:
    """编码对话请求"""
    config = route.generation_config(request.config)
    profile = route.profile
    common = config.common
    instructions, items = await _build_input(request, route)
    body: dict[str, Any] = {"model": route.model.name, "input": items, "store": False}
    if profile.sampling:
        if common.temperature is not None:
            body["temperature"] = common.temperature
        if common.top_p is not None:
            body["top_p"] = common.top_p
    max_tokens = common.max_tokens
    if max_tokens and profile.max_output_tokens:
        max_tokens = min(max_tokens, profile.max_output_tokens)
    if max_tokens:
        body["max_output_tokens"] = max_tokens
    if profile.reasoning_levels or profile.reasoning_off:
        body["include"] = ["reasoning.encrypted_content"]
        effort = map_reasoning_effort(common.reasoning_effort, profile)
        reasoning: dict[str, Any] = {"effort": effort} if effort else {}
        if effort != "none":
            reasoning["summary"] = "auto"
        body["reasoning"] = reasoning

    definitions, server = await split_tools(request.tools)
    kind, allowed, name = resolve_tool_choice(request.tool_choice, config.tools.mode)
    if allowed:
        definitions = [d for d in definitions if d.name in allowed]
    tools: list[dict[str, Any]] = []
    if definitions and kind != "none":
        if not profile.tools:
            raise ConfigurationException(f"模型 {route.name} 不支持函数调用")
        tools.extend(
            {
                "type": "function",
                "name": d.name,
                "description": d.description,
                "parameters": tool_schema(d.parameters),
                "strict": False,
            }
            for d in definitions
        )
        if kind == "function":
            body["tool_choice"] = {"type": "function", "name": name}
        elif kind == "required":
            body["tool_choice"] = "required"
    for tool in server:
        type_id = getattr(tool, "type_id", "")
        payload = deepseek_server_tool(route, type_id) or _server_tool(tool)
        if payload and native_tool_allowed(route, type_id):
            tools.append(payload)
            vendor = detect_preset(route.base_url) or route.provider.preset
            if (
                type_id == "web_search"
                and vendor in _SOURCE_VENDORS
                and vendor != "deepseek"
            ):
                body.setdefault("include", []).append("web_search_call.action.sources")
        else:
            logger.warning(
                f"模型 {route.name} 不支持或未确认支持内置工具 {type_id}，已忽略"
            )
    if tools:
        body["tools"] = tools

    schema = config.output.response_schema
    if config.output.response_format == "json" or schema:
        if schema and profile.structured_output == "json_schema":
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "schema": strict_schema(schema),
                    "strict": True,
                }
            }
        else:
            if profile.structured_output != "none":
                body["text"] = {"format": {"type": "json_object"}}
            hint = schema_instruction(schema) if schema else "请只输出一个 JSON 对象。"
            instructions = f"{instructions}\n\n{hint}" if instructions else hint
    if instructions:
        body["instructions"] = instructions
    return HttpCall("/responses", json=deep_merge(body, config.params))


def _grounding(output: list[dict[str, Any]]) -> LLMGroundingMetadata | None:
    """提取网页搜索的查询词与来源，来源取正文引用与 action.sources，
    忽略 DeepSeek 附加的 ws_call_id 回放标记

    参数:
        output: 响应的 output 列表

    返回:
        LLMGroundingMetadata | None: 搜索溯源信息，没有搜索时为 None
    """
    queries: list[str] = []
    sources: dict[str, LLMGroundingAttribution] = {}

    def add_source(url: Any, title: Any = None) -> None:
        if isinstance(url, str) and url and url not in sources:
            sources[url] = LLMGroundingAttribution(title=title, uri=url)

    for item in output:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                for note in content.get("annotations") or []:
                    if note.get("type") == "url_citation":
                        add_source(note.get("url"), note.get("title"))
    for item in output:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        for query in [action.get("query"), *(action.get("queries") or [])]:
            if (
                isinstance(query, str)
                and query
                and not query.startswith("ws_call_id=")
                and query not in queries
            ):
                queries.append(query)
        for source in action.get("sources") or []:
            if isinstance(source, dict):
                add_source(source.get("url"), source.get("title"))
            else:
                add_source(source)
    if not queries and not sources:
        return None
    return LLMGroundingMetadata(
        web_search_queries=queries or None,
        grounding_attributions=list(sources.values()) or None,
    )


def parse_chat(route, request: ChatRequest, payload: dict[str, Any]) -> ChatResponse:
    """解码对话响应"""
    output = payload.get("output")
    if not isinstance(output, list):
        raise ResponseParseException(
            f"[{route.name}] 响应中没有 output",
            details={"response": str(payload)[:500]},
        )
    parts: list[Any] = []
    for item in output:
        kind = item.get("type")
        if kind == "reasoning":
            text = "\n".join(
                s.get("text", "") for s in item.get("summary") or []
            ).strip()
            replay = {
                k: item[k]
                for k in (
                    "type",
                    "id",
                    "summary",
                    "encrypted_content",
                    "content",
                    "signature",
                )
                if k in item
            }
            parts.append(
                ThoughtPart(
                    thought_text=text, metadata=replay_metadata(PROTOCOL, replay)
                )
            )
        elif kind == "web_search_call" and deepseek_server_tool(route, "web_search"):
            parts.append(
                ThoughtPart(thought_text="", metadata=replay_metadata(PROTOCOL, item))
            )
        elif kind == "message":
            for content in item.get("content") or []:
                if content.get("type") == "refusal":
                    raise ContentFilteredException(
                        f"模型拒绝回答: {content.get('refusal')}"
                    )
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(TextPart(text=content["text"]))
        elif kind == "function_call":
            parts.append(
                ToolCallPart(
                    id=item.get("call_id") or item.get("id") or "",
                    tool_name=item.get("name") or "",
                    args=item.get("arguments") or "{}",
                )
            )
        elif kind == "image_generation_call" and item.get("result"):
            parts.append(output_image(item["result"]))
    incomplete = (payload.get("incomplete_details") or {}).get("reason")
    if incomplete == "content_filter" and not any(
        isinstance(p, TextPart | ToolCallPart) for p in parts
    ):
        raise ContentFilteredException(f"[{route.name}] 响应被内容审核拦截")
    ensure_content(
        route,
        parts,
        incomplete is None,
        (payload.get("usage") or {}).get("output_tokens"),
    )
    return ChatResponse(
        content_parts=parts,
        usage_info=payload.get("usage"),
        raw_response=payload,
        grounding_metadata=_grounding(output),
    )


class ResponsesStream(ChatStream):
    """Responses 流式累加器：完整响应在 response.completed 事件中"""

    def __init__(self):
        self.payload: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []

    @classmethod
    def prepare(cls, call: HttpCall) -> None:
        if call.json is not None:
            call.json["stream"] = True
        call.stream = cls

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, str]]:
        kind = data.get("type") or event
        if kind == "response.output_text.delta":
            return [("text", str(data.get("delta") or ""))]
        if kind == "response.reasoning_summary_text.delta":
            return [("thought", str(data.get("delta") or ""))]
        if kind == "response.output_item.done" and isinstance(data.get("item"), dict):
            self.items.append(data["item"])
        elif kind in ("response.created", "response.completed", "response.incomplete"):
            self.payload = data.get("response") or self.payload
        elif kind in ("response.failed", "error"):
            response = data.get("response") or {}
            message = (
                stream_error(response) or stream_error(data) or data.get("message")
            )
            raise UpstreamServerException(
                f"服务端返回错误: {message}", details={"response": str(data)[:500]}
            )
        return []

    def result(self) -> dict[str, Any]:
        if not self.payload.get("output") and self.items:
            return {**self.payload, "output": self.items}
        return self.payload


STREAM = ResponsesStream
TASKS = {"chat": (build_chat, parse_chat), **openai_tasks()}
