"""
Anthropic Messages 协议

适用于 Claude 及兼容该格式的厂商 (DeepSeek、GLM、Kimi、MiniMax 的 anthropic 接口)。
思考块 (含 signature / redacted_thinking) 原样保存在片段元数据中回传。
"""

import ast
import json
import re
from typing import Any

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
    encodable_inputs,
    ensure_content,
    filter_modalities,
    native_tool_allowed,
    parse_arguments,
    replay_data,
    replay_metadata,
    resolve_tool_choice,
    split_tools,
    stream_error,
    tool_output_text,
)
from .schema import anthropic_schema, schema_instruction, tool_schema

PROTOCOL = "anthropic"

_DEFAULT_MAX_TOKENS = 16000
_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
_LEGACY_CODE_BETA = "code-execution-2025-08-25"


def _source(part: Any, data: str, default_mime: str) -> dict[str, Any]:
    return {
        "type": "base64",
        "media_type": part.mime_type or default_mime,
        "data": data,
    }


async def _block(part: Any) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text} if part.text else None
    if isinstance(part, ImagePart | FilePart):
        kind = "image" if isinstance(part, ImagePart) else "document"
        if part.url and part.url.startswith(("http://", "https://")):
            return {"type": kind, "source": {"type": "url", "url": part.url}}
        default = "image/png" if kind == "image" else "application/pdf"
        return {
            "type": kind,
            "source": _source(part, await part.get_base64_data(), default),
        }
    return None


def _assistant_blocks(message: AssistantMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, ThoughtPart):
            if isinstance(data := replay_data(part, PROTOCOL), dict):
                blocks.append(data)
        elif isinstance(part, TextPart) and part.text:
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ToolCallPart):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.id,
                    "name": part.tool_name,
                    "input": parse_arguments(part.args),
                }
            )
    return blocks


async def _build_messages(request: ChatRequest, route) -> tuple[list[str], list]:
    messages = filter_modalities(
        request.messages, route.profile, route.name, encodable_inputs("anthropic")
    )
    system: list[str] = []
    result: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})

    for message in messages:
        if isinstance(message, SystemMessage):
            if text := "".join(p.text for p in message.content):
                system.append(text)
        elif isinstance(message, ToolMessage):
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.tool_call_id,
                        "content": tool_output_text(item.output),
                    }
                    for item in message.tool_returns
                ],
            )
        elif isinstance(message, AssistantMessage):
            append("assistant", _assistant_blocks(message))
        else:
            blocks = [b for p in message.content if (b := await _block(p))]
            append("user", blocks)
    return system, result


def _server_tool(tool: Any, adaptive: bool) -> dict[str, Any] | None:
    type_id = getattr(tool, "type_id", "")
    if type_id == "web_search":
        version = "web_search_20260209" if adaptive else "web_search_20250305"
        return {"type": version, "name": "web_search", "max_uses": 5}
    if type_id == "code_execution":
        version = "code_execution_20260521" if adaptive else "code_execution_20250825"
        return {"type": version, "name": "code_execution"}
    return None


def _thinking(body: dict[str, Any], config, profile, max_tokens: int | None) -> bool:
    """写入推理配置，返回是否开启了思考"""
    effort = map_reasoning_effort(config.common.reasoning_effort, profile)
    if profile.reasoning_param == "adaptive":
        if effort == "none":
            body["thinking"] = {"type": "disabled"}
            return False
        body["thinking"] = {"type": "adaptive", "display": "summarized"}
        if effort:
            body.setdefault("output_config", {})["effort"] = effort
        return True
    if profile.reasoning_param != "budget" or not effort or effort == "none":
        return False
    budget = _BUDGETS.get(effort, 32000)
    if max_tokens is None:
        body["max_tokens"] = max(body["max_tokens"], budget + 8192)
    elif max_tokens <= 2048:
        return False
    else:
        budget = min(budget, max_tokens - 1024)
    body["thinking"] = {"type": "enabled", "budget_tokens": max(budget, 1024)}
    return True


async def build_chat(route, request: ChatRequest) -> HttpCall:
    """编码对话请求"""
    config = route.generation_config(request.config)
    profile = route.profile
    common = config.common
    adaptive = profile.reasoning_param == "adaptive"
    system, messages = await _build_messages(request, route)
    max_tokens = common.max_tokens
    if max_tokens and profile.max_output_tokens:
        max_tokens = min(max_tokens, profile.max_output_tokens)
    body: dict[str, Any] = {
        "model": route.model.name,
        "max_tokens": max_tokens or profile.max_output_tokens or _DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    thinking = _thinking(body, config, profile, max_tokens)
    if profile.sampling and not (thinking and not adaptive):
        for key in ("temperature", "top_p", "top_k"):
            if (value := getattr(common, key)) is not None:
                body[key] = value
    if common.stop is not None:
        body["stop_sequences"] = (
            common.stop if isinstance(common.stop, list) else [common.stop]
        )

    headers: dict[str, str] = {}
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
                "name": d.name,
                "description": d.description,
                "input_schema": tool_schema(d.parameters),
            }
            for d in definitions
        )
        if kind in ("required", "function"):
            if adaptive or thinking:
                system.append(
                    f"本轮必须调用工具 {name}。"
                    if name
                    else "本轮必须调用至少一个工具。"
                )
            elif name:
                body["tool_choice"] = {"type": "tool", "name": name}
            else:
                body["tool_choice"] = {"type": "any"}
    server_types = set()
    for tool in server:
        type_id = getattr(tool, "type_id", "")
        payload = deepseek_server_tool(route, type_id) or _server_tool(tool, adaptive)
        if payload and native_tool_allowed(route, type_id):
            server_types.add(type_id)
            tools.append(payload)
        else:
            logger.warning(
                f"模型 {route.name} 不支持或未确认支持内置工具 {type_id}，已忽略"
            )
    if {"web_search", "code_execution"} <= server_types and adaptive:
        tools = [t for t in tools if t.get("name") != "code_execution"]
    elif "code_execution" in server_types and not adaptive:
        headers["anthropic-beta"] = _LEGACY_CODE_BETA
    if tools:
        body["tools"] = tools

    schema = config.output.response_schema
    if config.output.response_format == "json" or schema:
        if schema and profile.structured_output == "json_schema":
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": anthropic_schema(schema),
            }
        else:
            system.append(
                schema_instruction(schema) if schema else "请只输出一个 JSON 对象。"
            )
    if system:
        body["system"] = "\n\n".join(system)
    return HttpCall(
        "/v1/messages", json=deep_merge(body, config.params), headers=headers
    )


_SEARCH_TOOLS = ("web_search", "web_search_prime", "deepseek_web_search")
"""服务端网页搜索的工具名，web_search_prime 为智谱的写法"""
_STOP_UNFINISHED = ("max_tokens", "pause_turn", "model_context_window_exceeded")
_RESULT_TEXT = re.compile(
    r"""['"]title['"]:\s*(['"])(.*?)\1,\s*['"](?:link|url)['"]:\s*(['"])(.*?)\3"""
)


def _search_results(value: Any) -> list[dict[str, Any]]:
    """从搜索结果中找出带链接的条目

    智谱的结果为字符串，可能是 Python 字面量与 JSON 拼接而成，无法整体解析时按文本提取
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [
                    {"title": m.group(2), "link": m.group(4)}
                    for m in _RESULT_TEXT.finditer(value)
                ]
    if isinstance(value, dict):
        if value.get("url") or value.get("link"):
            return [value]
        return [r for item in value.values() for r in _search_results(item)]
    if isinstance(value, list):
        return [r for item in value for r in _search_results(item)]
    return []


def _grounding(content: list[dict[str, Any]]) -> LLMGroundingMetadata | None:
    """提取网页搜索的查询词与结果来源

    参数:
        content: 响应的 content 列表

    返回:
        LLMGroundingMetadata | None: 搜索溯源信息，没有搜索时为 None
    """
    queries: list[str] = []
    sources: dict[str, LLMGroundingAttribution] = {}
    search_ids: set[str] = set()
    for block in content:
        kind = block.get("type")
        if kind == "server_tool_use" and block.get("name") in _SEARCH_TOOLS:
            search_ids.add(block.get("id") or "")
            args = block.get("input") or {}
            query = args.get("query") or args.get("search_query")
            if isinstance(query, str) and query not in queries:
                queries.append(query)
            for query in args.get("queries") or []:
                if isinstance(query, str) and query not in queries:
                    queries.append(query)
        elif kind == "web_search_tool_result" or (
            kind == "tool_result" and block.get("tool_use_id") in search_ids
        ):
            for item in _search_results(block.get("content")):
                url = item.get("url") or item.get("link")
                if url and url not in sources:
                    sources[url] = LLMGroundingAttribution(
                        title=item.get("title"), uri=url
                    )
    if not queries and not sources:
        return None
    return LLMGroundingMetadata(
        web_search_queries=queries or None,
        grounding_attributions=list(sources.values()) or None,
    )


def parse_chat(route, request: ChatRequest, payload: dict[str, Any]) -> ChatResponse:
    """解码对话响应"""
    content = payload.get("content")
    if not isinstance(content, list):
        raise ResponseParseException(
            f"[{route.name}] 响应中没有 content",
            details={"response": str(payload)[:500]},
        )
    if payload.get("stop_reason") == "refusal":
        details = payload.get("stop_details") or {}
        raise ContentFilteredException(
            f"[{route.name}] 模型拒绝回答: {details.get('category') or ''}",
            details={"stop_details": details},
        )
    parts: list[Any] = []
    for block in content:
        kind = block.get("type")
        if kind in ("thinking", "redacted_thinking"):
            parts.append(
                ThoughtPart(
                    thought_text=block.get("thinking") or "",
                    metadata=replay_metadata(PROTOCOL, block),
                )
            )
        elif kind in {
            "server_tool_use",
            "web_search_tool_result",
            "web_fetch_tool_result",
        } and deepseek_server_tool(route, "web_search"):
            parts.append(
                ThoughtPart(thought_text="", metadata=replay_metadata(PROTOCOL, block))
            )
        elif kind == "text" and block.get("text"):
            parts.append(TextPart(text=block["text"]))
        elif kind == "tool_use":
            parts.append(
                ToolCallPart(
                    id=block.get("id") or "",
                    tool_name=block.get("name") or "",
                    args=block.get("input") or {},
                )
            )
    ensure_content(
        route,
        parts,
        payload.get("stop_reason") not in _STOP_UNFINISHED,
        (payload.get("usage") or {}).get("output_tokens"),
    )
    return ChatResponse(
        content_parts=parts,
        usage_info=payload.get("usage"),
        raw_response=payload,
        grounding_metadata=_grounding(content),
    )


class AnthropicStream(ChatStream):
    """Messages 流式累加器"""

    def __init__(self):
        self.message: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.inputs: dict[int, str] = {}

    @classmethod
    def prepare(cls, call: HttpCall) -> None:
        if call.json is not None:
            call.json["stream"] = True
        call.stream = cls

    def _delta(self, index: int, delta: dict[str, Any]) -> list[tuple[str, str]]:
        block = self.blocks.setdefault(index, {"type": "text", "text": ""})
        kind = delta.get("type")
        if kind == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
            return [("text", delta.get("text", ""))]
        if kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            return [("thought", delta.get("thinking", ""))]
        if kind == "signature_delta":
            block["signature"] = block.get("signature", "") + delta.get("signature", "")
        elif kind == "input_json_delta":
            self.inputs[index] = self.inputs.get(index, "") + delta.get(
                "partial_json", ""
            )
        elif kind == "citations_delta" and delta.get("citation"):
            block.setdefault("citations", []).append(delta["citation"])
        return []

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, str]]:
        kind = data.get("type") or event
        index = int(data.get("index") or 0)
        if kind == "error":
            raise UpstreamServerException(
                f"服务端返回错误: {stream_error(data)}",
                details={"response": str(data)[:500]},
            )
        if kind == "message_start":
            self.message = dict(data.get("message") or {})
        elif kind == "content_block_start":
            self.blocks[index] = dict(data.get("content_block") or {})
        elif kind == "content_block_delta":
            return self._delta(index, data.get("delta") or {})
        elif kind == "content_block_stop" and index in self.inputs:
            self.blocks[index]["input"] = parse_arguments(self.inputs.pop(index))
        elif kind == "message_delta":
            self.message.update(data.get("delta") or {})
            usage = self.message.setdefault("usage", {})
            usage.update(data.get("usage") or {})
        return []

    def result(self) -> dict[str, Any]:
        content = [self.blocks[i] for i in sorted(self.blocks)]
        return {**self.message, "content": content} if self.message else {}


STREAM = AnthropicStream
TASKS = {"chat": (build_chat, parse_chat)}
