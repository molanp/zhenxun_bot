"""
接口协议编解码

每个协议模块提供 TASKS: {任务: (build, parse)}。
build 把统一请求编码为 HttpCall，parse 把响应解码为统一响应。
"""

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib import import_module
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

from zhenxun.configs.path_config import TEMP_PATH
from zhenxun.services.ai.config.presets import PRESETS, detect_preset
from zhenxun.services.ai.core.exceptions import (
    ConfigurationException,
    EmptyResponseException,
)
from zhenxun.services.ai.core.messages import (
    AudioPart,
    FilePart,
    ImagePart,
    LLMMessage,
    TextPart,
    ThoughtPart,
    VideoPart,
)
from zhenxun.services.ai.core.models import ModelProfile, ToolChoice, ToolDefinition
from zhenxun.services.ai.utils.logger import log_llm as logger
from zhenxun.utils.pydantic_compat import model_copy, model_validate

if TYPE_CHECKING:
    from zhenxun.services.ai.llm.client import Route


@dataclass
class HttpCall:
    """一次编码完成的 HTTP 请求"""

    path: str
    """追加在 base_url 之后的路径"""
    json: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    base_url: str | None = None
    """覆盖线路的 base_url"""
    raw: bool = False
    """响应为二进制数据"""
    stream: "type[ChatStream] | None" = None
    """流式请求的累加器类型，为空时为普通请求"""


class ChatStream:
    """流式对话响应的累加器：逐个接收 SSE 事件，给出文本增量，
    结束后还原为与非流式相同的响应体，交给协议原有的解码函数"""

    @classmethod
    def prepare(cls, call: HttpCall) -> None:
        """把请求改为流式请求

        参数:
            call: 编码后的请求
        """
        raise NotImplementedError

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, str]]:
        """接收一个 SSE 事件

        参数:
            event: 事件名，没有 event 字段时为空
            data: 事件数据

        返回:
            list[tuple[str, str]]: 增量 (text 正文 / thought 思考, 文本)
        """
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        """还原出的非流式响应体

        返回:
            dict[str, Any]: 响应体
        """
        raise NotImplementedError


def stream_error(data: dict[str, Any]) -> str | None:
    """提取流式事件中携带的错误信息

    参数:
        data: 事件数据

    返回:
        str | None: 错误描述，没有错误时为空
    """
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error) if error else None


Build = Callable[["Route", Any], Awaitable[HttpCall]]
Parse = Callable[["Route", Any, Any], Any]

_MODULES = {
    "openai_chat": "openai_chat",
    "openai_responses": "openai_responses",
    "anthropic": "anthropic",
    "gemini": "gemini",
}


def get_task(protocol: str, task: str) -> tuple[Build, Parse]:
    """获取协议对指定任务的编解码函数

    参数:
        protocol: 接口协议
        task: 任务类型

    返回:
        tuple[Build, Parse]: 编码与解码函数
    """
    module = import_module(f"{__name__}.{_MODULES[protocol]}")
    if task not in module.TASKS:
        raise ConfigurationException(f"{protocol} 协议不支持 {task} 任务")
    return module.TASKS[task]


def get_stream(protocol: str) -> type[ChatStream] | None:
    """获取协议的流式对话累加器

    参数:
        protocol: 接口协议

    返回:
        type[ChatStream] | None: 累加器类型，协议不支持流式时为空
    """
    module = import_module(f"{__name__}.{_MODULES[protocol]}")
    return getattr(module, "STREAM", None)


_PLACEHOLDERS = {
    ImagePart: ("image", "<图片>"),
    AudioPart: ("audio", "<音频>"),
    VideoPart: ("video", "<视频>"),
    FilePart: ("file", "<文件>"),
}


_MEDIA_INPUTS = frozenset({"image", "audio", "video", "file"})
_PROTOCOL_INPUTS = {
    "openai_responses": frozenset({"image", "file"}),
    "anthropic": frozenset({"image", "file"}),
}


def encodable_inputs(protocol: str) -> frozenset[str]:
    """协议能够编码的媒体输入模态

    参数:
        protocol: 协议名

    返回:
        frozenset[str]: 媒体模态 (image / audio / video / file)
    """
    return _PROTOCOL_INPUTS.get(protocol, _MEDIA_INPUTS)


def filter_modalities(
    messages: list[LLMMessage],
    profile: ModelProfile,
    model_name: str,
    encodable: frozenset[str] = _MEDIA_INPUTS,
) -> list[LLMMessage]:
    """把模型不支持或协议无法编码的输入模态替换为文字占位

    参数:
        messages: 原始消息
        profile: 模型能力档案
        model_name: 模型全名，用于日志
        encodable: 协议能够编码的媒体模态

    返回:
        list[LLMMessage]: 处理后的消息，未改动的消息保持原对象
    """

    def unusable(modality: str | None) -> bool:
        return bool(modality) and (
            modality not in encodable or not profile.accepts_input(modality)
        )

    warned: set[str] = set()
    result = []
    for message in messages:
        if not any(
            unusable(_PLACEHOLDERS.get(type(part), (None,))[0])
            for part in message.content
        ):
            result.append(message)
            continue
        content = []
        for part in message.content:
            modality, placeholder = _PLACEHOLDERS.get(type(part), (None, None))
            if modality and unusable(modality):
                if modality not in warned:
                    warned.add(modality)
                    logger.warning(
                        f"模型 {model_name} 不支持 {modality} 输入，已替换为占位文字"
                    )
                content.append(TextPart(text=placeholder))
            else:
                content.append(part)
        result.append(model_copy(message, update={"content": content}))
    return result


_MEDIA_TYPES = tuple(_PLACEHOLDERS)


def has_media(message: LLMMessage) -> bool:
    """消息是否包含图片、语音、视频等媒体片段

    参数:
        message: 消息

    返回:
        bool: 是否包含媒体
    """
    return isinstance(message.content, list) and any(
        isinstance(p, _MEDIA_TYPES) for p in message.content
    )


def drop_tool_media(messages: list[LLMMessage]) -> list[LLMMessage]:
    """去掉工具消息中的媒体片段，用于写入持久化历史

    参数:
        messages: 消息列表

    返回:
        list[LLMMessage]: 处理后的消息，未改动的消息保持原对象
    """
    return [
        model_copy(
            message,
            update={
                "content": [
                    p for p in message.content if not isinstance(p, _MEDIA_TYPES)
                ]
            },
        )
        if message.role == "tool"
        and any(isinstance(p, _MEDIA_TYPES) for p in message.content)
        else message
        for message in messages
    ]


def lift_tool_media(messages: list[LLMMessage]) -> list[LLMMessage]:
    """把工具返回的媒体移到这组连续工具消息之后的一条用户消息中，
    使各协议都能把媒体交给模型

    参数:
        messages: 消息列表

    返回:
        list[LLMMessage]: 处理后的消息，无工具媒体时为原列表
    """
    stripped = drop_tool_media(messages)
    if all(a is b for a, b in zip(messages, stripped)):
        return messages
    result: list[LLMMessage] = []
    pending: list = []
    for original, message in zip(messages, stripped):
        if message.role != "tool" and pending:
            result.append(LLMMessage.user(pending))
            pending = []
        if message is not original:
            name = message.tool_returns[0].tool_name
            pending.append(TextPart(text=f"[工具 {name} 返回的内容]"))
            pending.extend(p for p in original.content if isinstance(p, _MEDIA_TYPES))
        result.append(message)
    if pending:
        result.append(LLMMessage.user(pending))
    return result


_PROTOCOL_VENDORS = {
    "openai_responses": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
}
"""协议的原厂商，网页搜索以外的内置工具只在原厂商官方接口上默认发送"""


def native_tool_allowed(route: "Route", type_id: str) -> bool:
    """判断是否向线路发送服务端内置工具

    Chat Completions 协议只能发送有厂商写法的网页搜索；其余已确认支持或不支持的
    按确认结果；未确认时只在厂商官方接口上发送：网页搜索要求厂商在当前协议下提供，
    其他内置工具要求是协议的原厂商。中转站需核验通过或手动强制开启。

    参数:
        route: 线路
        type_id: 内置工具类型

    返回:
        bool: 是否发送
    """
    profile = route.profile
    if type_id in profile.native_tools_off:
        return False
    vendor = detect_preset(route.base_url) or route.provider.preset
    if vendor == "deepseek" and type_id in {"web_search", "url_context"}:
        return type_id in profile.native_tools or (
            detect_preset(route.base_url) == "deepseek"
            and route.model.name
            in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}
        )
    if route.protocol == "openai_chat":
        vendor = detect_preset(route.base_url) or route.provider.preset
        searchable = "openai_chat" in PRESETS.get(vendor or "", {}).get(
            "web_search", ()
        )
        if type_id != "web_search" or not searchable:
            return False
    if type_id in profile.native_tools:
        return True
    vendor = detect_preset(route.base_url)
    if vendor is None:
        return False
    if type_id == "web_search":
        return route.protocol in PRESETS[vendor].get("web_search", ())
    return _PROTOCOL_VENDORS.get(route.protocol) == vendor


def deepseek_server_tool(route: "Route", type_id: str) -> dict[str, str] | None:
    """编码 DeepSeek 的原生搜索与读链接工具。"""
    if (detect_preset(route.base_url) or route.provider.preset) != "deepseek":
        return None
    kind = {
        "web_search": "deepseek_web_search",
        "url_context": "deepseek_read_url",
    }.get(type_id)
    if not kind:
        return None
    return (
        {"type": kind, "name": kind}
        if route.protocol == "anthropic"
        else {"type": kind}
    )


def ensure_content(
    route: "Route", parts: list[Any], finished: bool, output_tokens: int | None = None
) -> None:
    """检查对话响应是否有内容：正常结束却没有任何内容，或因长度截断而一个输出
    token 也没有生成时，视为空回复

    参数:
        route: 线路
        parts: 解析出的内容片段
        finished: 是否正常结束
        output_tokens: 输出 token 数（含思考），未知时为 None

    异常:
        EmptyResponseException: 只有空文本或思考内容
    """
    if not all(
        isinstance(p, ThoughtPart) or (isinstance(p, TextPart) and not p.text.strip())
        for p in parts
    ):
        return
    if finished:
        raise EmptyResponseException(f"[{route.name}] 模型返回了空回复")
    if output_tokens == 0:
        raise EmptyResponseException(
            f"[{route.name}] 输出长度上限过小，模型没有生成任何内容，"
            "请检查生成参数 max_tokens"
        )


def tool_output_text(output: Any) -> str:
    """把工具返回值转换为文本"""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except Exception:
        return str(output)


def parse_arguments(args: dict[str, Any] | str) -> dict[str, Any]:
    """把工具调用参数解析为字典，无法解析时保留原文"""
    if isinstance(args, dict):
        return args
    try:
        value = json.loads(args or "{}")
    except json.JSONDecodeError:
        return {"__raw__": args}
    return value if isinstance(value, dict) else {"value": value}


def dump_arguments(args: dict[str, Any] | str) -> str:
    """把工具调用参数序列化为 JSON 字符串"""
    if isinstance(args, str):
        return args or "{}"
    return json.dumps(args, ensure_ascii=False)


def _definition_from_dict(tool: dict[str, Any]) -> ToolDefinition | None:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    if not function.get("name"):
        return None
    return ToolDefinition(
        name=function["name"],
        description=function.get("description") or "",
        parameters=function.get("parameters") or {"type": "object", "properties": {}},
    )


async def split_tools(tools: Any) -> tuple[list[ToolDefinition], list[Any]]:
    """把请求中的工具拆分为函数工具定义与服务端内置工具

    参数:
        tools: 工具对象、ToolDefinition 或字典组成的列表 (或字典)

    返回:
        tuple[list[ToolDefinition], list[Any]]: 函数工具定义、服务端内置工具
    """
    if not tools:
        return [], []
    items = list(tools.values()) if isinstance(tools, dict) else list(tools)
    definitions: list[ToolDefinition | None] = []
    pending = []
    server = []
    for tool in items:
        if getattr(tool, "execution_side", "client") == "server":
            server.append(tool)
        elif isinstance(tool, ToolDefinition):
            definitions.append(tool)
        elif isinstance(tool, dict):
            definitions.append(_definition_from_dict(tool))
        elif hasattr(tool, "get_definition"):
            pending.append(tool.get_definition())
    if pending:
        definitions.extend(await asyncio.gather(*pending))
    return [d for d in definitions if d], server


def resolve_tool_choice(
    tool_choice: Any, mode: str
) -> tuple[str, list[str] | None, str | None]:
    """统一工具选择策略

    参数:
        tool_choice: 请求中的 tool_choice
        mode: GenerationConfig.tools.mode

    返回:
        tuple: (auto / none / required / function, 允许的函数名, 指定的函数名)
    """
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get(
            "name"
        )
        if name:
            return "function", None, name
        tool_choice = model_validate(ToolChoice, tool_choice)
    if isinstance(tool_choice, str):
        tool_choice = ToolChoice(mode=tool_choice)  # type: ignore[arg-type]
    if isinstance(tool_choice, ToolChoice):
        kind = {"any": "required"}.get(tool_choice.mode, tool_choice.mode)
        allowed = tool_choice.allowed_function_names
    else:
        kind = {"ANY": "required", "NONE": "none"}.get(mode, "auto")
        allowed = None
    if allowed and len(allowed) == 1 and kind == "required":
        return "function", None, allowed[0]
    return kind, allowed, None


def save_image(data: bytes, suffix: str = "png") -> bytes | Path:
    """输出图片超过 2MB 时落盘，返回文件路径，否则返回原始字节"""
    if len(data) <= 2 * 1024 * 1024:
        return data
    save_dir = TEMP_PATH / "llm"
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{uuid.uuid4()}.{suffix}"
    path.write_bytes(data)
    logger.info(f"图片数据过大 ({len(data)} bytes)，已保存到临时文件: {path}")
    return path.resolve()


def decode_data_uri(uri: str) -> tuple[bytes, str]:
    """解析 data URI

    参数:
        uri: data:mime;base64,... 格式的字符串

    返回:
        tuple[bytes, str]: 原始字节与 MIME 类型
    """
    header, _, data = uri.partition(",")
    mime = header[5:].split(";")[0] or "application/octet-stream"
    return base64.b64decode(data), mime


def output_image(source: str | bytes, mime: str = "image/png") -> ImagePart:
    """把模型输出的图片 (URL、data URI、base64 或字节) 转换为图片片段"""
    if isinstance(source, str):
        if source.startswith("data:"):
            source, mime = decode_data_uri(source)
        elif source.startswith(("http://", "https://")):
            return ImagePart(url=source)
        else:
            source = base64.b64decode(source)
    stored = save_image(source, mime.split("/")[-1])
    if isinstance(stored, Path):
        return ImagePart(path=stored, mime_type=mime)
    return ImagePart(raw=stored, mime_type=mime)


async def media_url(part: Any, default_mime: str) -> str:
    """获取媒体片段的 URL，本地数据转换为 data URI"""
    return part.url or await part.get_data_uri(default_mime)


def replay_data(part: Any, protocol: str) -> Any:
    """读取片段上保存的同协议回传数据"""
    replay = (part.metadata or {}).get("replay")
    if isinstance(replay, dict) and replay.get("protocol") == protocol:
        return replay.get("data")
    return None


def replay_metadata(protocol: str, data: Any) -> dict[str, Any]:
    """生成保存回传数据的片段元数据"""
    return {"replay": {"protocol": protocol, "data": data}}
