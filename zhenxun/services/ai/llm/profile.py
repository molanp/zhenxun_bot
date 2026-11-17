"""
模型能力档案解析

优先级：用户配置 > 实测记录 > 发现快照 > 能力库 > 名称提示表 > 协议默认值。
能力库来自 models.dev 公开数据，按模型名称补全模态、推理等级、上下文等能力；
结构化输出方式与内置工具只采信接口自报和实测。
"""

import json
import os
from pathlib import Path
import re
import time
from typing import Any

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.config.manager import get_llm_config
from zhenxun.services.ai.config.models import ModelConfig, ProviderConfig
from zhenxun.services.ai.config.presets import PRESETS
from zhenxun.services.ai.core.exceptions import ResponseParseException
from zhenxun.services.ai.core.models import (
    REASONING_LEVELS,
    ModelModality,
    ModelProfile,
)
from zhenxun.services.ai.utils.logger import log_llm as logger
from zhenxun.utils.http_utils import AsyncHttpx

CATALOG_FILE = DATA_PATH / "ai" / "model_catalog.json"
CATALOG_MAX_AGE = 7 * 86400
"""能力库自动更新间隔 (秒)"""
_VENDORS = (
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "zhipuai",
    "moonshotai",
    "minimax",
    "alibaba",
    "xiaomi",
    "volcengine",
    "xai",
    "mistral",
)
"""能力库中同名模型优先采用的厂商"""
_MODALITIES = {m.value for m in ModelModality}
_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|latest)$")
_catalog: dict[str, Any] | None = None

_ALL_INPUT = ["text", "image", "audio", "video", "file"]
_NO_CHAT = {"tools": False, "structured_output": "none"}

_HINTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (r"tts", {"tasks": ["tts"], "output_modalities": ["audio"], **_NO_CHAT}),
    (
        r"whisper|transcri|sensevoice|paraformer|asr",
        {"tasks": ["stt"], "input_modalities": ["audio"], **_NO_CHAT},
    ),
    (r"rerank|colbert", {"tasks": ["rerank"], "context_window": 8192, **_NO_CHAT}),
    (
        r"gemini-embedding-2|embeddings-v\d+-omni",
        {
            "tasks": ["embedding"],
            "input_modalities": _ALL_INPUT,
            "context_window": 8192,
            **_NO_CHAT,
        },
    ),
    (
        r"embed|bge-(m3|large|base|small)",
        {"tasks": ["embedding"], "context_window": 8192, **_NO_CHAT},
    ),
    (
        r"gemini-.*image|nano-banana",
        {
            "tasks": ["chat", "image"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["text", "image"],
            "tools": False,
        },
    ),
    (
        r"gpt-image|dall-e|cogview|seedream|imagen|flux|kolors|qwen-image|wanx",
        {
            "tasks": ["image"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["image"],
            **_NO_CHAT,
        },
    ),
    (r"gemini-3|claude", {"mixed_tools": True}),
    (r"^deepseek-flash$", {"input_modalities": ["text", "image"]}),
    (r"vision|-vl\b|-vl-|qvq|omni|pixtral", {"input_modalities": ["text", "image"]}),
)
"""能力库未收录时按名称推断任务类型与输入模态，同一字段以先匹配的条目为准"""

_FAMILY_REASONING = (
    (r"deepseek", "deepseek"),
    (r"glm-|kimi|mimo", "thinking"),
    (r"qwen", "enable_thinking"),
    (r"doubao|seed", "ark"),
    (r"minimax", "minimax"),
)
"""OpenAI Chat 协议下未使用厂商预设时，按模型系列推断推理参数写法"""

_DEFAULT_VOICES = {
    "gemini": "Aoede",
    "minimax": "female-shaonv",
    "mimo": "mimo_default",
}

_PROFILE_FIELDS = set(ModelProfile.model_fields) - {"sources"}


def normalize_model_name(name: str) -> str:
    """去掉厂商前缀与 ":" 后的标签并转为小写，用于匹配提示表与模型目录

    参数:
        name: 模型名称

    返回:
        str: 规范化后的名称
    """
    return name.rsplit("/", 1)[-1].split(":", 1)[0].strip().lower()


def hint_profile(name: str) -> dict[str, Any]:
    """按名称提示表推断能力字段

    参数:
        name: 模型名称

    返回:
        dict[str, Any]: 推断出的字段
    """
    key = normalize_model_name(name)
    result: dict[str, Any] = {}
    for pattern, fields in _HINTS:
        if re.search(pattern, key):
            for field, value in fields.items():
                result.setdefault(field, value)
    return result


def _protocol_defaults(
    provider: ProviderConfig, name: str, protocol: str
) -> dict[str, Any]:
    key = normalize_model_name(name)
    preset = PRESETS.get(provider.preset or "", {})
    defaults: dict[str, Any] = {
        "structured_output": {
            "openai_chat": "json_object",
            "anthropic": "none",
        }.get(protocol, "json_schema")
    }
    if protocol == "gemini":
        defaults["reasoning_param"] = "level" if "gemini-3" in key else "budget"
    elif protocol == "anthropic":
        defaults["reasoning_param"] = "budget"
    elif protocol == "openai_responses":
        defaults["reasoning_param"] = "openai"
    else:
        family = next(
            (item[1] for item in _FAMILY_REASONING if re.search(item[0], key)), None
        )
        defaults["reasoning_param"] = (
            preset.get("reasoning_param") or family or "openai"
        )
    voice_key = protocol if protocol == "gemini" else provider.preset or ""
    defaults["default_voice"] = _DEFAULT_VOICES.get(voice_key, "alloy")
    return defaults


def load_json_file(path: Path) -> dict[str, Any]:
    """读取 JSON 对象文件，不存在或无法解析时返回空字典

    参数:
        path: 文件路径

    返回:
        dict[str, Any]: 文件内容
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"{path.name} 无法解析，已忽略: {e}")
        return {}


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    """原子写入 JSON 对象文件

    参数:
        path: 文件路径
        data: 文件内容
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temp_path, path)


def parse_modalities(values: Any) -> list[str] | None:
    """把接口给出的模态列表转换为档案模态 (pdf 视为 file)

    参数:
        values: 模态列表

    返回:
        list[str] | None: 可识别的模态，无法识别时为 None
    """
    if not isinstance(values, list):
        return None
    mapped = ["file" if v == "pdf" else v for v in values if isinstance(v, str)]
    return [v for v in dict.fromkeys(mapped) if v in _MODALITIES] or None


def _reasoning_fields(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("reasoning") is False:
        return {
            "reasoning_levels": [],
            "reasoning_off": False,
            "reasoning_param": "none",
        }
    options = {
        option.get("type"): option
        for option in item.get("reasoning_options") or []
        if isinstance(option, dict)
    }
    if not options:
        return {}
    values = (options.get("effort") or {}).get("values") or []
    budget = options.get("budget_tokens")
    fields: dict[str, Any] = {
        "reasoning_off": "toggle" in options
        or "none" in values
        or (budget or {}).get("min") == 0
    }
    if values:
        fields["reasoning_levels"] = [v for v in values if v in REASONING_LEVELS]
    elif budget is not None:
        fields["reasoning_levels"] = ["low", "medium", "high"]
    if values and budget is None:
        fields["reasoning_kind"] = "effort"
    elif budget is not None and not values:
        fields["reasoning_kind"] = "budget"
    return fields


def _catalog_fields(item: dict[str, Any]) -> dict[str, Any]:
    """把 models.dev 的模型条目转换为能力档案字段"""
    modalities = item.get("modalities") or {}
    inputs = parse_modalities(modalities.get("input"))
    outputs = parse_modalities(modalities.get("output"))
    fields: dict[str, Any] = {}
    if outputs and "image" in outputs:
        fields["tasks"] = ["chat", "image"] if "text" in outputs else ["image"]
    elif "embed" in str(item.get("family") or ""):
        fields["tasks"] = ["embedding"]
    if inputs:
        fields["input_modalities"] = inputs
    if outputs:
        fields["output_modalities"] = outputs
    if isinstance(item.get("tool_call"), bool):
        fields["tools"] = item["tool_call"]
    if isinstance(item.get("temperature"), bool):
        fields["sampling"] = item["temperature"]
    limit = item.get("limit") or {}
    if isinstance(limit.get("context"), int) and limit["context"] > 0:
        fields["context_window"] = limit["context"]
    chat = "tasks" not in fields or "chat" in fields["tasks"]
    if chat and isinstance(limit.get("output"), int) and limit["output"] > 0:
        fields["max_output_tokens"] = limit["output"]
    if chat:
        fields.update(_reasoning_fields(item))
    return fields


def _load_catalog() -> dict[str, Any]:
    global _catalog
    if _catalog is None:
        data = load_json_file(CATALOG_FILE)
        _catalog = data if isinstance(data.get("models"), dict) else {"models": {}}
    return _catalog


def catalog_status() -> dict[str, Any]:
    """能力库状态

    返回:
        dict[str, Any]: 数据来源、更新时间 (时间戳，未下载为 None) 与模型数量
    """
    catalog = _load_catalog()
    return {
        "source": catalog.get("source"),
        "updated_at": catalog.get("updated_at"),
        "count": len(catalog["models"]),
    }


def catalog_fields(name: str, protocol: str) -> dict[str, Any]:
    """按模型名称查询能力库

    参数:
        name: 模型名称，可带厂商前缀和日期后缀
        protocol: 调用协议，用于确定推理参数写法

    返回:
        dict[str, Any]: 能力档案字段，未收录时为空字典
    """
    models = _load_catalog()["models"]
    key = normalize_model_name(name)
    fields = models.get(key) or models.get(_DATE_SUFFIX.sub("", key))
    if not fields:
        return {}
    fields = dict(fields)
    fields.pop("structured_output", None)
    kind = fields.pop("reasoning_kind", None)
    if kind and fields.get("reasoning_param") != "none":
        if protocol == "gemini":
            fields["reasoning_param"] = "level" if kind == "effort" else "budget"
        elif protocol == "anthropic":
            fields["reasoning_param"] = "adaptive" if kind == "effort" else "budget"
    return fields


def build_catalog(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """把 models.dev 数据转换为按模型名称索引的能力字段

    参数:
        data: models.dev 的 api.json 内容 (厂商 -> models)

    返回:
        dict[str, dict[str, Any]]: 规范化模型名称 -> 能力字段，同名时优先主要厂商
    """
    order = {vendor: index for index, vendor in enumerate(_VENDORS)}
    models: dict[str, dict[str, Any]] = {}
    for vendor in sorted(data, key=lambda v: (order.get(v, len(order)), v)):
        entry = data[vendor]
        items = entry.get("models") if isinstance(entry, dict) else None
        for model_id, item in (items or {}).items():
            key = normalize_model_name(model_id)
            if key not in models and isinstance(item, dict):
                models[key] = _catalog_fields(item)
    return models


async def refresh_catalog() -> dict[str, Any]:
    """下载能力库并保存到本地

    返回:
        dict[str, Any]: 更新后的能力库状态
    """
    global _catalog
    url = get_llm_config().client_settings.catalog_url
    response = await AsyncHttpx.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ResponseParseException("能力库数据格式无效")
    models = build_catalog(data)
    if not models:
        raise ResponseParseException("能力库中没有模型数据")
    _catalog = {"source": url, "updated_at": time.time(), "models": models}
    save_json_file(CATALOG_FILE, _catalog)
    logger.info(f"模型能力库已更新，共 {len(models)} 个模型")
    return catalog_status()


def resolve_profile(
    provider: ProviderConfig,
    model: ModelConfig,
    snapshot: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
) -> ModelProfile:
    """解析模型的能力档案

    参数:
        provider: 提供商配置
        model: 模型配置
        snapshot: 模型发现时保存的能力快照
        observation: 核验与运行时记录的实测结果

    返回:
        ModelProfile: 合并后的能力档案，sources 记录各字段来源
    """
    protocol = model.protocol or provider.protocol
    observation = observation or {}
    layers = (
        ("default", _protocol_defaults(provider, model.name, protocol)),
        ("hint", hint_profile(model.name)),
        ("catalog", catalog_fields(model.reference or model.name, protocol)),
        ("discovery", snapshot or {}),
        ("observed", observation.get("fields") or {}),
        ("user", model.profile),
    )
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    native: dict[str, bool] = {}
    for source, fields in layers:
        for field, value in fields.items():
            if field in ("native_tools", "native_tools_off"):
                for type_id in value or ():
                    native[type_id] = field == "native_tools"
                    sources[f"native_tools.{type_id}"] = source
            elif field in _PROFILE_FIELDS and value is not None:
                merged[field] = value
                sources[field] = source
        if source == "observed":
            for type_id, supported in (observation.get("native_tools") or {}).items():
                native[type_id] = bool(supported)
                sources[f"native_tools.{type_id}"] = source
            inputs = set(merged.get("input_modalities") or ["text"])
            for modality, supported in (observation.get("inputs") or {}).items():
                if supported != (modality in inputs):
                    inputs ^= {modality}
                    merged["input_modalities"] = sorted(inputs)
                    sources["input_modalities"] = source
    tasks = set(merged.get("tasks") or ["chat"])
    inputs = set(merged.get("input_modalities") or ["text"])
    if protocol == "gemini" and "chat" in tasks and "audio" in inputs:
        merged["tasks"] = sorted(tasks | {"stt"})
    merged["native_tools"] = {t for t, supported in native.items() if supported}
    merged["native_tools_off"] = {t for t, supported in native.items() if not supported}
    levels = merged.get("reasoning_levels") or []
    merged["reasoning_levels"] = [
        level for level in REASONING_LEVELS if level in levels and level != "none"
    ]
    return ModelProfile(**merged, sources=sources)


def map_reasoning_effort(effort: str | None, profile: ModelProfile) -> str | None:
    """把统一推理强度映射为模型支持的等级

    参数:
        effort: 请求的推理强度
        profile: 模型能力档案

    返回:
        str | None: None 表示不发送推理参数，"none" 表示关闭推理，其余为具体等级
    """
    if not effort or profile.reasoning_param == "none":
        return None
    levels = profile.reasoning_levels
    if effort == "none":
        if profile.reasoning_off:
            return "none"
        return levels[0] if levels else None
    if not levels:
        return effort
    target = REASONING_LEVELS.index(effort)
    return min(
        levels,
        key=lambda level: (
            abs(REASONING_LEVELS.index(level) - target),
            -REASONING_LEVELS.index(level),
        ),
    )
