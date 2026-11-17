import contextvars
from functools import lru_cache
from typing import Any

from zhenxun.configs.config import Config
from zhenxun.configs.utils import parse_as
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump

from .models import DebugLogOptions, LLMConfig, ProviderConfig
from .presets import GEMINI_SAFETY_CATEGORIES, detect_preset

LOG_COMMAND = "AIConfig"


def get_ai_config():
    """获取 AI 配置组"""
    return Config.get("AI")


_LEGACY_TYPES = {
    "openai": ("openai", "openai_chat", "v1"),
    "deepseek": ("deepseek", "openai_chat", "v1"),
    "doubao": ("ark", "openai_chat", "v3"),
    "glm": ("glm", "openai_chat", "api/paas/v4"),
    "zhipu": ("glm", "openai_chat", "api/paas/v4"),
    "gemini": ("gemini", "gemini", "v1beta"),
    "openrouter": ("openrouter", "openai_chat", "v1"),
    "minimax": ("minimax", "openai_chat", "v1"),
    "mimo": ("mimo", "openai_chat", "v1"),
    "jina": ("jina", "openai_chat", "v1"),
    "openai_responses": (None, "openai_responses", "v1"),
    "smart": (None, "openai_chat", "v1"),
}

_LEGACY_TASKS = (
    ("embed", "embedding"),
    ("rerank", "rerank"),
    ("tts", "tts"),
    ("speech", "tts"),
    ("image", "image"),
)


def _legacy_url(api_base: str, prefix: str | None, version: str) -> str:
    base = api_base.rstrip("/")
    for segment in (prefix, version):
        segment = (segment or "").strip("/")
        if segment and not base.lower().endswith("/" + segment.lower()):
            first, _, rest = segment.partition("/")
            if base.rsplit("/", 1)[-1].lower() == first.lower():
                segment = rest
            base = f"{base}/{segment}" if segment else base
    return base


def _legacy_type(api_type: Any) -> tuple[str | None, str, str]:
    return _LEGACY_TYPES.get(str(api_type or "openai").lower(), _LEGACY_TYPES["openai"])


def _legacy_params(raw: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """读取旧版配置中的生成参数；max_output_tokens 不大于 1 是未填写的占位值，
    此时与它同时出现的 temperature 0 也视为未填写

    参数:
        raw: 旧版服务商或模型配置
        fields: 要读取的旧版字段

    返回:
        dict[str, Any]: 当前结构的生成参数
    """
    names = {
        "temperature": "temperature",
        "max_output_tokens": "max_tokens",
        "reasoning_effort": "reasoning_effort",
    }
    params = {names[key]: raw[key] for key in fields if raw.get(key) is not None}
    tokens = params.get("max_tokens")
    if isinstance(tokens, int | float) and tokens <= 1:
        del params["max_tokens"]
        if params.get("temperature") == 0:
            del params["temperature"]
    return params


def _migrate_legacy_model(
    raw: dict[str, Any], provider_type: str, api_base: str | None
) -> dict[str, Any]:
    model: dict[str, Any] = {"name": str(raw.get("model_name") or raw.get("name"))}
    if raw.get("is_available") is False:
        model["enabled"] = False
    params = _legacy_params(
        raw, ("temperature", "max_output_tokens", "reasoning_effort")
    )
    if params:
        model["params"] = params
    profile: dict[str, Any] = {}
    if raw.get("max_input_tokens"):
        profile["context_window"] = raw["max_input_tokens"]
    task_type = str(raw.get("task_type") or "").lower()
    if task := next((t for key, t in _LEGACY_TASKS if key in task_type), None):
        profile["tasks"] = [task]
    if profile:
        model["profile"] = profile
    model_type = str(raw.get("api_type") or provider_type).lower()
    if model_type == "smart" and "gemini" in model["name"].lower():
        model_type = "gemini"
    _, protocol, version = _legacy_type(model_type)
    _, provider_protocol, provider_version = _legacy_type(provider_type)
    if protocol != provider_protocol:
        model["protocol"] = protocol
    if api_base and (raw.get("path_prefix") or version != provider_version):
        model["base_url"] = _legacy_url(api_base, raw.get("path_prefix"), version)
    if raw.get("endpoint"):
        logger.warning(
            f"模型 {model['name']} 的 endpoint 配置已不再支持，已忽略: "
            f"{raw['endpoint']}",
            LOG_COMMAND,
        )
    return model


def _migrate_legacy_provider(
    raw: dict[str, Any], gemini_threshold: str | None
) -> dict[str, Any]:
    api_type = str(raw.get("api_type") or "openai").lower()
    preset, protocol, version = _legacy_type(api_type)
    api_base = raw.get("api_base") or None
    detected = detect_preset(api_base)
    if detected or (api_base and api_type == "openai"):
        preset = detected
    if api_type == "openai_responses" and preset != "openai":
        preset = None
    keys = raw.get("api_key") or raw.get("api_keys") or []
    provider: dict[str, Any] = {
        "name": raw.get("name"),
        "preset": preset,
        "protocol": protocol,
        "base_url": _legacy_url(api_base, None, version) if api_base else "",
        "api_keys": [keys] if isinstance(keys, str) else list(keys),
        "timeout": raw.get("timeout") or 120,
        "models": [
            _migrate_legacy_model(m, api_type, api_base)
            for m in raw.get("models") or []
            if isinstance(m, dict)
        ],
    }
    params = _legacy_params(raw, ("temperature", "max_output_tokens"))
    if protocol == "gemini" and gemini_threshold and gemini_threshold != "BLOCK_NONE":
        params["safetySettings"] = [
            {"category": category, "threshold": gemini_threshold}
            for category in GEMINI_SAFETY_CATEGORIES
        ]
    if params:
        provider["params"] = params
    return provider


def _is_legacy_provider(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    if {"api_key", "api_base", "api_type"} & raw.keys():
        return True
    return any(
        isinstance(m, dict) and "model_name" in m for m in raw.get("models") or []
    )


def migrate_legacy_ai_config() -> bool:
    """把旧版 AI 配置结构转换为当前结构并写回配置文件

    返回:
        bool: 是否发生了迁移
    """
    changed = False
    providers = Config.get_config("AI", "PROVIDERS", build_model=False)
    if isinstance(providers, list) and any(map(_is_legacy_provider, providers)):
        provider_settings = Config.get_config(
            "AI", "PROVIDER_SETTINGS", build_model=False
        )
        threshold = None
        if isinstance(provider_settings, dict):
            threshold = (provider_settings.get("gemini") or {}).get("safety_threshold")
        migrated = [
            _migrate_legacy_provider(p, threshold) if _is_legacy_provider(p) else p
            for p in providers
        ]
        Config.set_config("AI", "PROVIDERS", migrated)
        changed = True
    client_settings = Config.get_config("AI", "CLIENT_SETTINGS", build_model=False)
    if isinstance(client_settings, dict) and "timeout" in client_settings:
        client_settings = {k: v for k, v in client_settings.items() if k != "timeout"}
        Config.set_config("AI", "CLIENT_SETTINGS", client_settings)
        changed = True
    web_search = Config.get_config("AI", "WEB_SEARCH", build_model=False)
    if isinstance(web_search, dict) and "api_key" in web_search:
        web_search = dict(web_search)
        if key := web_search.pop("api_key"):
            web_search["api_keys"] = [*(web_search.get("api_keys") or []), key]
        Config.set_config("AI", "WEB_SEARCH", web_search)
        changed = True
    if changed:
        Config.save(save_simple_data=True)
        logger.info("已将旧版 AI 配置转换为新结构并写回配置文件", LOG_COMMAND)
    return changed


def register_llm_configs():
    """注册 LLM 服务的配置项"""

    llm_config = LLMConfig()
    Config.add_plugin_config(
        "AI",
        "default_models",
        model_dump(llm_config.default_models),
        help=(
            "不同任务类型的全局默认模型，值为 服务商/模型 或路由组名。\n"
            "键: chat(对话), embedding(向量嵌入), tts(语音合成), "
            "image(图像生成), rerank(文本重排), "
            "vision(视觉模型，对话模型不支持图片时先用它把图片转为文字描述)"
        ),
        type=dict,
        ui={
            "label": "默认模型",
            "section": "default_models",
            "component": "object",
            "order": 10,
        },
    )
    Config.add_plugin_config(
        "AI",
        "client_settings",
        model_dump(llm_config.client_settings),
        help=(
            "LLM客户端高级设置。\n"
            "包含: max_retries(同线路重试次数), retry_delay(重试延迟), "
            "structured_retries(结构化生成重试), rate_limit_cooldown(限流冷却秒数), "
            "quota_cooldown(额度耗尽冷却秒数), route_cooldown(线路熔断冷却秒数), "
            "catalog_url(模型能力库下载地址，models.dev 格式)"
        ),
        type=dict,
        ui={
            "label": "客户端设置",
            "section": "advanced",
            "component": "object",
            "order": 70,
            "advanced": True,
        },
    )
    Config.add_plugin_config(
        "AI",
        "debug_log",
        model_dump(llm_config.debug_log),
        help=(
            "LLM日志详情开关。示例: {'show_tools': True, 'show_schema': False, "
            "'show_safety': False}"
        ),
        type=dict,
        ui={
            "label": "调试日志",
            "section": "advanced",
            "component": "object",
            "order": 80,
            "advanced": True,
        },
    )

    Config.add_plugin_config(
        "AI",
        "context_settings",
        model_dump(llm_config.context_settings),
        help=(
            "智能上下文管理与压缩配置。\n"
            "包含:\n"
            "  - llm_summary: 大模型总结策略配置\n"
            "    - enable: 是否开启大模型对话总结以压缩上下文\n"
            "    - trigger_threshold: 触发压缩的 Token 阈值。<=1.0为比例，>1.0为绝对 Token 数\n"  # noqa: E501
            "    - max_history_turns: 触发压缩的最大历史对话轮数\n"
            "    - summarization_model: 指定用于总结的大模型名称\n"
            "    - summarization_prompt: 指导大模型总结的系统提示词\n"
            "    - keep_recent_turns: 总结外强制原样保留的最近对话轮数\n"
            "  - vision_window_size: 多模态滑动窗口大小。0表示无限制，>0表示仅保留最近N轮包含多模态真实数据的消息，超过则自动降级为占位符\n"  # noqa: E501
            "  - image_max_side: 发给模型的图片长边上限（像素），超过 512KB 时继续缩小，动图拼成 3×3，0 表示不压缩\n"  # noqa: E501
            "  - file_max_chars: 收到文件时读取内容交给模型的最大字数（默认 20000），0 表示不读取\n"  # noqa: E501
            "  - tool_pruning: 工具结果过载修剪策略配置\n"
            "    - enable: 是否开启长工具输出结果的自动修剪\n"
            "    - trigger_threshold: 触发修剪的工具纯 Token 阈值。<=1.0为比例，>1.0为绝对 Token 数\n"  # noqa: E501
            "    - max_history_turns: 触发修剪的最大工具消息轮数。设为 0 表示不限制轮数\n"  # noqa: E501
            "    - keep_recent_turns: 修剪时强制原样保留的最新的工具消息轮数\n"
            "  - history_retention_days: 短期对话记忆保留天数，0 表示不限制\n"
            "  - history_max_messages: 每个会话最多保留的短期对话记忆条数，0 表示不限制"
        ),
        type=dict,
        ui={
            "label": "上下文管理",
            "section": "context",
            "component": "object",
            "order": 30,
        },
    )

    Config.add_plugin_config(
        "AI",
        "MODEL_GROUPS",
        llm_config.model_groups,
        help=(
            "虚拟模型路由组配置 (Virtual Router Groups)。\n"
            "键为组名，值为模型名称或其它组名的列表。\n"
            "使用 chat(model='cheap_models') 时系统将自动按列表顺序轮询和故障转移。"
        ),
        type=dict,
        ui={
            "label": "模型路由组",
            "section": "model_groups",
            "component": "key_value",
            "order": 20,
        },
    )

    Config.add_plugin_config(
        "AI",
        "agent_settings",
        model_dump(llm_config.agent_settings),
        help=(
            "Agent 执行引擎默认设置。\n"
            "包含: max_cycles(最大工具循环数), "
            "global_max_cycles(跨嵌套智能体的总循环上限), "
            "enable_parallel_calls(允许并行), "
            "reflexion_retries(反思重试次数), "
            "enable_fallback_summary(达到最大循环时兜底总结), "
            "tool_output_max_chars(单个工具结果的最大字符数), "
            "tool_timeout(单次工具调用超时秒数), "
            "mcp_cleanup_timeout(MCP 闲置回收时间)"
        ),
        type=dict,
        ui={
            "label": "Agent 设置",
            "section": "agent",
            "component": "object",
            "order": 40,
        },
    )

    Config.add_plugin_config(
        "AI",
        "sandbox",
        model_dump(llm_config.sandbox),
        help=(
            "Agent 代码沙箱（Docker）配置，建议在 WebUI“AI 聊天 → Agent”中设置。\n"
            "包含: enable_sandbox(启用), docker_image(镜像), cpus(CPU 核数), "
            "memory_mb(内存 MB), exec_timeout(执行超时秒数), network(联网), "
            "cleanup_timeout(闲置清理秒数，0 不清理)。"
        ),
        type=dict,
        ui={
            "label": "沙箱",
            "section": "sandbox",
            "component": "object",
            "order": 50,
            "effect": "new_session",
        },
    )

    Config.add_plugin_config(
        "AI",
        "web_search",
        model_dump(llm_config.web_search),
        help=(
            "联网搜索设置。\n"
            "mode: auto(模型支持原生联网时用原生，否则用搜索接口) / native(只用原生) / "
            "api(只用搜索接口) / off(关闭)。\n"
            "原生联网只在厂商官方接口上默认启用，中转站需核验通过或在模型能力中强制开启。\n"
            "provider: 搜索接口 deepseek / kimi / glm / jina / tavily / bocha；\n"
            "api_keys: 搜索接口密钥，可填写多个，失败时换下一个；"
            "留空时复用同厂商服务商的密钥；"
            "max_results: 每次搜索的结果数。"
        ),
        type=dict,
        ui={
            "label": "联网搜索",
            "section": "web_search",
            "component": "object",
            "order": 25,
            "secret": True,
        },
    )

    Config.add_plugin_config(
        "AI",
        "deepseek_tools",
        model_dump(llm_config.deepseek_tools),
        help=(
            "DeepSeek 独立工具 API，默认关闭收费媒体转换。provider 为已保存的官方"
            "服务商名称，留空自动选择；ocr_enabled 仅在无视觉模型时识别图片文字，"
            "asr_enabled/tts_enabled 仅在未指定对应模型时生效；"
            "voice 为 stella 或 tide；timeout 为请求总超时秒数。"
            "不使用 DSH 登录凭据，不修改 DSH 原生工具。"
        ),
        type=dict,
        ui={
            "label": "DeepSeek 工具",
            "section": "deepseek_tools",
            "component": "object",
            "order": 26,
        },
    )

    Config.add_plugin_config(
        "AI",
        "PROVIDERS",
        [],
        help=(
            "配置多个 AI 服务提供商及其模型信息。\n"
            "preset: 厂商预设 (openai/deepseek/ark/glm/kimi/mimo/dashscope/"
            "siliconflow/minimax/openrouter/jina/gemini/anthropic)，"
            "用于补全默认协议与地址。\n"
            "protocol: 接口协议 (openai_chat/openai_responses/anthropic/gemini)。\n"
            "base_url: 与对应协议官方 SDK 一致的地址，如 https://api.openai.com/v1。\n"
            "api_keys: 密钥列表，多个轮询；headers: 附加请求头；timeout: 超时秒数；\n"
            "params: 默认生成参数，非通用参数原样并入请求体。\n"
            "models: 模型列表，可直接写模型名，或写 name / enabled / protocol / "
            "base_url / reference(能力库参照名) / profile / params。\n"
            "profile: 手动指定的能力，留空的字段依次取实测核验、模型发现、"
            "能力库、名称推断，常用字段:\n"
            "  - tasks: chat / embedding / rerank / tts / image\n"
            "  - input_modalities: text / image / audio / video / file\n"
            "  - tools: true / false 强制开关函数调用\n"
            "  - structured_output: json_schema / json_object / none\n"
            "  - native_tools / native_tools_off: 强制开启 / 关闭的内置工具 "
            "(web_search / code_execution / url_context)\n"
            "  - context_window / max_output_tokens / reasoning_levels\n"
            "未经核验的已启用对话模型在启动后自动核验一次，"
            "结果保存在 data/ai/model_observations.json；\n"
            "强制开启的能力被接口拒绝时只对当次请求降级重试，并在控制台警告。"
        ),
        default_value=[],
        type=list[ProviderConfig],
        ui={
            "label": "服务商与模型",
            "section": "providers",
            "component": "object_list",
            "order": 0,
            "secret": True,
        },
    )
    # 在空上下文中读取，避免把正在导入的插件登记为 AI 配置项的导入期读取方
    contextvars.Context().run(migrate_legacy_ai_config)


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """获取 LLM 配置实例"""
    ai_config = get_ai_config()

    raw_debug = ai_config.get("debug_log", False)
    if isinstance(raw_debug, bool):
        debug_log_val = DebugLogOptions(
            show_tools=raw_debug, show_schema=raw_debug, show_safety=raw_debug
        )
    else:
        debug_log_val = raw_debug

    config_data = {
        "default_models": ai_config.get("default_models", {}),
        "client_settings": ai_config.get("client_settings", {}),
        "debug_log": debug_log_val,
        "providers": ai_config.get("PROVIDERS", []),
        "context_settings": ai_config.get("context_settings", {}),
        "model_groups": ai_config.get("MODEL_GROUPS", {}),
        "agent_settings": ai_config.get("agent_settings", {}),
        "sandbox": ai_config.get("sandbox", {}),
        "web_search": ai_config.get("web_search", {}),
        "deepseek_tools": ai_config.get("deepseek_tools", {}),
    }

    return parse_as(LLMConfig, config_data)
