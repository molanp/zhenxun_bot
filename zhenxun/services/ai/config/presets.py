from typing import Any

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic", "gemini")
"""支持的接口协议"""

PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "key_url": "https://platform.openai.com/api-keys",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://api.openai.com/v1",
            "openai_responses": "https://api.openai.com/v1",
        },
        "reasoning_param": "openai",
        "web_search": ("openai_responses",),
    },
    "deepseek": {
        "label": "DeepSeek",
        "key_url": "https://platform.deepseek.com/api_keys",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://api.deepseek.com",
            "openai_responses": "https://api.deepseek.com",
            "anthropic": "https://api.deepseek.com/anthropic",
        },
        "reasoning_param": "deepseek",
        "web_search": ("openai_chat", "openai_responses", "anthropic"),
    },
    "ark": {
        "label": "火山方舟 (豆包)",
        "key_url": "https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://ark.cn-beijing.volces.com/api/v3",
            "openai_responses": "https://ark.cn-beijing.volces.com/api/v3",
        },
        "reasoning_param": "ark",
        "web_search": ("openai_responses",),
    },
    "glm": {
        "label": "智谱 GLM",
        "key_url": "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://open.bigmodel.cn/api/paas/v4",
            "anthropic": "https://open.bigmodel.cn/api/anthropic",
        },
        "reasoning_param": "thinking",
        "web_search": ("openai_chat",),
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "key_url": "https://platform.moonshot.cn/console/api-keys",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://api.moonshot.cn/v1",
            "openai_responses": "https://api.moonshot.cn/v1",
            "anthropic": "https://api.moonshot.cn/anthropic",
        },
        "reasoning_param": "thinking",
        "web_search": ("openai_responses",),
    },
    "mimo": {
        "label": "小米 MiMo",
        "key_url": "https://platform.xiaomimimo.com",
        "protocol": "openai_chat",
        "base_urls": {"openai_chat": "https://api.xiaomimimo.com/v1"},
        "reasoning_param": "thinking",
        "web_search": ("openai_chat",),
        "tts_api": "mimo_chat",
    },
    "dashscope": {
        "label": "阿里云百炼",
        "key_url": "https://bailian.console.aliyun.com/cn-beijing?tab=model#/api-key",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "openai_responses": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "reasoning_param": "enable_thinking",
        "web_search": ("openai_chat", "openai_responses"),
    },
    "siliconflow": {
        "label": "硅基流动",
        "key_url": "https://cloud.siliconflow.cn/account/ak",
        "protocol": "openai_chat",
        "base_urls": {"openai_chat": "https://api.siliconflow.cn/v1"},
        "reasoning_param": "enable_thinking",
    },
    "minimax": {
        "label": "MiniMax",
        "key_url": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
        "protocol": "openai_chat",
        "base_urls": {
            "openai_chat": "https://api.minimaxi.com/v1",
            "anthropic": "https://api.minimaxi.com/anthropic",
        },
        "reasoning_param": "minimax",
        "web_search": ("anthropic",),
        "tts_api": "minimax_t2a",
    },
    "openrouter": {
        "label": "OpenRouter",
        "key_url": "https://openrouter.ai/settings/keys",
        "protocol": "openai_chat",
        "base_urls": {"openai_chat": "https://openrouter.ai/api/v1"},
        "reasoning_param": "openrouter",
        "image_api": "chat",
    },
    "jina": {
        "label": "Jina AI",
        "key_url": "https://jina.ai/api-dashboard",
        "protocol": "openai_chat",
        "base_urls": {"openai_chat": "https://api.jina.ai/v1"},
        "embedding_api": "jina",
    },
    "gemini": {
        "label": "Google Gemini",
        "key_url": "https://aistudio.google.com/apikey",
        "protocol": "gemini",
        "base_urls": {
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "openai_chat": "https://generativelanguage.googleapis.com/v1beta/openai",
        },
        "reasoning_param": "openai",
        "web_search": ("gemini",),
    },
    "anthropic": {
        "label": "Anthropic",
        "key_url": "https://console.anthropic.com/settings/keys",
        "protocol": "anthropic",
        "base_urls": {"anthropic": "https://api.anthropic.com"},
        "web_search": ("anthropic",),
    },
}
"""厂商预设：默认协议、各协议的默认地址、API Key 申请地址、推理参数写法，
以及官方接口上提供原生网页搜索的协议 (web_search)"""

_PRESET_HOSTS = (
    ("api.openai.com", "openai"),
    ("deepseek.com", "deepseek"),
    ("volces.com", "ark"),
    ("bigmodel.cn", "glm"),
    ("moonshot.cn", "kimi"),
    ("moonshot.ai", "kimi"),
    ("xiaomimimo.com", "mimo"),
    ("aliyuncs.com", "dashscope"),
    ("siliconflow.cn", "siliconflow"),
    ("siliconflow.com", "siliconflow"),
    ("minimaxi.com", "minimax"),
    ("minimax.io", "minimax"),
    ("openrouter.ai", "openrouter"),
    ("jina.ai", "jina"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("anthropic.com", "anthropic"),
)
"""厂商官方接口的域名后缀"""

GEMINI_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)
"""Gemini 可配置的安全过滤类别"""


def detect_preset(base_url: str | None) -> str | None:
    """根据接口地址的域名识别厂商官方接口

    参数:
        base_url: 接口地址

    返回:
        str | None: 预设名称，不是已知厂商的官方域名 (如中转站) 时为 None
    """
    host = (base_url or "").lower().split("://", 1)[-1].split("/", 1)[0]
    host = host.rsplit("@", 1)[-1].split(":", 1)[0]
    return next(
        (
            preset
            for suffix, preset in _PRESET_HOSTS
            if host == suffix or host.endswith(f".{suffix}")
        ),
        None,
    )


def preset_base_url(preset: str | None, protocol: str) -> str | None:
    """获取预设在指定协议下的默认地址

    参数:
        preset: 预设名称
        protocol: 接口协议

    返回:
        str | None: 默认地址，预设不支持该协议时为 None
    """
    return PRESETS.get(preset or "", {}).get("base_urls", {}).get(protocol)
