import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .presets import PRESETS, PROTOCOLS, preset_base_url

_PLACEHOLDER_KEYS = {"change_me", "changeme", "replace_me"}


def is_placeholder_key(value: Any) -> bool:
    """判断 API Key 是否为空值或示例占位符

    参数:
        value: 密钥

    返回:
        bool: 是否为空值或 YOUR_API_KEY、AIzaSy****、${KEY}、<KEY> 一类占位符
    """
    text = str(value or "").strip()
    folded = text.casefold()
    return bool(
        not text
        or folded in _PLACEHOLDER_KEYS
        or folded.startswith("your_")
        or "****" in text
        or re.fullmatch(r"\$\{[^{}]+\}", text)
        or re.fullmatch(r"<[^<>]+>", text)
    )


class DebugLogOptions(BaseModel):
    """调试日志细粒度控制选项"""

    show_tools: bool = True
    """是否在日志中显示工具定义 (JSON Schema)"""
    show_schema: bool = True
    """是否在日志中显示结构化输出 Schema (response_format)"""
    show_safety: bool = True
    """是否在日志中显示安全设置 (safetySettings)"""

    def __bool__(self) -> bool:
        return self.show_tools or self.show_schema or self.show_safety


class ClientSettings(BaseModel):
    """LLM 客户端重试、冷却与能力库设置"""

    max_retries: int = Field(default=3, ge=0)
    """同一线路上可重试错误的最大重试次数"""
    retry_delay: int = Field(default=2, ge=0)
    """请求重试的基础延迟时间 (秒)"""
    structured_retries: int = Field(default=2, ge=0)
    """结构化生成校验失败时的最大重试次数"""
    rate_limit_cooldown: int = Field(default=60, ge=0)
    """触发限流且服务端未给出 Retry-After 时，密钥的冷却时间 (秒)"""
    quota_cooldown: int = Field(default=3600, ge=0)
    """额度耗尽时密钥的冷却时间 (秒)"""
    route_cooldown: int = Field(default=60, ge=0)
    """线路连续失败后熔断的冷却时间 (秒)"""
    catalog_url: str = "https://models.dev/api.json"
    """模型能力库 (models.dev 格式) 的下载地址"""


class LLMSummaryConfig(BaseModel):
    """LLM 自然语言总结压缩策略配置"""

    enable: bool = True
    """是否开启大模型对话总结以压缩上下文"""
    trigger_threshold: float = Field(default=0.8, ge=0)
    """触发压缩的 Token 阈值。<=1.0 为比例，>1.0 为绝对 Token 数"""
    max_history_turns: int = Field(default=0, ge=0)
    """触发压缩的最大历史对话轮数。设为 0 表示不限制轮数（仅受 Token 阈值控制）。"""
    summarization_model: str | None = None
    """指定用于执行总结任务的大模型名称，为空则使用全局默认"""
    summarization_prompt: str = (
        "概括以下对话，供之后继续对话时参考。保留：重要发言来自谁（写名字加用户ID）、"
        "各人的偏好与个人情况、做出的决定和约定、进行中的任务与未解决的问题。"
        "寒暄、刷屏和重复内容不用保留。"
    )
    """指导大模型进行总结的系统提示词"""
    keep_recent_turns: int = Field(default=3, ge=0)
    """在总结之外，强制原样保留的最近对话轮数"""


class ToolPruningConfig(BaseModel):
    """工具结果修剪策略配置"""

    enable: bool = False
    """是否开启长工具输出结果的自动修剪"""
    trigger_threshold: float = Field(default=0.6, ge=0)
    """触发修剪的工具纯 Token 阈值。<=1.0 为比例，>1.0 为绝对 Token 数"""
    max_history_turns: int = Field(default=15, ge=0)
    """触发修剪的最大工具消息轮数。设为 0 表示不限制轮数。"""
    keep_recent_turns: int = Field(default=3, ge=0)
    """修剪时强制原样保留的最新的工具消息轮数，确保当下反思不受影响"""


class ContextManagementSettings(BaseModel):
    """智能上下文管理与压缩算法设置"""

    llm_summary: LLMSummaryConfig = Field(default_factory=LLMSummaryConfig)
    """大模型自然语言总结策略"""

    vision_window_size: int = Field(default=3, ge=0)
    """多模态滑动窗口大小。0表示无限制，>0表示仅保留最近N轮包含多模态真实数据的消息，超龄则自动降级为占位符"""

    image_max_side: int = Field(default=1280, ge=0, le=8192)
    """发给模型的图片长边上限（像素），超过 512KB 时继续缩小，动图拼成 3×3。
    0 表示不压缩"""

    file_max_chars: int = Field(default=20000, ge=0, le=200000)
    """聊天中收到文件时读取内容交给模型的最大字数（文本、代码与配置文件、
    html/csv/json/epub，安装文档解析组件后支持 docx/xlsx/pptx/pdf）。0 表示不读取"""

    tool_pruning: ToolPruningConfig = Field(default_factory=ToolPruningConfig)
    """工具结果过载修剪策略"""

    history_retention_days: int = Field(default=30, ge=0)
    """短期对话记忆的保留天数，每天清理一次超期消息。0 表示不限制"""

    history_max_messages: int = Field(default=500, ge=0)
    """每个会话最多保留的短期对话记忆条数，超出的旧消息每天清理一次。0 表示不限制"""


class ModelConfig(BaseModel):
    """提供商下的单个模型配置"""

    name: str
    """模型名称 (请求时发送给服务端的 model)"""
    enabled: bool = True
    """是否启用"""
    protocol: str | None = None
    """覆盖提供商的接口协议"""
    base_url: str | None = None
    """覆盖提供商的接口地址"""
    reference: str | None = None
    """能力参照：按此模型名称查询能力库，用于中转站自定义的模型别名"""
    profile: dict[str, Any] = Field(default_factory=dict)
    """手动指定的能力档案字段，优先级最高"""
    params: dict[str, Any] = Field(default_factory=dict)
    """默认生成参数，通用参数名之外的键原样合并进请求体"""


class ProviderConfig(BaseModel):
    """LLM 服务提供商配置"""

    name: str
    """提供商唯一标识名称"""
    preset: str | None = None
    """厂商预设，用于补全默认协议、地址和推理参数写法"""
    protocol: str = ""
    """接口协议：openai_chat / openai_responses / anthropic / gemini"""
    base_url: str = ""
    """接口地址，与对应协议官方 SDK 的 base_url 一致"""
    api_keys: list[str] = Field(default_factory=list)
    """API 密钥列表，多个密钥轮询使用"""
    headers: dict[str, str] = Field(default_factory=dict)
    """附加请求头"""
    params: dict[str, Any] = Field(default_factory=dict)
    """提供商级默认生成参数，优先级低于模型参数"""
    timeout: int = Field(default=120, ge=1)
    """请求超时时间 (秒)"""
    models: list[ModelConfig] = Field(default_factory=list)
    """提供的模型列表"""

    @field_validator("api_keys", mode="before")
    @classmethod
    def _keys_as_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [str(key).strip() for key in value if not is_placeholder_key(key)]
        return value

    @field_validator("models", mode="before")
    @classmethod
    def _models_from_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value

    @model_validator(mode="after")
    def _apply_preset(self) -> "ProviderConfig":
        if self.preset and self.preset not in PRESETS:
            raise ValueError(f"未知的厂商预设: {self.preset}")
        preset = PRESETS.get(self.preset or "")
        if not self.protocol:
            self.protocol = preset["protocol"] if preset else "openai_chat"
        if not self.base_url:
            self.base_url = preset_base_url(self.preset, self.protocol) or ""
        self.base_url = self.base_url.rstrip("/")
        for protocol in (self.protocol, *(m.protocol for m in self.models)):
            if protocol and protocol not in PROTOCOLS:
                raise ValueError(f"未知的接口协议: {protocol}")
        return self

    def get_model(self, name: str) -> ModelConfig | None:
        """按名称查找模型配置"""
        return next((m for m in self.models if m.name == name), None)


class DefaultModelsConfig(BaseModel):
    """按任务分类的默认模型配置"""

    chat: str | None = None
    embedding: str | None = None
    tts: str | None = None
    stt: str | None = None
    """语音转文字模型"""
    image: str | None = None
    rerank: str | None = None
    vision: str | None = None
    """视觉模型：对话模型不支持图片时，先用它把图片转为文字描述"""


WEB_SEARCH_PROVIDERS = ("deepseek", "kimi", "glm", "jina", "tavily", "bocha")
"""可用的搜索接口"""


class WebSearchSettings(BaseModel):
    """联网搜索设置"""

    mode: str = "off"
    """off: 关闭（默认）；auto: 模型支持原生联网时用原生，否则用搜索接口；
    native: 只用原生联网；api: 只用搜索接口"""
    provider: str | None = None
    """搜索接口：deepseek / kimi / glm / jina / tavily / bocha"""
    api_keys: list[str] = Field(default_factory=list)
    """搜索接口密钥，可填写多个轮换使用，为空时复用同厂商服务商的密钥"""
    max_results: int = Field(default=5, ge=1, le=20)
    """每次搜索返回的结果数"""

    @field_validator("api_keys")
    @classmethod
    def _clean_keys(cls, value: list[str]) -> list[str]:
        return [key.strip() for key in value if key and key.strip()]

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, value: str) -> str:
        if value not in ("auto", "native", "api", "off"):
            raise ValueError(f"未知的联网方式: {value}")
        return value

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str | None) -> str | None:
        value = (value or "").strip() or None
        if value and value not in WEB_SEARCH_PROVIDERS:
            raise ValueError(f"未知的搜索接口: {value}")
        return value


class DeepSeekToolsSettings(BaseModel):
    """DeepSeek 独立工具接口，复用已保存的官方服务商密钥。"""

    provider: str = ""
    """官方 DeepSeek 服务商名称；为空时取第一个官方服务商，不使用 DSH 登录凭据。"""
    ocr_enabled: bool = False
    """对话线路不支持图片且未指定视觉模型时，使用 OCR 提取图片文字。"""
    asr_enabled: bool = False
    """未指定语音识别模型时，使用 DeepSeek ASR 转写音频。"""
    tts_enabled: bool = False
    """未指定语音合成模型时，使用 DeepSeek TTS。"""
    voice: Literal["stella", "tide"] = "stella"
    """语音合成默认音色：stella（海星女声）或 tide（暗潮男声）。"""
    timeout: int = Field(default=180, ge=10, le=300)
    """单次工具请求的总超时秒数；超时不自动重新上传或重放。"""


class AgentEngineSettings(BaseModel):
    """全局默认的 Agent 推理引擎配置"""

    max_cycles: int = Field(default=10, ge=1)
    """工具调用最大循环次数"""
    global_max_cycles: int = Field(default=30, ge=1)
    """整个会话生命周期内（跨嵌套智能体）的大模型绝对循环次数上限，用于防死循环"""
    enable_parallel_calls: bool = True
    """允许并行工具调用"""
    reflexion_retries: int = Field(default=1, ge=0)
    """工具未单独配置 max_retries 时允许的连续出错次数"""
    enable_fallback_summary: bool = True
    """达到最大循环次数时，是否触发大模型兜底总结（而不是直接报错）"""
    tool_output_max_chars: int = Field(default=16000, ge=0)
    """单个工具结果交给模型的最大字符数，超出部分截去中间；0 表示不限制"""
    tool_timeout: int = Field(default=300, ge=0)
    """单次工具调用的最长执行时间(秒)，超时后作为错误结果交给模型；0 表示不限制"""
    mcp_cleanup_timeout: int = Field(default=900, ge=0)
    """MCP 服务自动清理的闲置超时时间(秒)。0表示关闭自动清理机制（永久驻留）"""


class SandboxSettings(BaseModel):
    """Agent 代码沙箱（Docker）配置"""

    enable_sandbox: bool = Field(default=False)
    """是否启用沙箱；启用且 Docker 与镜像可用时才向 Agent 提供执行类工具"""
    docker_image: str = Field(default="python:3.12-slim")
    """沙箱容器使用的镜像"""
    cpus: float = Field(default=1.0, ge=0.1, le=64)
    """单个沙箱可用的 CPU 核数"""
    memory_mb: int = Field(default=512, ge=64, le=65536)
    """单个沙箱可用的内存(MB)"""
    exec_timeout: int = Field(default=60, ge=5, le=3600)
    """单次执行代码或命令的超时时间(秒)"""
    network: bool = Field(default=False)
    """沙箱是否可以联网"""
    cleanup_timeout: int = Field(default=1800, ge=0)
    """沙箱闲置超过该时间(秒)后删除容器；0 表示不自动清理"""

    @field_validator("docker_image")
    @classmethod
    def _default_image(cls, value: str) -> str:
        value = value.strip()
        return "python:3.12-slim" if value in {"", "zhenxun-sandbox:latest"} else value


class LLMConfig(BaseModel):
    """AI 模块全局持久化配置总模型"""

    default_models: DefaultModelsConfig = Field(default_factory=DefaultModelsConfig)
    """全局按任务分类的默认模型路由表"""
    client_settings: ClientSettings = Field(default_factory=ClientSettings)
    """客户端通用连接配置"""
    providers: list[ProviderConfig] = Field(default_factory=list)
    """已配置的提供商列表"""
    debug_log: DebugLogOptions = Field(default_factory=DebugLogOptions)
    """日志调试开关配置"""
    context_settings: ContextManagementSettings = Field(
        default_factory=ContextManagementSettings
    )
    """上下文管理相关配置"""
    model_groups: dict[str, list[str]] = Field(default_factory=dict)
    """虚拟模型路由组配置 (Virtual Router Groups)"""
    agent_settings: AgentEngineSettings = Field(default_factory=AgentEngineSettings)
    """Agent 执行引擎层核心默认参数配置"""
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    """沙箱基础设施环境相关配置"""
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    """联网搜索设置"""
    deepseek_tools: DeepSeekToolsSettings = Field(default_factory=DeepSeekToolsSettings)
    """DeepSeek 独立 OCR、语音识别与语音合成设置"""
