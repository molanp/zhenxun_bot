"""
自定义异常与错误码定义
"""

from typing import Any


class ModelRetry(Exception):
    """用于通知大模型修正并重试的异常"""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        """
        初始化用于通知大模型重试的异常。

        参数：
            message: 用于提示大模型的具体重试和自我纠错信息。
            payload: 携带产生错误时的上下文状态字典，用于渲染反馈提示词。
        """
        self.message = message
        self.payload = payload or {}
        super().__init__(self.message)

    def get_template_name(self) -> str:
        """多态：子类告诉渲染引擎，自己应该使用哪个 Prompt 模板标识"""
        return "default_retry"

    def get_feedback_context(self) -> dict[str, Any]:
        """多态：子类提供渲染模板所需的领域数据上下文"""
        return {"error_msg": self.message, **self.payload}


class SchemaParseError(ModelRetry):
    """格式解析异常。当大模型返回的 JSON 损坏或不符合 Schema 时抛出。"""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        """
        初始化 Schema 格式解析错误异常。

        参数：
            message: 详细的 JSON 解析失败或 Schema 校验报错信息。
        """
        super().__init__(message, payload)

    def get_template_name(self) -> str:
        return "schema_parse_error"


class SchemaValidationError(ModelRetry):
    """数据校验异常。当大模型返回的 JSON 格式正确，但业务字段约束不满足时抛出。"""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        """
        初始化 Schema 业务字段约束验证错误异常。
        """
        super().__init__(message, payload)

    def get_template_name(self) -> str:
        return "schema_validation_error"


class ControlFlowExit(BaseException):
    """控制流退出基类，继承自BaseException以避免被常规Exception捕获，用于静默中断。"""

    pass


class RunInterrupted(ControlFlowExit):
    """本轮通信未完成，禁止恢复请求或让模型继续执行。"""

    def __init__(self, stage: str):
        self.reason = "network"
        self.stage = stage
        super().__init__(f"本轮在 {stage} 阶段中断")


class RunBudgetExceeded(ControlFlowExit):
    """根运行及子代理共享的模型请求预算已经用完。"""


class ToolFatalError(ControlFlowExit):
    """
    致命工具异常（不可恢复）。
    当工具执行遇到权限不足、严重系统故障等大模型无法通过重试解决的问题时抛出。
    这会直接熔断 Agent 推理流，并将 display_content 抛给用户。
    """

    def __init__(self, message: str, display_content: str | None = None):
        """
        初始化不可恢复的致命工具执行异常。

        参数：
            message: 供大模型及系统调试日志记录的底层致命错误详情。
            display_content: 直接向终端用户呈现的友好拦截文案。
        """
        self.message = message
        self.display_content = display_content or f"❌ 工具遇到致命错误: {message}"
        super().__init__(self.message)


class ToolRetryError(Exception):
    """
    可恢复工具异常。
    当参数解析错误、业务逻辑校验失败、网络超时等问题发生时抛出。
    会被 ToolExecutor 捕获并转化为引导大模型自我反思 (Reflexion) 的 ToolResult。
    """

    def __init__(self, message: str):
        """
        初始化触发大模型反思与自愈的可恢复工具错误。

        参数：
            message: 会被传递给大模型用于进行 Reflexion 的错误反馈 Prompt。
        """
        self.message = message
        super().__init__(self.message)


class AbortException(ControlFlowExit):
    """异常中止当前 Agent 思考流。"""

    def __init__(self, reason: str, display: Any = None):
        """
        初始化用于强制中止 Agent 推理执行流的异常。

        参数：
            reason: 触发强制中断的技术或业务原因。
            display: 中断后向用户展示的显示结果。
        """
        self.reason = reason
        self.display = display
        super().__init__(f"Aborted: {reason}")


class InterventionHandledException(ControlFlowExit):
    """
    干预成功处理异常。
    当用户的消息被追加到同一会话正在运行的 Agent 任务时抛出，
    用于中断当前的新请求生命周期，避免重复启动。
    """

    def __init__(self, message: str, display_content: str | None = None):
        """
        初始化干预处理成功以安全熔断生命周期的异常。

        参数：
            message: 内部调试与审计的干预详情描述。
            display_content: 向发起干预的用户端展现的进度提醒提示。
        """
        self.message = message
        self.display_content = display_content
        super().__init__(self.message)


class LLMException(Exception):
    """LLM 服务相关的基础异常类 (多态基类)"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        初始化底层大模型 API 调用及服务异常。

        参数：
            message: 通用的调用错误或失败总结说明。
            details: 包含接口名、重试指示、服务端回传原始信息的字典。
            cause: 触发此错误的根源协议请求异常实例。
        """
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(message)

    @property
    def is_retryable(self) -> bool:
        """是否允许在当前节点进行退避重试（如偶发网络抖动）"""
        return False

    @property
    def should_failover(self) -> bool:
        """是否允许触发节点故障转移（切换到下一个备用模型）"""
        return False

    @property
    def should_rotate_key(self) -> bool:
        """是否应该标记当前 Key 失效并轮换 API Key"""
        return False

    @property
    def user_friendly_message(self) -> str:
        """返回适合向用户展示的错误消息"""
        return "AI服务暂时不可用，请稍后再试。"

    def __str__(self) -> str:
        if self.details:
            safe_details = {k: v for k, v in self.details.items() if k != "api_key"}
            if safe_details:
                return f"{self.message} (详情: {safe_details})"
        return self.message


class InvalidRequestException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "请求参数错误或API类型不支持，请检查输入内容。"


class ContextLengthExceededException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "输入内容过长，请缩短后重试。"


class ContentFilteredException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "内容被安全过滤，请修改后重试。"


class ConfigurationException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI模型配置错误或未找到，请联系管理员检查配置。"


class AuthenticationException(LLMException):
    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "API密钥无效或权限不足，请联系管理员更新配置。"


class PermissionDeniedException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "当前密钥无权访问该模型或接口，请联系管理员检查配置。"


class QuotaExceededException(LLMException):
    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "API使用配额已用尽，请稍后再试或联系管理员。"


class LocationNotSupportedException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return (
            "当前网络环境不支持此 AI 模型。\n"
            "建议: 请尝试更换代理节点至支持的地区或切换备用模型。"
        )


class RateLimitException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "请求过于频繁，已被AI服务限流，请稍后再试。"


class UpstreamServerException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务响应异常或端点宕机，请稍后再试。"


class NetworkTimeoutException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务请求超时，请稍后再试。"


class ResponseParseException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务响应解析失败，请稍后再试。"


class EmptyResponseException(ResponseParseException):
    """模型正常结束但没有返回任何内容"""

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务返回了空回复，请稍后再试。"


class NoImageException(ResponseParseException):
    """生图请求的响应中没有图片，重试会再次计费，不在原线路重试"""

    @property
    def is_retryable(self) -> bool:
        return False

    @property
    def user_friendly_message(self) -> str:
        return "模型没有生成图片，请调整描述后再试。"


def get_user_friendly_error_message(error: Exception) -> str:
    """将任何异常转换为用户友好的错误消息"""
    if isinstance(error, LLMException):
        return error.user_friendly_message

    error_str = str(error).lower()

    if "timeout" in error_str or "timed out" in error_str:
        return "网络请求超时，请检查服务器网络或代理连接。"
    if "connect" in error_str and ("refused" in error_str or "error" in error_str):
        return "无法连接到 AI 服务商，请检查网络连接或代理设置。"
    if "proxy" in error_str:
        return "代理连接失败，请检查代理服务器是否正常运行。"
    if "ssl" in error_str or "certificate" in error_str:
        return "SSL 证书验证失败，请检查网络环境。"

    return f"服务暂时不可用 ({type(error).__name__})，请稍后再试。"
