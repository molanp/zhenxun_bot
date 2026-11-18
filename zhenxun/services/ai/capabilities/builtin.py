from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.exceptions import (
    AbortException,
    LLMException,
    ModelRetry,
    ResponseParseException,
    ToolFatalError,
    ToolRetryError,
    UpstreamServerException,
)
from zhenxun.services.ai.core.messages import (
    ChatRequest,
    ChatResponse,
    LLMMessage,
)
from zhenxun.services.ai.core.models import LLMContext
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.utils.logger import log_capability as logger
from zhenxun.services.ai.utils.runtime import PermissionUtils

from .base import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)


def _get_tool_meta(tool: Any, key: str, default: Any = None) -> Any:
    """辅助方法：安全地提取工具元数据中指定的键值"""
    if not tool:
        return default
    settings = getattr(tool, "settings", None)
    meta = (settings.metadata if settings else None) or getattr(tool, "metadata", {})
    return meta.get(key, default)


class StuckDetectionCapability(AbstractCapability):
    """相同工具参数连续重复时分级提醒，超过五次不再执行。"""

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        from zhenxun.services.ai.tools.models import ToolResult

        signature = json.dumps(
            arguments, sort_keys=True, ensure_ascii=False, default=str
        )
        history = context.state.setdefault("_tool_repetitions", {})
        previous, count = history.get(tool_name, (None, 0))
        count = count + 1 if signature == previous else 1
        history[tool_name] = (signature, count)
        if count >= 5:
            context.state.setdefault("_disabled_tools", set()).add(tool_name)
            return ToolResult(
                output=(
                    "这个工具的相同调用已重复多次，本轮停止调用，" "请用已有信息回答。"
                )
            ).as_error(False)
        result = await handler(arguments)
        if count >= 3 and isinstance(result, ToolResult):
            result.output = (
                f"{result.output}\n[提醒] 相同调用已重复 {count} 次，"
                "请停止重复并总结现有结果。"
            )
        return result


class GlobalCycleLimitCapability(AbstractCapability):
    """全局防死循环检测中间件：跨 Agent 追踪大模型调用总次数"""

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext[ChatRequest, ChatResponse],
        handler: WrapModelRequestHandler,
    ) -> ChatResponse:
        global_cycles = (
            context.session.shared_state.get("__global_cycle_count__", 0) + 1
        )
        context.session.shared_state["__global_cycle_count__"] = global_cycles

        global_max = get_llm_config().agent_settings.global_max_cycles
        if global_cycles > global_max:
            logger.error(
                "🚨 触发全局防护：整个流水线执行步数已达到全局上限 "
                f"({global_max})，强制熔断！"
            )
            raise AbortException(
                reason=f"全局大模型思考循环次数已超限 ({global_max}次)",
                display="🚨 系统保护触发：任务过于复杂或陷入多智能体死循环，"
                "已被强行中断以节省资源。",
            )

        return await handler(llm_context)


class PermissionCapability(AbstractCapability):
    """权限校验中间件：在执行前根据确定参数进行动态鉴权"""

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> dict[str, Any]:
        tool = context.call.current_tool
        admin_level = _get_tool_meta(tool, "admin_level", 0)
        if admin_level > 0:
            if not await PermissionUtils.check_admin_level(context, admin_level):
                context.run.operations["permission_reason"] = context.state.get(
                    "_permission_reason", "缺少群管理授权"
                )
                msg = (
                    "系统警告：用户权限不足（仅限群主或管理员）。"
                    "请温和地向用户解释权限不足，并拒绝执行。"
                )
                user_id = context.get_user_id()
                logger.warning(
                    f"🛡️ [Capability] 权限拦截: 用户 {user_id} 尝试调用 "
                    f"{getattr(tool, 'name', 'unknown')}"
                )
                raise ToolFatalError(
                    msg, display_content="❌ 权限不足: 仅限群主或管理员"
                )
        return await handler(arguments)


class ToolRetryAndReflectionCapability(AbstractCapability):
    """连续失败回填模型；不可重试或结果未知的写工具本轮禁用。"""

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        from zhenxun.services.ai.tools.models import ToolResult

        disabled = context.state.setdefault("_disabled_tools", set())
        if tool_name in disabled:
            return ToolResult(
                output="该工具本轮已停用，请根据已有信息回答，不要换参数重试。"
            ).as_error(False)
        try:
            result = await handler(arguments)
        except Exception as error:
            from zhenxun.services.ai.run.context import interrupt_transport

            interrupt_transport(error, "tool")
            readonly = bool(
                _get_tool_meta(context.call.current_tool, "read_only", False)
            ) and not isinstance(error, PermissionError)
            result = ToolResult(
                output=f"工具执行失败（{type(error).__name__}），未确认完成。"
            ).as_error(
                readonly,
                uncertain=not readonly and not isinstance(error, PermissionError),
            )
            logger.warning(f"工具 {tool_name} 失败: {type(error).__name__}")
        if not isinstance(result, ToolResult) or not result.is_error:
            context.run.tool_retries.pop(tool_name, None)
            return result
        failures = context.run.tool_retries.get(tool_name, 0) + 1
        context.run.tool_retries[tool_name] = failures
        if not result.is_retryable or failures > context.run.tool_max_retries:
            disabled.add(tool_name)
            result.output = (
                f"{result.output}\n该工具本轮停用。不要再次执行，"
                "请用已有信息回答并说明未完成部分。"
            )
            result.is_retryable = False
        return result


class ReflexionCapability(AbstractCapability):
    """自愈反思与验证引擎 (Reflexion Engine)。
    统一处理结构化解析失败和语义护栏拦截。"""

    _PROMPT_TEMPLATES: ClassVar[dict[str, str]] = {
        "schema_validation_error": (
            "上一次输出的 JSON 格式正确，但部分字段没有通过校验：\n"
            "{error_msg}\n"
            "请只修改出错的字段，按要求重新输出完整的 JSON。"
        ),
        "schema_parse_error": (
            "上一次输出的 JSON 无法解析，或字段类型不符：\n"
            "{error_msg}\n"
            "请检查缺失的必填字段、错误的类型和未闭合的括号，"
            "按 Schema 重新输出完整的 JSON。"
        ),
        "default_retry": (
            "上一次输出或工具调用没有通过校验：\n"
            "{error_msg}\n"
            "请根据上面的错误修正后重试。"
        ),
    }

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        try:
            return await handler(arguments)
        except Exception as error:
            from zhenxun.services.ai.run.context import interrupt_transport

            interrupt_transport(error, "tool")
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(error, ToolRetryError | ModelRetry):
                error_msg = getattr(error, "message", str(error))
                feedback_prompt = self._PROMPT_TEMPLATES["default_retry"].format(
                    error_msg=error_msg
                )
                context.run.add_system_prompt(feedback_prompt)
                return ToolResult(
                    output=f"执行失败：{error_msg}",
                ).as_error()
            raise error.with_traceback(None) from None

    def _extract_error_info(
        self, e: Exception, current_response_text: str
    ) -> tuple[str, str, bool]:
        """
        统一提取异常报错详情与可恢复标记
        返回 (error_msg, raw_response, is_recoverable)
        """
        is_model_retry = isinstance(e, ModelRetry)
        is_llm_error = isinstance(e, LLMException)
        llm_error = cast(LLMException, e) if is_llm_error else None

        if (
            not is_model_retry
            and llm_error
            and not isinstance(
                llm_error, ResponseParseException | UpstreamServerException
            )
        ):
            raise e

        if is_model_retry:
            error_msg = getattr(e, "message", str(e))
            raw_response = current_response_text
        else:
            error_msg = (
                llm_error.details.get("validation_error", str(e))
                if llm_error
                else str(e)
            )
            raw_response = current_response_text or (
                llm_error.details.get("raw_response", "") if llm_error else ""
            )

        is_recoverable = getattr(llm_error, "recoverable", True) if llm_error else True
        return error_msg, raw_response, is_recoverable

    def _generate_feedback_prompt(
        self, e: Exception, error_msg: str, error_template: str | None
    ) -> str:
        """
        基于异常多态与模板字典生成自愈反思提示词
        """
        if error_template:
            return error_template.format(error_msg=error_msg)

        template_name = (
            e.get_template_name() if isinstance(e, ModelRetry) else "default_retry"
        )

        template = self._PROMPT_TEMPLATES.get(
            template_name, self._PROMPT_TEMPLATES["default_retry"]
        )
        context_data = e.get_feedback_context() if isinstance(e, ModelRetry) else {}
        context_data["error_msg"] = error_msg

        return template.format(**context_data)

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext[ChatRequest, ChatResponse],
        handler: WrapModelRequestHandler,
    ) -> ChatResponse:
        output_processor = llm_context.request.extra.get("output_processor")
        if not output_processor:
            return await handler(llm_context)

        max_retries = llm_context.request.extra.get("max_retries", 3)
        error_template = output_processor.error_template

        ivr_messages = list(llm_context.request.messages)
        last_exception: Exception | None = None

        for attempt in range(max_retries + 1):
            llm_context.request.messages = list(ivr_messages)
            current_response_text: str = ""

            try:
                response = await handler(llm_context)
                current_response_text = response.text

                if response.tool_calls:
                    return response

                final_obj = await output_processor.validate_and_parse(
                    current_response_text, context=context
                )
                response.parsed_obj = final_obj
                return response

            except Exception as e:
                last_exception = e
                try:
                    error_msg, raw_response, is_recoverable = self._extract_error_info(
                        e, current_response_text
                    )
                except Exception as fatal_e:
                    raise fatal_e.with_traceback(None) from None

                if attempt < max_retries:
                    logger.warning(
                        "输出校验未通过 "
                        f"(尝试 {attempt + 1}/{max_retries + 1})。"
                        f"启动反思修复闭环... 失败原因: {error_msg}"
                    )

                    if raw_response:
                        ivr_messages.append(
                            cast(
                                LLMMessage,
                                LLMMessage.assistant_text_response(raw_response),
                            )
                        )

                    feedback_prompt = self._generate_feedback_prompt(
                        e, error_msg, error_template
                    )
                    ivr_messages.append(
                        cast(LLMMessage, LLMMessage.user(feedback_prompt))
                    )
                    continue

                if not is_recoverable:
                    raise last_exception.with_traceback(None) from None

        if last_exception:
            raise last_exception.with_traceback(None) from None
        raise UpstreamServerException(
            "反思循环耗尽，未能生成符合所有校验规则的合法结果。",
        ).with_traceback(None) from None


def default_capabilities() -> list[AbstractCapability]:
    """每次运行默认挂载的内置能力

    返回:
        list[AbstractCapability]: 内置能力实例列表
    """
    return [
        StuckDetectionCapability(),
        GlobalCycleLimitCapability(),
        PermissionCapability(),
        ToolRetryAndReflectionCapability(),
        ReflexionCapability(),
    ]
