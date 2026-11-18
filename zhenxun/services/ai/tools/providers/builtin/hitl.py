from zhenxun.services.ai.core.exceptions import AbortException, ToolFatalError
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.run.hitl import HITLController
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ResolvedToolPayload, ToolResult
from zhenxun.services.ai.utils.logger import log_tool as logger


class HITLToolkit(BaseToolkit):
    """
    Human-in-the-Loop (HITL) 工具箱。
    提供大模型主动挂起执行流并向用户提问求助的能力。
    """

    default_prefix = ""

    default_instructions = (
        "## 向用户确认\n"
        "缺少关键信息、意图不明确或同一操作反复失败时，用 `ask_user_for_help` "
        "向当前用户提问，拿到回答后继续；能自己判断的不要问。"
    )

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        """智能环境感知：如果处于无真实用户交互的后台环境(如定时任务)，自动隐身以节省Token"""
        if context is None or context.get_event() is None:
            return ResolvedToolPayload()
        return await super().resolve(context)

    @tool(
        name="ask_user_for_help",
        description="向当前对话的用户提一个问题并等待回答，用于补全关键信息或确认操作意图。",
    )
    async def ask_user_for_help(
        self,
        question: str,
        context: RunContext,
    ) -> ToolResult:
        try:
            user_reply = await HITLController(context).ask_text(
                f"🤖 [AI 提问]\n{question}\n\n(请在 60 秒内回复，或回复'取消')",
                timeout=60.0,
            )
        except ToolFatalError:
            return ToolResult(
                output="用户未在规定时间内回复 (超时未响应)，"
                "你必须尝试自行解决或中止任务。",
            ).as_error()
        except AbortException as e:
            raise e

        logger.info(f"收到用户求助回复: {user_reply}")
        return ToolResult(output=f"用户提供的回复是: {user_reply}")
