import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Any

from nonebot_plugin_alconna import UniMessage

from zhenxun.models.ai_chat_group import AIChatGroup
from zhenxun.models.ai_persona import AIPersona
from zhenxun.models.ai_token_usage import AITokenUsage
from zhenxun.services.ai.context.memory.engine import chat_backend
from zhenxun.services.ai.context.memory.models import HistorySettlement
from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.message_segments import is_voice
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.run.session import session_manager
from zhenxun.services.log import logger

from ._agent import record_run, start_run
from ._data_source import MODULE
from ._reply import DeliveryResult

FAILED_NOTICE = "这次回复没有完成，请稍后再试。"
"""回复失败且人设没有设置失败提示时的提示语"""
AfterSettle = Callable[[dict, BaseException | None], Awaitable[None]]
"""结算时在写运行记录前执行的附加步骤，参数为投递报告与异常"""


class ChatTurn:
    """一轮聊天的登记与结算：生成前分配运行id并标记本轮写入的历史；结束后按实际
    送达的内容改写短期记忆、计入用量并写运行记录。成功、失败或被取消都只结算一次，
    结算不会被取消打断"""

    def __init__(
        self,
        context: RunContext,
        agent: Agent,
        source: str,
        persona: AIPersona | None,
        text: str,
        sender: str,
        *,
        usage_id: str | None = None,
        revision: int | None = None,
    ):
        """
        参数:
            context: 运行上下文
            agent: 运行的 Agent
            source: 来源：qq、webui、task 或 interject
            persona: 人设
            text: 写入运行记录的输入文本
            sender: 发送者标识
            usage_id: 计入每日用量的用户，为空时不计入
            revision: 发起时的会话修订号，为空时取当前值
        """
        session_id = context.session_id or ""
        self.context = context
        self.agent = agent
        self.source = source
        self.persona = persona.name if persona else None
        self.text = text
        self.usage_id = usage_id
        self.revision = (
            session_manager.revision(session_id) if revision is None else revision
        )
        self.run_id = start_run(context)
        session_manager.register(context)
        context.run.root_session_id = session_id
        context.run.root_revision = self.revision
        self.started, self.started_at = time.monotonic(), time.time()
        self.result: Any = None
        self.delivery = DeliveryResult()
        self._task: asyncio.Task | None = None
        context.run.history_metadata = {
            "chat_run": self.run_id,
            "chat_revision": self.revision,
            "chat_settled": False,
            "source": source,
            "sender": sender,
            "persona": self.persona,
        }

    def select_agent(self, agent: Agent, persona: AIPersona | None) -> None:
        """在执行窗口选定 Agent，并让历史及结算采用同一人设。"""
        self.agent = agent
        self.persona = persona.name if persona else None
        self.context.run.history_metadata["persona"] = self.persona

    @property
    def current(self) -> bool:
        """会话在本轮开始后没有被重置"""
        return self.revision == session_manager.revision(self.context.session_id or "")

    @property
    def tokens(self) -> int:
        """本轮已知的模型用量，含子代理"""
        fallback = self.result.usage.total_tokens if self.result else 0
        return int(self.context.run.budget.get("tokens", fallback))

    @property
    def tool_texts(self) -> list[str]:
        """本轮通过发送工具发出的文本"""
        return [
            entry["text"]
            for record in self.context.run.operations.get("deliveries", {}).values()
            for entry in record["entries"]
            if entry["origin"] == "tool"
        ]

    def report(self, error: BaseException | None = None) -> dict:
        """汇总投递结果：本轮回复（含失败提示）确认送达时为已送达，出错前只由发送工具
        发出部分内容时为部分送达

        参数:
            error: 本轮的异常

        返回:
            dict: 投递报告
        """
        report = self.delivery.as_dict()
        operations = self.context.run.operations
        if attachment := operations.get("attachment"):
            report["attachment"] = dict(attachment)
        records = list(operations.get("deliveries", {}).values())
        entries = [entry for record in records for entry in record["entries"]]
        report["text"] = "\n".join(
            [entry["text"] for entry in entries] if entries else self.delivery.parts
        )
        report["uncertain"] |= any(record["uncertain"] for record in records)
        final_sent = (
            not entries and self.delivery.status == "sent" and bool(self.delivery.parts)
        )
        failed = (
            (bool(error) and not final_sent)
            or any(record["failed"] for record in records)
            or report["status"] in {"failed", "partial"}
        )
        if report["uncertain"]:
            report["status"] = "partial" if report["text"] else "unknown"
        elif failed:
            report["status"] = "partial" if report["text"] else report["status"]
        elif report["text"]:
            report["status"] = "sent"
        if entries:
            report["receipts"] = [entry["receipt"] for entry in entries]
        report["deduplicated"] = sum(record["deduplicated"] for record in records)
        report["reused_operations"] = operations.get("reused_operations", 0)
        report["permission_reason"] = operations.get("permission_reason")
        report["operation_uncertain"] = (
            bool(operations.get("uncertain_tools"))
            or any(
                item.get("status") == "unknown"
                for item in operations.get("dsh_tools", {}).values()
            )
            or operations.get("dsh", {}).get("stop") == "unknown"
        )
        report["tool_conflicts"] = self.context.state.get("_tool_conflicts", [])
        report["contributors"] = sorted(
            {entry["agent"] for entry in entries if entry["agent"]}
        )
        from ._reply import parse_silent

        report["effective_reply"] = bool(
            self.tool_texts
            or (
                self.delivery.parts
                and self.result
                and not self.result.stopped
                and parse_silent(str(self.result.output))
            )
        )
        return report

    async def reply(
        self,
        value: Any,
        error: BaseException | None,
        message: UniMessage,
        setting: AIChatGroup | None,
        persona: AIPersona | None,
    ) -> None:
        """投递本轮回复：失败时提示用户；主动插话遇到排队中的定向请求时放弃；
        定向回复为空（不含模型选择沉默）时给出兜底文本；已由发送工具发出的内容不重复
        发送

        参数:
            value: 运行结果，失败时为空
            error: 运行异常
            message: 用户消息
            setting: 群聊设置
            persona: 人设
        """
        from zhenxun.services.ai.core.exceptions import (
            InterventionHandledException,
            LLMException,
            RunInterrupted,
        )

        from ._data_source import AIChatManager
        from ._reply import clean_output, parse_silent, send_reply
        from ._stickers import send_marked, split_sticker

        self.result = value
        context, delivery = self.context, self.delivery
        group_id = context.get_group_id()
        directed = self.source != "interject"
        custom = persona.error_message if persona else None
        if error is not None:
            if isinstance(error, RunInterrupted):
                if not self.report(error).get(
                    "uncertain"
                ) and not context.run.operations.get("notice_sent"):
                    context.run.operations["notice_sent"] = True
                    await send_reply(
                        "本次执行因通信异常中断，未自动重试。",
                        group_id,
                        result=delivery,
                        context=context,
                    )
                return
            interrupted = isinstance(
                error, asyncio.CancelledError | InterventionHandledException
            )
            if not interrupted and directed and not self.tool_texts:
                notice = context.run.user_notice or custom or FAILED_NOTICE
                if (
                    isinstance(error, LLMException)
                    and (error.details or {}).get("code") == "follow_up_unconsumed"
                ):
                    notice = "上一轮已结束，追加消息未被处理，请重新发送。"
                await send_reply(notice, group_id, result=delivery, context=context)
            elif delivery.status == "pending":
                delivery.status = "skipped" if interrupted else "failed"
            return
        if value.stopped:
            delivery.status = "skipped"
            return
        if not directed and AIChatManager.directed_pending(context.session_id or ""):
            delivery.status = "skipped"
            return
        if payload := context.state.pop("_qq_attachment", None):
            from ._attachments import send_export

            await send_export(context, payload, delivery)
            if delivery.status == "failed":
                await send_reply(
                    "附件未发送：" + (delivery.error or "平台拒绝了上传"),
                    group_id,
                    result=delivery,
                    context=context,
                )
            return
        if context.run.operations.get("attachment", {}).get("name"):
            return
        reply = parse_silent(str(value.output))
        empty = not clean_output(str(value.output)).strip()
        if empty and directed and not self.tool_texts:
            reply = custom or "这次没能整理出有效回复，请再试一次。"
        sticker_id = None
        if reply:
            reply, sticker_id = split_sticker(reply)
        if reply:
            voice = setting.voice_reply if setting and setting.voice_reply else None
            voice_input = any(is_voice(seg) for seg in message)
            await send_reply(
                reply, group_id, voice_input, voice, delivery, context=context
            )
        else:
            delivery.status = "skipped"
        if sticker_id is not None and delivery.status in {"sent", "skipped"}:
            await send_marked(context, sticker_id)

    async def settle(
        self, error: BaseException | None = None, after: AfterSettle | None = None
    ) -> None:
        """结算本轮，重复调用时等待第一次结算完成

        参数:
            error: 本轮的异常
            after: 写运行记录前执行的附加步骤
        """
        if self._task is None:
            self._task = asyncio.create_task(self._settle(error, after))
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            await asyncio.shield(self._task)
            raise

    async def _settle(
        self, error: BaseException | None, after: AfterSettle | None
    ) -> None:
        report = self.report(error)
        self.context.state["_delivery"] = report
        if self.usage_id and (used := self.tokens):
            try:
                await AITokenUsage.add_usage(self.usage_id, used)
            except Exception as exc:
                report["usage_error"] = type(exc).__name__
                logger.warning("本轮用量结算未确认", MODULE)
        receipt = HistorySettlement()
        backend = chat_backend(self.agent.memory_config)
        if backend:
            try:
                receipt = await backend.settle_run(
                    self.context.session_id or "",
                    self.run_id,
                    self.revision,
                    report,
                    self.started_at,
                )
            except Exception as exc:
                report["history_error"] = type(exc).__name__
                logger.warning("本轮历史保存未确认", MODULE)
        self._confirm_background(receipt, report, history_enabled=backend is not None)
        if after:
            try:
                await after(report, error)
            except Exception as exc:
                logger.warning(f"本轮结算后续步骤失败: {type(exc).__name__}", MODULE)
        await record_run(
            self.context,
            self.source,
            self.persona,
            self.text,
            self.started,
            result=self.result,
            error=error,
        )

    def _confirm_background(
        self, receipt, report: dict, *, history_enabled: bool
    ) -> None:
        """仅确认当前修订中有效送达且输入已保存的背景区段。"""
        from ._buffer import ChatBuffer

        snapshots = self.context.state.get("_background_snapshots", {})
        if not snapshots:
            return
        confirmed = 0
        reason = "no_effective_delivery"
        if not self.current:
            reason = "stale_revision"
        elif report.get("effective_reply"):
            claims = (
                set(snapshots)
                if not history_enabled
                else receipt.visible_claims
                if isinstance(receipt, HistorySettlement)
                and receipt.status == "confirmed"
                else set()
            )
            for claim in claims:
                if snapshot := snapshots.get(claim):
                    confirmed += ChatBuffer.confirm(
                        snapshot, self.run_id, self.revision
                    )
            reason = "history_unconfirmed" if confirmed < len(snapshots) else None
        self.context.state["_background_receipt"] = {
            "status": "confirmed" if confirmed == len(snapshots) else "unconfirmed",
            "read": len(snapshots),
            "confirmed": confirmed,
            "truncated": any(s.truncated for s in snapshots.values()),
            "reason": reason,
        }
