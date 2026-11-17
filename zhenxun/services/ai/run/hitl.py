from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ClassVar
import uuid

from nonebot.adapters import Bot, Event
from nonebot_plugin_waiter import waiter

from zhenxun.services.ai.core.exceptions import AbortException, ToolFatalError
from zhenxun.services.ai.utils.logger import log_agent as logger

from .context import RunContext

CANCEL_WORDS = {"取消", "cancel", "0", "退出", "quit"}


class HITLController:
    """
    人机协同控制器 (Human-in-the-Loop Controller)。
    封装底层的物理环境隔离等待逻辑，供上层工具和中间件发起提问或审批。
    """

    pending: ClassVar[dict[str, tuple[RunContext, asyncio.Future, str]]] = {}

    def __init__(self, context: RunContext):
        self.context = context

    async def wait_event(
        self,
        prompt_msg: str | None = None,
        timeout: float = 60.0,
        custom_checker: Callable[[Event], bool] | None = None,
        strict_user: bool = True,
    ) -> Event | None:
        """
        底层交互方法：进行严格群组与用户隔离的挂起等待。
        """
        bot = self.context.get_bot()
        event = self.context.get_event()

        if not bot or not event:
            logger.warning("HITLController: 当前环境无 Bot/Event 实例，无法发起交互。")
            raise ToolFatalError(
                "当前处于自动化后台调度环境，无法发起人工交互。任务已强行中止。"
            )

        if prompt_msg:
            await bot.send(event, prompt_msg)

        orig_user_id = self.context.get_user_id()
        orig_group_id = self.context.get_group_id()

        @waiter(waits=["message"], keep_session=False)
        async def event_waiter(event_bot: Bot, e: Event):
            from .context import NoneBotDeps

            if (
                str(event_bot.self_id) != str(bot.self_id)
                or event_bot.adapter.get_name() != bot.adapter.get_name()
            ):
                return None
            current = RunContext(deps=NoneBotDeps(bot=event_bot, event=e))
            if current.get_group_id() != orig_group_id:
                return None

            if strict_user and current.get_user_id() != orig_user_id:
                return None

            if custom_checker and not custom_checker(e):
                return None

            return e

        task = asyncio.create_task(event_waiter.wait(timeout=timeout))
        cancellation_token = self.context.run.cancellation_token
        if cancellation_token:
            cancellation_token.link_future(task)

        try:
            return await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def ask_text(self, prompt_msg: str, timeout: float = 60.0) -> str:
        """
        高阶交互：发起文本提问，内置超时与取消词检测。
        如用户取消或超时，自动抛出异常切断 Agent 思考循环。
        """
        question_id = uuid.uuid4().hex[:12]
        answer = asyncio.get_running_loop().create_future()
        self.pending[question_id] = (self.context, answer, prompt_msg)
        task = asyncio.create_task(
            self.wait_event(
                f"{prompt_msg}\n问题 {question_id}，"
                f"也可使用 /ai answer {question_id} <回答>",
                timeout=timeout,
                strict_user=True,
                custom_checker=lambda event: not event.get_plaintext()
                .strip()
                .startswith("/ai "),
            )
        )
        try:
            done, _ = await asyncio.wait(
                {task, answer}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if answer in done:
                user_input = answer.result().strip()
            elif task in done and (event := task.result()) is not None:
                user_input = event.get_plaintext().strip()
            else:
                raise ToolFatalError("参数收集超时，任务已中止。")
        finally:
            self.pending.pop(question_id, None)
            answer.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if user_input.lower() in CANCEL_WORDS:
            raise AbortException(reason="用户主动取消了操作")

        return user_input

    @classmethod
    async def command(cls, context: RunContext, action: str, argument: str) -> str:
        """按可信调用者和运行归属查询或回答问题，迟到回答不重新放行。"""
        from zhenxun.services.ai.utils.runtime import PermissionUtils

        caller = context.run.caller
        entries = {}
        for ident, (owner, future, prompt) in list(cls.pending.items()):
            other = owner.run.caller
            if (
                caller
                and other
                and not future.done()
                and (
                    caller.source,
                    caller.platform,
                    caller.bot_id,
                    caller.user_id,
                    caller.group_id,
                )
                == (
                    other.source,
                    other.platform,
                    other.bot_id,
                    other.user_id,
                    other.group_id,
                )
            ):
                entries[ident] = (owner, future, prompt)
        if action == "approval":
            return (
                "\n".join(
                    f"{ident}: {entry[2][:300]}" for ident, entry in entries.items()
                )
                or "当前没有等待你处理的问题。"
            )
        ident, _, answer = argument.partition(" ")
        if not answer.strip() or ident not in entries:
            raise ValueError("问题已失效、不属于你，或回答为空。")
        owner, future, _ = entries[ident]
        PermissionUtils.ensure_current(owner)
        future.set_result(answer.strip())
        return "已提交本次回答。"
