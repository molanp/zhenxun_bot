import asyncio
import json
from typing import Any

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.engine.token_counter import (
    parse_usage_info,
    token_counter,
)
from zhenxun.services.ai.core.exceptions import LLMException
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    LLMMessage,
    TextPart,
    ToolReturnPart,
)
from zhenxun.services.ai.core.models import LLMContext
from zhenxun.services.ai.core.options import GenerationConfig
from zhenxun.services.ai.core.stream_events import (
    STREAM_TEXT_KEY,
    LLMEndEvent,
    LLMStartEvent,
    TextDeltaEvent,
)
from zhenxun.services.ai.llm.api import invoke
from zhenxun.services.ai.llm.client import stream_sink
from zhenxun.services.ai.message_builder import MessageBuilder
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.run.models import AgentRunResult
from zhenxun.services.ai.run.session import session_manager
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.ai.utils.logger import log_agent as logger
from zhenxun.utils.pydantic_compat import model_construct, model_copy

from .models import AgentRunResources, AgentState


class AgentExecutor:
    """
    Agent 推理循环执行器。
    负责调用模型、执行工具、处理追加消息，并在达到最大循环次数时兜底总结。
    """

    def __init__(self):
        self.tool_executor = ToolExecutor()

    async def run(
        self, state: AgentState, resources: AgentRunResources
    ) -> AgentRunResult[Any]:
        """运行推理循环直到模型不再调用工具或达到最大循环次数

        参数:
            state: 运行状态
            resources: 运行资源

        返回:
            AgentRunResult[Any]: 运行结果
        """
        cycle = 0
        run = resources.run_context.run
        run.budget.setdefault("model_limit", resources.config.max_cycles)
        if not run.parent_run_id:
            run.budget["reserve_summary"] = resources.config.enable_fallback_summary
        while cycle < resources.config.max_cycles:
            state.current_cycle = cycle
            if token := resources.run_context.run.cancellation_token:
                token.raise_if_cancelled()
            remaining = run.budget["model_limit"] - run.budget.get("model_calls", 0)
            if (
                not run.parent_run_id
                and run.budget.get("model_calls")
                and remaining == 1
                and resources.config.enable_fallback_summary
            ):
                return await self._fallback(state, resources)
            self._build_request(state, resources)
            state.current_response = await self._invoke_and_record(
                state,
                resources,
                state.current_request_messages,
                list(state.tools)
                if state.tools
                and not (remaining == 1 and run.budget.get("reserve_summary"))
                else None,
            )
            if await self._collect_tool_calls(state, resources):
                session_id = resources.run_context.session_id or "default_session"
                session_manager.set_tools_running(session_id, True)
                try:
                    await self._execute_tools(state, resources)
                finally:
                    session_manager.set_tools_running(session_id, False)
                await self._add_follow_ups(state, resources)
            else:
                self._finish(state, resources, state.current_response.text)
            if state.is_finished:
                assert state.pending_result is not None
                return state.pending_result
            cycle = 0 if state.should_reset_cycle else cycle + 1
            state.should_reset_cycle = False
        return await self._fallback(state, resources)

    @staticmethod
    def _finish(state: AgentState, resources: AgentRunResources, output: Any) -> None:
        """标记循环结束并记录最终结果

        参数:
            state: 运行状态
            resources: 运行资源
            output: 最终输出
        """
        state.is_finished = True
        state.pending_result = model_construct(
            AgentRunResult,
            output=output,
            messages=list(resources.run_context.run.messages),
            usage=state.usage,
        )

    @staticmethod
    async def _add_follow_ups(state: AgentState, resources: AgentRunResources) -> None:
        """把工具执行期间并入的用户输入接在工具结果之后，并重新计算循环轮数

        参数:
            state: 运行状态
            resources: 运行资源
        """
        context = resources.run_context
        items = session_manager.drain_follow_ups(
            context.session_id or "default_session"
        )
        try:
            for prompt, receipt in items:
                if receipt.done():
                    continue
                if context.run.prepare_follow_up:
                    prompt = await context.run.prepare_follow_up(prompt)
                msgs = await MessageBuilder.normalize_to_llm_messages(prompt)
                if not msgs or receipt.done():
                    continue
                context.run.messages.append(msgs[-1])
                receipt.set_result(True)
                if injections := context.state.get("_injections"):
                    injections["sources"].append(
                        {
                            "source": "follow_up_with_background",
                            "role": "user",
                            "characters": len(msgs[-1].extract_text),
                            "count": 1,
                            "temporary": resources.memory_context is None,
                        }
                    )
                state.should_reset_cycle = True
        finally:
            for _, receipt in items:
                if not receipt.done():
                    receipt.set_result(False)

    async def _invoke_and_record(
        self,
        state: AgentState,
        resources: AgentRunResources,
        messages: list[LLMMessage],
        tools: list[Any] | None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        """经能力拦截链请求模型，统计用量并把回复追加到对话历史

        参数:
            state: 运行状态
            resources: 运行资源
            messages: 请求消息
            tools: 可用工具
            tool_choice: 工具选择策略

        返回:
            ChatResponse: 模型响应
        """
        run_context = resources.run_context
        flattened = await self._fit_request(messages, tools, state, resources)
        budget = run_context.run.budget
        call_limit = budget.get("model_limit", resources.config.max_cycles)
        if run_context.run.parent_run_id and budget.get("reserve_summary"):
            call_limit -= 1
        if budget.get("model_calls", 0) >= call_limit:
            from zhenxun.services.ai.core.exceptions import RunBudgetExceeded

            raise RunBudgetExceeded("本轮模型调用预算已用完，未继续请求。")
        budget["model_calls"] = budget.get("model_calls", 0) + 1
        model_name = run_context.run.current_model
        await run_context.run.emit(
            LLMStartEvent(model_name=model_name or "unknown", messages=flattened)
        )
        request = ChatRequest(
            messages=flattened,
            config=resources.generation_config or GenerationConfig(),
            tools=tools,
            tool_choice=tool_choice,
            extra={**run_context.state, "run_context": run_context},
        )

        async def handler(ctx: LLMContext[ChatRequest, ChatResponse]) -> ChatResponse:
            response = await invoke(
                "chat", ctx.request, model_name, ctx.cancellation_token
            )
            usage = parse_usage_info(response.usage_info)
            budget = run_context.run.budget
            budget["tokens"] = budget.get("tokens", 0) + usage.total_tokens
            raw = response.usage_info or {}
            if any(
                key in raw
                for key in (
                    "prompt_cache_hit_tokens",
                    "cachedContentTokenCount",
                    "cache_read_input_tokens",
                )
            ) or any(
                "cached_tokens" in (raw.get(key) or {})
                for key in ("input_tokens_details", "prompt_tokens_details")
            ):
                budget["reported_cache_read"] = (
                    budget.get("reported_cache_read", 0) + usage.prompt_cache_hit_tokens
                )
            return response

        async def emit_delta(kind: str, text: str) -> None:
            await run_context.run.emit(TextDeltaEvent(kind=kind, text=text))

        sink = stream_sink.set(
            emit_delta if run_context.state.get(STREAM_TEXT_KEY) else None
        )
        try:
            response = await resources.capability.wrap_model_request(
                run_context,
                LLMContext(
                    request=request,
                    cancellation_token=run_context.run.cancellation_token,
                ),
                handler,
            )
        finally:
            stream_sink.reset(sink)
        await run_context.run.emit(LLMEndEvent(response=response))

        assistant_message = AssistantMessage(content=response.content_parts)
        usage = parse_usage_info(response.usage_info)
        state.usage += usage
        if usage.completion_tokens > 0:
            assistant_message.token_cost = usage.completion_tokens
        run_context.run.messages.append(assistant_message)
        return response

    @staticmethod
    async def _fit_request(messages, tools, state, resources):
        """裁剪单次请求视图，保留当前问题及工具配对，不改写持久历史。"""
        from zhenxun.services.ai.context.memory.compression import (
            MessageDropper,
            ToolPrunerReducer,
        )
        from zhenxun.services.ai.llm.protocols import split_tools
        from zhenxun.utils.pydantic_compat import model_dump

        context = resources.run_context
        model = context.run.current_model or ""
        window = resources.model_profile.context_window or 128000
        threshold = get_llm_config().context_settings.llm_summary.trigger_threshold
        if resources.memory_context:
            local = resources.memory_context.memory_config.compression.threshold
            if local is not None:
                threshold = local
        limit = (
            window
            if threshold <= 0
            else int(window * threshold)
            if threshold <= 1
            else int(threshold)
        )
        reserve = getattr(resources.generation_config, "max_tokens", None) or min(
            4096, window // 4
        )
        limit = max(1, min(limit, window - reserve))
        definitions, server = await split_tools(tools)
        schemas = [model_dump(t) for t in definitions] + [
            {"type": t.type_id, "name": t.name, **t.metadata} for t in server
        ]
        overhead = sum(
            token_counter._count_text(
                json.dumps(
                    t,
                    ensure_ascii=False,
                    default=str,
                )
            )
            for t in schemas
        )

        def count(values):
            return token_counter.count_context(values, model, overhead)

        if count(messages) <= limit:
            return list(messages)
        head_size = sum(bool(text.strip()) for text in state.static_system_prompt)
        old_size = next(
            (i for i, msg in enumerate(context.run.messages) if msg is state._input),
            max(0, state._origin_msg_len - 1),
        )
        head = list(messages[:head_size])
        old = list(messages[head_size : head_size + old_size])
        current = list(messages[head_size + old_size :])
        sentinel = LLMMessage.user("")
        fixed = count([*head, *current])
        old, _, _ = await MessageDropper(limit).reduce(
            [*old, sentinel], count(messages), model, fixed
        )
        old = [m for m in old if m is not sentinel]
        fitted = [*head, *old, *current]
        if count(fitted) > limit:
            pruner = ToolPrunerReducer(keep_recent_turns=2, trigger_tokens=0)
            current, _, _ = await pruner.reduce(
                current, count(fitted), model, count([*head, *old])
            )
            fitted = [*head, *old, *current]
        if count(fitted) > limit:
            notice = "当前输入或工具结果超过模型上下文容量，请缩短输入或分步提问。"
            context.run.user_notice = notice
            raise LLMException(notice, details={"code": "chat_context_overflow"})
        return fitted

    @staticmethod
    def _build_request(state: AgentState, resources: AgentRunResources) -> None:
        """组装本轮请求消息：系统消息、带提示块的对话历史与接近循环上限时的
        收尾提醒

        参数:
            state: 运行状态
            resources: 运行资源
        """
        state.current_request_messages = AgentExecutor._request_messages(
            state, resources.run_context
        )
        if hint := AgentExecutor._budget_hint(
            state.current_cycle, resources.config.max_cycles
        ):
            state.current_request_messages.append(hint)

    @staticmethod
    def _budget_hint(cycle: int, max_cycles: int) -> LLMMessage | None:
        """工具循环轮数用到 80% 起提醒模型收尾，只加在本轮请求中，不写入历史

        参数:
            cycle: 当前轮次 (从 0 开始)
            max_cycles: 最大循环次数

        返回:
            LLMMessage | None: 提醒消息，未到提醒阈值时为 None
        """
        if cycle == 0 or cycle < max_cycles * 0.8:
            return None
        remaining = max_cycles - cycle
        if remaining <= 1:
            text = (
                f"[系统提示] 工具调用轮数已用 {cycle}/{max_cycles}，这是最后一轮。"
                "请不要再调用工具，根据已有结果直接给出最终答复；未完成的部分如实说明。"
            )
        else:
            text = (
                f"[系统提示] 工具调用轮数已用 {cycle}/{max_cycles}，"
                f"还剩 {remaining} 轮。请尽快收尾，只做必要的调用。"
            )
        return LLMMessage.user(text)

    @staticmethod
    def _request_messages(
        state: AgentState, run_context: RunContext
    ) -> list[LLMMessage]:
        """组装请求消息：静态系统提示词在前，对话历史在后；本次运行的提示与
        运行中追加的提示合成 `<system_reminder>` 文本块，贴在本轮用户输入末尾
        （找不到时贴在最后一条用户消息上），只出现在请求中，不写入历史

        参数:
            state: Agent 运行状态
            run_context: 运行上下文

        返回:
            list[LLMMessage]: 请求消息
        """
        messages: list[LLMMessage] = [
            LLMMessage.system(text)
            for text in state.static_system_prompt
            if text and text.strip()
        ]
        history = list(run_context.run.messages)
        if injections := run_context.state.get("_injections"):
            dynamic = list(run_context.run.dynamic_prompts.values())
            injections["dynamic"] = {
                "source": "runtime_reminders",
                "role": "user",
                "characters": sum(len(text) for text in dynamic),
                "count": len(dynamic),
                "temporary": True,
            }
        texts = [
            text
            for text in [*state.reminders, *run_context.run.dynamic_prompts.values()]
            if text and text.strip()
        ]
        if texts:
            block = "<system_reminder>\n" + "\n\n".join(texts) + "\n</system_reminder>"
            index = next((i for i, m in enumerate(history) if m is state._input), None)
            if index is None:
                index = next(
                    (
                        i
                        for i in range(len(history) - 1, -1, -1)
                        if history[i].role == "user"
                    ),
                    None,
                )
            if index is None:
                history.append(LLMMessage.user(block))
            else:
                target = model_copy(history[index])
                target.content = [*target.content, TextPart(text=block)]
                target.token_cost = None
                history[index] = target
        messages.extend(history)
        return messages

    async def _collect_tool_calls(
        self, state: AgentState, resources: AgentRunResources
    ) -> bool:
        """取出需要在本地执行的工具调用，服务端已执行的调用只发送事件

        参数:
            state: 运行状态
            resources: 运行资源

        返回:
            bool: 是否有需要本地执行的工具调用
        """
        response = state.current_response
        assert response is not None
        returns = {
            p.tool_call_id: p
            for p in response.content_parts
            if isinstance(p, ToolReturnPart)
        }
        event_bus = resources.run_context.run.event_bus
        state.current_tool_calls = []
        for call in response.tool_calls or []:
            if call.id not in returns:
                state.current_tool_calls.append(call)
                continue
            logger.debug(f"☁️ 检测到云端工具调用: {call.tool_name}，已跳过本地执行。")
            async with self.tool_executor._tool_stream_scope(
                event_bus,
                call.tool_name,
                call.args if isinstance(call.args, dict) else {},
                resources.run_context,
                call.id,
            ) as box:
                box["result"] = ToolResult(output=returns[call.id].output)
        return bool(state.current_tool_calls)

    async def _execute_tools(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """校验并执行工具调用，把结果追加到对话历史

        参数:
            state: 运行状态
            resources: 运行资源
        """
        run_context = resources.run_context
        validated = await asyncio.gather(
            *[
                self.tool_executor.validate_tool_call(call, state.tools, run_context)
                for call in state.current_tool_calls
            ]
        )
        results = await self.tool_executor.execute_tool_calls(
            list(validated),
            run_context,
            event_bus=run_context.run.event_bus,
            parallel=resources.config.enable_parallel_calls,
        )
        max_chars = get_llm_config().agent_settings.tool_output_max_chars
        for call, res_or_exc in zip(state.current_tool_calls, results):
            tool_res = None if isinstance(res_or_exc, BaseException) else res_or_exc[1]
            msg, usage = ToolExecutor.assemble_tool_message(
                call, res_or_exc, tool_res, max_chars
            )
            if usage:
                state.usage += usage
            run_context.run.messages.append(msg)

    async def _fallback(
        self, state: AgentState, resources: AgentRunResources
    ) -> AgentRunResult[Any]:
        """达到最大循环次数时让模型总结已有进展，关闭兜底总结时报错

        参数:
            state: 运行状态
            resources: 运行资源

        返回:
            AgentRunResult[Any]: 兜底总结结果
        """
        run_context = resources.run_context
        max_cycles = resources.config.max_cycles
        if not resources.config.enable_fallback_summary:
            from zhenxun.services.ai.core.exceptions import RunBudgetExceeded

            raise RunBudgetExceeded(
                f"超过最大工具调用循环次数 ({max_cycles})。",
            )
        logger.warning(f"达到最大循环次数 ({max_cycles})，触发兜底总结机制。")
        hint = LLMMessage.user(
            "[收尾提示] 工具步骤已用完。请根据已有信息回答，"
            "如实说明未完成的部分，不再调用工具。"
        )
        response = await self._invoke_and_record(
            state,
            resources,
            [*self._request_messages(state, run_context), hint],
            [],
            tool_choice="none",
        )
        self._finish(state, resources, response.text)
        assert state.pending_result is not None
        state.pending_result.outcome = "exhausted"
        return state.pending_result
