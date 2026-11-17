import asyncio
import time
from typing import Any

from nonebot import on_message
from nonebot.adapters import Bot, Event
from nonebot.plugin import PluginMetadata
from nonebot.typing import T_State
from nonebot_plugin_alconna import Alconna, Reply, UniMessage, UniMsg, on_alconna
from nonebot_plugin_alconna.uniseg import SerializeFailed
from nonebot_plugin_apscheduler import scheduler

from zhenxun.configs.config import Config
from zhenxun.configs.utils import Command, PluginExtraData
from zhenxun.models.ai_agent_call import AIAgentCall
from zhenxun.models.ai_agent_run import AIAgentRun
from zhenxun.models.ai_knowledge_base import AIKnowledgeBase
from zhenxun.models.ai_token_usage import AITokenUsage
from zhenxun.services.ai.context.memory.storage import HISTORY_RETAIN
from zhenxun.services.ai.run.context import NoneBotDeps
from zhenxun.services.ai.run.dsh import DshRuntime
from zhenxun.services.ai.utils.runtime import ContextUtils
from zhenxun.services.log import logger
from zhenxun.utils.manager.priority_manager import PriorityLifecycle
from zhenxun.utils.message import MessageUtils

from ._buffer import AtWaiter, ChatBuffer, ChatLine, sender_name, summarize
from ._data_source import MODULE, AIChatManager
from ._memory import Memory
from ._proactive import PROACTIVE_KEY, Proactive
from ._reply import quote_reply
from ._steal import Steal
from ._tasks import register_task_job
from .config import CONFIGS

MESSAGE_KEY = "ai_chat_message"
"""规则补上引用后的消息在 state 中的键"""

__plugin_meta__ = PluginMetadata(
    name="AI 聊天",
    description="和 Bot 用自然语言聊天",
    usage="""
    @Bot、回复 Bot 的消息、私聊、以 / 或唤醒词开头即可聊天
    群聊中全群共用当前选中的对话，Bot 能看到上次发言之后群里的新消息
    只 @ Bot 不说话时，Bot 会等你 15 秒内的下一条消息一起回复
    群聊中可按概率或由监听模型判断后主动插话，也能自动收集群里的表情包
    开启长期记忆后会把聊天整理成记忆，之后聊到相关内容时想起来
    指令:
        停止对话 / /停止对话: 中止自己发起的对话
        重置对话 / /重置对话: 清空当前群或私聊的对话记忆（群聊限群管理员）
        /ai help: 查看会话、模型、人设、记忆、计划和 DSH 控制命令
    """.strip(),
    extra=PluginExtraData(
        author="Copaan",
        version="0.1",
        menu_type="AI",
        commands=[Command(command="停止对话"), Command(command="重置对话")],
        configs=CONFIGS,
    ).to_dict(),
)


async def _with_reply(bot: Bot, event: Event, message: UniMessage) -> UniMessage:
    """适配器把引用从消息中移到 event.reply 时（如 OneBot V11），把引用补回消息开头

    参数:
        bot: Bot
        event: 事件
        message: 消息

    返回:
        UniMessage: 带引用的消息副本，没有引用时为原消息
    """
    if getattr(event, "reply", None) is None or message.has(Reply):
        return message
    try:
        return await UniMessage(message).attach_reply(event, bot)
    except SerializeFailed:
        return message


async def _rule(bot: Bot, event: Event, message: UniMsg, state: T_State) -> bool:
    if not message:
        return False
    from ._commands import command_text

    if command_text(message.extract_plain_text()) is not None:
        return False
    from ._sessions import Conversations

    if (await Conversations.preferences(AIChatManager.session_id(bot, event))).get(
        "disabled"
    ):
        return False
    message = state[MESSAGE_KEY] = await _with_reply(bot, event, message)
    deps = NoneBotDeps(bot=bot, event=event)
    group_id = ContextUtils.extract_group_id(deps)
    line = None
    if group_id:
        user_id = ContextUtils.extract_user_id(deps) or ""
        if await AIChatManager.blocked(bot, group_id, user_id):
            return False
        message_id = getattr(event, "message_id", None)
        line = ChatLine(
            str(message_id) if message_id is not None else None,
            user_id,
            sender_name(event, user_id),
            summarize(message),
            time.time(),
        )
        ChatBuffer.record(bot.self_id, group_id, line)
        state["ai_chat_anchor"] = line.sequence
        Proactive.wake(bot.self_id, group_id)
        await Steal.observe(bot, group_id, user_id, message)
        if AtWaiter.feed(bot.self_id, group_id, user_id, message, line.message_id):
            return False
    if await AIChatManager.should_reply(event, message, group_id, bot):
        if group_id:
            Proactive.answered(bot.self_id, group_id)
        return True
    if not group_id or not line or not line.text:
        return False
    Proactive.observe(bot.self_id, group_id, line)
    if (hint := await Proactive.decide(bot, group_id)) is not None:
        state[PROACTIVE_KEY] = hint
        return True
    return False


_chat_matcher = on_message(rule=_rule, priority=999, block=False)


async def _control_rule(message: UniMsg, state: T_State) -> bool:
    from ._commands import command_text

    value = command_text(message.extract_plain_text())
    if value is None:
        return False
    state["ai_control_command"] = value
    return True


_control_matcher = on_message(rule=_control_rule, priority=98, block=True)


@_control_matcher.handle()
async def _control(bot: Bot, event: Event, state: T_State):
    from ._commands import execute

    try:
        result = await execute(bot, event, state["ai_control_command"])
    except ValueError as error:
        result = str(error)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning(f"AI 控制命令未完成: {type(error).__name__}", "ai_chat")
        result = "操作未完成；没有发送给模型。请刷新状态后重试，详细原因见管理日志。"
    await MessageUtils.build_message(UniMessage.text(result)).finish(
        reply_to=quote_reply(
            ContextUtils.extract_group_id(NoneBotDeps(bot=bot, event=event))
        )
    )


_stop_matcher = on_alconna(
    Alconna("停止对话"), aliases={"/停止对话"}, priority=5, block=True
)
_reset_matcher = on_alconna(
    Alconna("重置对话"), aliases={"/重置对话"}, priority=5, block=True
)


@Bot.on_called_api
async def _record_bot_send(
    bot: Bot, exception: Exception | None, api: str, data: dict[str, Any], result: Any
):
    if exception is not None:
        return
    if api == "post_group_messages" and (openid := data.get("group_openid")):
        group_id = f"qq_api:{bot.self_id}:group:{openid}"
        data = {"group_id": group_id, "message": data.get("content") or ""}
    elif api not in ("send_msg", "send_group_msg"):
        return
    group_id = data.get("group_id")
    if group_id and await AIChatManager.blocked(bot, str(group_id), None):
        return
    ChatBuffer.record_bot_send(bot.self_id, data, result)


@_chat_matcher.handle()
async def _(bot: Bot, event: Event, message: UniMsg, state: T_State):
    await AIChatManager.chat(
        bot,
        event,
        state.get(MESSAGE_KEY, message),
        proactive=state.get(PROACTIVE_KEY),
        input_anchor=state.get("ai_chat_anchor"),
    )


@_stop_matcher.handle()
async def _(bot: Bot, event: Event):
    await _control(bot, event, {"ai_control_command": "stop"})


@_reset_matcher.handle()
async def _(bot: Bot, event: Event):
    await _control(bot, event, {"ai_control_command": "reset"})


register_task_job()
if Memory.cursors not in HISTORY_RETAIN:
    HISTORY_RETAIN.append(Memory.cursors)


@PriorityLifecycle.on_startup(
    priority=90,
    stage="warmup",
    timeout=60,
    component_id="warmup:ai_chat_persona_upgrade",
    depends_on=("management:database",),
    failure_policy="degrade",
)
async def _():
    Proactive.upgrade_config()
    from zhenxun.services.ai.run.session import session_manager

    await session_manager.start()
    await AIChatManager.upgrade_personas()
    if Config.get_config(MODULE, "MEMORY_MODE", "off") != "off" or (
        await AIKnowledgeBase.exists()
    ):
        import jieba

        await asyncio.to_thread(jieba.initialize)


@PriorityLifecycle.on_shutdown(
    priority=60, timeout=15, component_id="runtime:ai_chat_dsh"
)
async def _():
    from zhenxun.services.ai.run.session import session_manager

    await session_manager.shutdown()
    await Memory.stop(shutdown=True)
    await DshRuntime.close()


@scheduler.scheduled_job("cron", hour=4, minute=50, id="ai_chat_token_cleanup")
async def _():
    from zhenxun.services.scheduler.lifecycle import core_task_run

    async with core_task_run("ai_usage_cleanup") as run:
        run["failed"] = 0
        for name, cleanup in (
            ("token_usage", AITokenUsage.cleanup),
            ("agent_calls", AIAgentCall.cleanup),
            ("agent_runs", AIAgentRun.cleanup),
            ("memory_archived", Memory.decay),
        ):
            try:
                count = await cleanup()
                run[name] = count
                run["processed"] += count
            except Exception as error:
                run["failed"] += 1
                logger.error(f"AI 定时清理失败: {name}", MODULE, e=error)


@scheduler.scheduled_job(
    "interval", seconds=600, id="ai_chat_memory_retry", max_instances=1
)
async def _memory_tick():
    """补处理空闲尾段和需要重试的记忆区段。"""
    await Memory.tick()
