from typing import Any

from nonebot import on_message
from nonebot.adapters import Bot, Event
from nonebot.plugin import PluginMetadata
from nonebot.typing import T_State
from nonebot_plugin_alconna import UniMsg, get_message_id
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_uninfo import Uninfo
from tortoise.transactions import in_transaction

from zhenxun.configs.config import Config
from zhenxun.configs.utils import PluginExtraData, RegisterConfig
from zhenxun.models.chat_history import ChatHistory
from zhenxun.services.db_context import with_db_timeout
from zhenxun.services.log import logger
from zhenxun.services.low_priority_writer import (
    LowPriorityWriterConfig,
    append_low_priority_record,
    register_low_priority_writer,
)
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils
from zhenxun.utils.platform import PlatformUtils
from zhenxun.utils.time_utils import TimeUtils
from zhenxun.utils.utils import get_entity_ids

__plugin_meta__ = PluginMetadata(
    name="消息存储",
    description="消息存储，被动存储群消息与 Bot 发出的群消息",
    usage="",
    extra=PluginExtraData(
        author="HibiKier",
        version="0.1",
        plugin_type=PluginType.HIDDEN,
        configs=[
            RegisterConfig(
                module="chat_history",
                key="FLAG",
                value=True,
                help="是否开启消息自从存储",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                module="chat_history",
                key="SEGMENT_DAYS",
                value=30,
                help="结构化消息段保留天数，过期后只保留文本，0 为不清理",
                default_value=30,
                type=int,
            ),
            RegisterConfig(
                module="chat_history",
                key="RETENTION_DAYS",
                value=0,
                help="聊天记录保留天数，0 为不清理（发言排行依赖全部记录）",
                default_value=0,
                type=int,
            ),
        ],
    ).to_dict(),
)


def rule(message: UniMsg) -> bool:
    return bool(Config.get_config("chat_history", "FLAG") and message)


chat_history = on_message(rule=rule, priority=1, block=False)

_WRITER_NAME = "chat_history"
_FLUSH_BATCH_SIZE = 200
_FLUSH_MAX_PER_TICK = 1000
_FLUSH_DB_TIMEOUT = 5.0
_STRING_FIELDS = (
    "user_id",
    "group_id",
    "text",
    "plain_text",
    "bot_id",
    "platform",
    "message_id",
)


def _strip_nul(value: str) -> str:
    return value.replace("\x00", "")


def _sanitize_chat_history(record: ChatHistory) -> int:
    changed = 0
    for field in _STRING_FIELDS:
        value = getattr(record, field, None)
        if isinstance(value, str) and "\x00" in value:
            setattr(record, field, _strip_nul(value))
            changed += 1
    return changed


async def _write_chat_history_batch(batch: list[ChatHistory], reason: str) -> None:
    """在同一超时范围内写入并结算聊天记录批次事务。"""
    sanitized = sum(_sanitize_chat_history(record) for record in batch)
    if sanitized:
        logger.warning(
            f"聊天历史写库前清理了 {sanitized} 个含 NUL 的字符串字段",
            "chat_history",
        )

    async def write():
        async with in_transaction() as connection:
            await ChatHistory.bulk_create(batch, _FLUSH_BATCH_SIZE, using_db=connection)

    await with_db_timeout(
        write(),
        timeout=_FLUSH_DB_TIMEOUT,
        operation=f"ChatHistory.bulk_create[{len(batch)}]",
        source=f"chat_history:{reason}",
    )


register_low_priority_writer(
    LowPriorityWriterConfig(
        name=_WRITER_NAME,
        write_batch=_write_chat_history_batch,
        batch_size=_FLUSH_BATCH_SIZE,
        trigger_size=_FLUSH_BATCH_SIZE,
        max_retain=5000,
        flush_interval_seconds=60.0,
        max_items_per_cycle=_FLUSH_MAX_PER_TICK,
        backoff_base_seconds=30.0,
        backoff_max_seconds=600.0,
        log_command="chat_history",
    )
)


async def _append(record: ChatHistory) -> None:
    sanitized = _sanitize_chat_history(record)
    if sanitized:
        logger.warning(
            f"聊天历史入队前清理了 {sanitized} 个含 NUL 的字符串字段",
            "chat_history",
        )
    await append_low_priority_record(_WRITER_NAME, record)


@chat_history.handle()
async def _(bot: Bot, event: Event, message: UniMsg, session: Uninfo, state: T_State):
    entity = get_entity_ids(session)
    try:
        try:
            message_id = get_message_id(event, bot)
        except Exception:
            message_id = None
        await _append(
            ChatHistory(
                create_time=state.get("_zx_received_at") or TimeUtils.now(),
                user_id=entity.user_id,
                group_id=entity.group_id,
                text=str(message),
                plain_text=message.extract_plain_text(),
                bot_id=PlatformUtils.get_storage_bot_id(session),
                platform=session.platform,
                message_id=message_id,
                segments=ChatHistory.dump_segments(message),
            )
        )
    except Exception as e:
        logger.warning("存储聊天记录失败", "chat_history", e=e)


@Bot.on_called_api
async def _record_bot_send(
    bot: Bot, exception: Exception | None, api: str, data: dict[str, Any], result: Any
):
    if (
        exception is not None
        or api not in ("send_msg", "send_group_msg")
        or not data.get("group_id")
        or not Config.get_config("chat_history", "FLAG")
    ):
        return
    try:
        message = MessageUtils.from_onebot(data.get("message"))
        message_id = result.get("message_id") if isinstance(result, dict) else None
        await _append(
            ChatHistory(
                user_id=str(bot.self_id),
                group_id=str(data["group_id"]),
                text=str(message),
                plain_text=message.extract_plain_text(),
                bot_id=PlatformUtils.get_storage_bot_id(bot),
                platform=PlatformUtils.get_platform(bot),
                message_id=str(message_id) if message_id is not None else None,
                segments=ChatHistory.dump_segments(message),
                is_bot=True,
            )
        )
    except Exception as e:
        logger.warning("存储 Bot 发言失败", "chat_history", e=e)


@scheduler.scheduled_job("cron", hour=4, minute=45)
async def _cleanup_chat_history():
    from zhenxun.services.scheduler.lifecycle import core_task_run

    async with core_task_run("chat_history_cleanup") as run:
        result = await ChatHistory.cleanup(
            int(Config.get_config("chat_history", "SEGMENT_DAYS") or 0),
            int(Config.get_config("chat_history", "RETENTION_DAYS") or 0),
            progress=run,
        )
        cleared, deleted = result
        run.update(
            processed=cleared + deleted,
            cleared=cleared,
            deleted=deleted,
            batches=result.batches,
            scanned=result.scanned,
        )
        if result.pause_reason:
            run.update(
                status="partial" if run["processed"] else "skipped",
                reason=result.pause_reason,
            )
