import asyncio
import random

from nonebot import get_driver
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    GroupMessageEvent,
    Message,
    PrivateMessageEvent,
)
from nonebot.compat import model_dump, type_validate_python
from nonebot_plugin_alconna import Alconna, Args, on_alconna

from zhenxun.services.log import logger

tasks: set["asyncio.Task"] = set()


@get_driver().on_shutdown
async def cancel_tasks():
    for task in tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(
        *(asyncio.wait_for(task, timeout=10) for task in tasks),
        return_exceptions=True,
    )


def push_event(bot: Bot, event: PrivateMessageEvent | GroupMessageEvent):
    event.message = Message("签到")
    user_id = random.randint(1, 99999999999) + random.randint(1, 99999999999)
    event.user_id = user_id
    event.sender.user_id = user_id
    task = asyncio.create_task(bot.handle_event(event))
    task.add_done_callback(tasks.discard)
    tasks.add(task)
    logger.info(f"发送消息 --> {event.user_id} {event.message}")
    return event


_matcher = on_alconna(
    Alconna("test", Args["n", int]), priority=5, block=True, temp=True
)
_matcher = on_alconna(
    Alconna("test", Args["n", int]), priority=5, block=True, temp=True
)


@_matcher.handle()
async def handle_event(event: Event, bot: Bot, n: int):
    for _ in range(n):
        data = model_dump(event)
        data["message"] = "签到"
        if data.get("message_type") == "private":
            data["post_type"] = "message"
            push_event(bot, type_validate_python(PrivateMessageEvent, data))
        elif data.get("message_type") == "group":
            data["post_type"] = "message"
            push_event(bot, type_validate_python(GroupMessageEvent, data))
        await asyncio.sleep(0.1)
        logger.info(f"发送消息次数 --> {_ + 1}")
