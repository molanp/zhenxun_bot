import asyncio
import time
from typing import Any

import nonebot
from nonebot.adapters import Bot

from zhenxun.services.log import logger

_SEND_APIS = {"send_msg", "send_like"}
_QUEUE: asyncio.Queue[tuple[Bot, str, dict[str, Any], asyncio.Future]] = asyncio.Queue()
_WORKERS = 3
_MIN_INTERVAL = 0.05
_SEND_LOCK = asyncio.Lock()
_LAST_SEND_TS = 0.0
_API_SEMAPHORE = asyncio.Semaphore(3)
_ORIG_CALL_API = Bot.call_api
_PATCHED = False
_WORKER_TASKS: list[asyncio.Task] = []


async def _rate_limit():
    global _LAST_SEND_TS
    async with _SEND_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _LAST_SEND_TS)
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_SEND_TS = time.monotonic()


async def _worker(worker_id: int):
    while True:
        bot, api, data, future = await _QUEUE.get()
        try:
            await _rate_limit()
            async with _API_SEMAPHORE:
                result = await _ORIG_CALL_API(bot, api, **data)
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            logger.warning(
                f"send queue failed: {api}",
                "SendQueue",
                target=getattr(bot, "self_id", None),
                e=exc,
            )
        finally:
            _QUEUE.task_done()


async def _queued_call_api(self: Bot, api: str, **data: Any):
    if api not in _SEND_APIS:
        return await _ORIG_CALL_API(self, api, **data)
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await _QUEUE.put((self, api, data, future))
    return await future


def patch_send_queue() -> None:
    global _PATCHED
    if _PATCHED:
        return
    Bot.call_api = _queued_call_api  # type: ignore[assignment]
    _PATCHED = True


driver = nonebot.get_driver()


@driver.on_startup
async def _start_send_queue():
    patch_send_queue()
    for idx in range(_WORKERS):
        _WORKER_TASKS.append(asyncio.create_task(_worker(idx)))


@driver.on_shutdown
async def _stop_send_queue():
    tasks = _WORKER_TASKS.copy()
    _WORKER_TASKS.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
