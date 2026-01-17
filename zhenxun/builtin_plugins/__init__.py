from datetime import datetime

import nonebot
from nonebot.adapters import Bot
from nonebot.drivers import Driver
from tortoise.exceptions import IntegrityError

from zhenxun.models.bot_connect_log import BotConnectLog
from zhenxun.models.bot_console import BotConsole
from zhenxun.services.log import logger
from zhenxun.utils.platform import PlatformUtils

driver: Driver = nonebot.get_driver()


@driver.on_bot_connect
async def _(bot: Bot):
    logger.debug(f"Bot: {bot.self_id} 建立连接...")
    await BotConnectLog.create(
        bot_id=bot.self_id, platform=bot.adapter, connect_time=datetime.now(), type=1
    )
    if not await BotConsole.exists(bot_id=bot.self_id):
        try:
            await BotConsole.create(
                bot_id=bot.self_id, platform=PlatformUtils.get_platform(bot)
            )
        except IntegrityError as e:
            logger.warning(f"记录bot: {bot.self_id} 数据已存在...", e=e)


@driver.on_bot_disconnect
async def _(bot: Bot):
    logger.debug(f"Bot: {bot.self_id} 断开连接...")
    try:
        await BotConnectLog.create(
            bot_id=bot.self_id,
            platform=bot.adapter,
            connect_time=datetime.now(),
            type=0,
        )
    except Exception as e:
        logger.warning(f"记录bot: {bot.self_id} 断开连接失败", e=e)

