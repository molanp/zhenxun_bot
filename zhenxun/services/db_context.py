import asyncio

import nonebot
from nonebot.utils import is_coroutine_callable
from tortoise import Tortoise
from tortoise.connection import connections
from tortoise.models import Model as TortoiseModel

from zhenxun.configs.config import BotConfig
from zhenxun.utils.exception import HookPriorityException
from zhenxun.utils.manager.priority_manager import PriorityLifecycle

from .log import logger

SCRIPT_METHOD = []
MODELS: list[str] = []


driver = nonebot.get_driver()


DEFAULT_DB_TIMEOUT = 10  # 默认超时时间（秒）


class Model(TortoiseModel):
    """
    自动添加模块 + 数据库操作超时
    """

    db_timeout: int = DEFAULT_DB_TIMEOUT

    def __init_subclass__(cls, **kwargs):
        if cls.__module__ not in MODELS:
            MODELS.append(cls.__module__)

        if func := getattr(cls, "_run_script", None):
            SCRIPT_METHOD.append((cls.__module__, func))
        super().__init_subclass__(**kwargs)

    async def save(self, *args, **kwargs):
        return await asyncio.wait_for(
            super().save(*args, **kwargs), timeout=self.db_timeout
        )

    async def delete(self, *args, **kwargs):
        return await asyncio.wait_for(
            super().delete(*args, **kwargs), timeout=self.db_timeout
        )

    @classmethod
    async def create(cls, *args, **kwargs):
        return await asyncio.wait_for(
            super().create(*args, **kwargs), timeout=cls.db_timeout
        )

    @classmethod
    async def get_or_create(cls, *args, **kwargs):
        return await asyncio.wait_for(
            super().get_or_create(*args, **kwargs), timeout=cls.db_timeout
        )

    @classmethod
    async def get_or_none(cls, *args, **kwargs):
        return await asyncio.wait_for(
            super().get_or_none(*args, **kwargs), timeout=cls.db_timeout
        )

    @classmethod
    async def update_or_create(cls, *args, **kwargs):
        return await asyncio.wait_for(
            super().update_or_create(*args, **kwargs), timeout=cls.db_timeout
        )


class DbUrlIsNode(HookPriorityException):
    """
    数据库链接地址为空
    """

    pass


class DbConnectError(Exception):
    """
    数据库连接错误
    """

    pass


@PriorityLifecycle.on_startup(priority=1)
async def init():
    if not BotConfig.db_url:
        # raise DbUrlIsNode("数据库配置为空，请在.env.dev中配置DB_URL...")
        error = f"""
**********************************************************************
🌟 **************************** 配置为空 ************************* 🌟
🚀 请打开 WebUi 进行基础配置 🚀
🌐 配置地址：http://{driver.config.host}:{driver.config.port}/#/configure 🌐
***********************************************************************
***********************************************************************
        """
        raise DbUrlIsNode("\n" + error.strip())
    try:
        await Tortoise.init(
            db_url=BotConfig.db_url,
            modules={"models": MODELS},
            timezone="Asia/Shanghai",
        )
        if SCRIPT_METHOD:
            db = Tortoise.get_connection("default")
            logger.debug(
                "即将运行SCRIPT_METHOD方法, 合计 "
                f"<u><y>{len(SCRIPT_METHOD)}</y></u> 个..."
            )
            sql_list = []
            for module, func in SCRIPT_METHOD:
                try:
                    sql = await func() if is_coroutine_callable(func) else func()
                    if sql:
                        sql_list += sql
                except Exception as e:
                    logger.debug(f"{module} 执行SCRIPT_METHOD方法出错...", e=e)
            for sql in sql_list:
                logger.debug(f"执行SQL: {sql}")
                try:
                    await db.execute_query_dict(sql)
                    # await TestSQL.raw(sql)
                except Exception as e:
                    logger.debug(f"执行SQL: {sql} 错误...", e=e)
            if sql_list:
                logger.debug("SCRIPT_METHOD方法执行完毕!")
        await Tortoise.generate_schemas()
        logger.info("Database loaded successfully!")
    except Exception as e:
        raise DbConnectError(f"数据库连接错误... e:{e}") from e


async def disconnect():
    await connections.close_all()