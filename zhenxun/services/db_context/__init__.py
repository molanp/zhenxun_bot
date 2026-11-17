import asyncio
from dataclasses import replace
import inspect
import os
from pathlib import Path

from tortoise import Tortoise
from tortoise.exceptions import ConfigurationError

from zhenxun.configs.config import BotConfig
from zhenxun.services.lifecycle.deadline import (
    FINAL_SNAPSHOT_SECONDS,
    WORKER_SHUTDOWN_SECONDS,
)
from zhenxun.services.log import logger
from zhenxun.utils.manager.priority_manager import PriorityLifecycle

from .base_model import Model
from .config import (
    DB_TIMEOUT_SECONDS,
    MYSQL_CONFIG,
    POSTGRESQL_CONFIG,
    SQLITE_CONFIG,
    db_model,
    prompt,
)
from .exceptions import DbConnectError, DbUrlIsNode
from .schema_guard import (
    collect_script_migrations,
    finish_script_migrations,
    repair_safe_schema_drift,
    run_script_migrations,
)
from .utils import with_db_timeout

__all__ = [
    "DB_TIMEOUT_SECONDS",
    "DbConnectError",
    "DbUrlIsNode",
    "Model",
    "database_ready",
    "disconnect",
    "init",
    "with_db_timeout",
]

_database_ready = False
_orm_context = None


def database_ready() -> bool:
    """数据库表结构准备完成后才允许写入

    返回:
        bool: 数据库是否可用
    """
    return _database_ready


def get_config(endpoint=None) -> dict:
    """获取数据库配置"""
    if not BotConfig.db_url:
        raise DbUrlIsNode("数据库Url连接字符串为空，请检查配置文件（.env.dev）")
    config = {
        "connections": {"default": BotConfig.db_url},
        "apps": {
            "models": {
                "models": db_model.models,
                "default_connection": "default",
            }
        },
        "timezone": BotConfig.timezone,
    }

    from zhenxun.configs.database import DatabaseConnection

    endpoint = endpoint or DatabaseConnection.parse(BotConfig.db_url)
    defaults = {
        "postgres": POSTGRESQL_CONFIG,
        "mysql": MYSQL_CONFIG,
        "sqlite": SQLITE_CONFIG,
    }[endpoint.engine]
    if endpoint.engine == "sqlite" and endpoint.database != ":memory:":
        Path(endpoint.database).parent.mkdir(parents=True, exist_ok=True)
    config["connections"]["default"] = endpoint.orm_config(defaults)
    return config


@PriorityLifecycle.on_startup(
    priority=1,
    stage="management",
    timeout=60,
    component_id="management:database",
)
async def init():
    """连接数据库，执行模型迁移脚本并生成表结构"""
    global _database_ready, _orm_context

    _database_ready = False
    if not BotConfig.db_url:
        raise DbUrlIsNode("\n" + prompt.strip())
    try:
        from zhenxun.configs.database import (
            DatabaseConnection,
            bind_database_connection,
            inspect_database_tls,
        )
        import zhenxun.models.ai_agent_call
        import zhenxun.models.ai_agent_run
        import zhenxun.models.ai_chat_conversation
        import zhenxun.models.ai_chat_group
        import zhenxun.models.ai_chat_message
        import zhenxun.models.ai_chat_scene
        import zhenxun.models.ai_knowledge_base
        import zhenxun.models.ai_knowledge_chunk
        import zhenxun.models.ai_knowledge_doc
        import zhenxun.models.ai_memory
        import zhenxun.models.ai_memory_cursor
        import zhenxun.models.ai_model_stat
        import zhenxun.models.ai_persona
        import zhenxun.models.ai_sticker
        import zhenxun.models.ai_sticker_block
        import zhenxun.models.ai_token_usage
        from zhenxun.models.chat_history import ensure_chat_history_nullable_columns
        from zhenxun.models.group_plugin_setting import (
            ensure_group_plugin_scope_constraint,
        )
        import zhenxun.models.migration_import  # noqa: F401

        bind_database_connection(None)
        endpoint = replace(
            DatabaseConnection.parse(BotConfig.db_url),
            source="environment"
            if any(key.upper() == "DB_URL" for key in os.environ)
            else "configuration_file",
        )
        if _orm_context is not None:
            await _orm_context.close_connections()
            _orm_context = None
        context_options = (
            {"_enable_global_fallback": True}
            if "_enable_global_fallback" in inspect.signature(Tortoise.init).parameters
            else {}
        )
        _orm_context = await Tortoise.init(
            config=get_config(endpoint), **context_options
        )
        if endpoint.engine == "mysql" and endpoint.tls != "DISABLED":
            await asyncio.wait_for(
                inspect_database_tls(endpoint, Tortoise.get_connection("default")),
                DB_TIMEOUT_SECONDS,
            )
        from zhenxun.services.migration.restore import prepare_schema

        prepared = await prepare_schema(Tortoise.get_connection("default"))
        if not prepared:
            migrations = await collect_script_migrations()
            await run_script_migrations(migrations)
            # Tortoise may emit column comments/index SQL during generate_schemas().
            # On existing databases with newly added nullable fields, PostgreSQL can
            # fail before the post-generate SchemaGuard gets a chance to repair drift.
            await repair_safe_schema_drift()
            logger.debug("开始生成数据库表结构...")
            await Tortoise.generate_schemas()
            await run_script_migrations(migrations)
            finish_script_migrations(migrations)
            logger.debug("数据库表结构生成完毕!")
            await ensure_chat_history_nullable_columns()
            await ensure_group_plugin_scope_constraint()
            await repair_safe_schema_drift()
        bind_database_connection(endpoint)
        from zhenxun.services.migration.merge import ensure_local_identity

        if not prepared:
            await ensure_local_identity(Tortoise.get_connection("default"))
        _database_ready = True
        logger.info("Database loaded successfully!")
    except BaseException as e:
        try:
            await asyncio.wait_for(
                (_orm_context or Tortoise).close_connections(), DB_TIMEOUT_SECONDS
            )
        except BaseException as cleanup_error:
            logger.error("数据库初始化失败后的连接清理未完成", e=cleanup_error)
        _orm_context = None
        if isinstance(e, Exception):
            raise DbConnectError(f"数据库连接错误... e:{e}") from e
        raise


@PriorityLifecycle.on_startup(
    priority=2,
    stage="management",
    component_id="management:restore-import",
    depends_on=("management:database",),
)
async def import_restore():
    """导入从备份恢复或切换数据库时待写入的数据"""
    global _database_ready

    from zhenxun.services.migration.restore import import_pending

    try:
        await import_pending()
    except BaseException:
        _database_ready = False
        raise


@PriorityLifecycle.on_shutdown(
    priority=100,
    timeout=WORKER_SHUTDOWN_SECONDS + FINAL_SNAPSHOT_SECONDS,
    component_id="management:database",
)
async def disconnect():
    global _database_ready, _orm_context

    from zhenxun.configs.database import bind_database_connection
    from zhenxun.services.migration.restore import final_snapshot

    snapshot_error = None
    try:
        await final_snapshot()
    except BaseException as error:
        snapshot_error = error
    finally:
        _database_ready = False
        bind_database_connection(None)
    try:
        if _orm_context is not None:
            await _orm_context.close_connections()
            _orm_context = None
        else:
            await Tortoise.close_connections()
    except ConfigurationError:
        logger.debug("数据库连接未初始化，跳过关闭")
    except Exception as e:
        logger.error(f"关闭数据库连接时发生意外错误: {e}")

    if snapshot_error is not None:
        raise snapshot_error
