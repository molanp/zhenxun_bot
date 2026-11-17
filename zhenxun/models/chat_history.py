import asyncio
from contextlib import suppress
from datetime import datetime, timedelta
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from typing_extensions import Self

from tortoise import Tortoise, fields
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from zhenxun.services.db_context import Model
from zhenxun.services.db_context.schema_ops import (
    AlterColumnType,
    RenameColumn,
    SchemaOpRisk,
    SetColumnNullable,
)

if TYPE_CHECKING:
    from nonebot_plugin_alconna import UniMessage


class CleanupResult(tuple):
    """聊天清理计数与本轮批次状态，支持清空数、删除数解包。"""

    pause_reason: str
    batches: int
    scanned: int

    def __new__(
        cls, cleared: int, deleted: int, pause_reason: str, batches: int, scanned: int
    ):
        result = super().__new__(cls, (cleared, deleted))
        result.pause_reason = pause_reason
        result.batches = batches
        result.scanned = scanned
        return result


async def ensure_chat_history_nullable_columns() -> None:
    """Rebuild legacy SQLite chat history tables with nullable text columns."""
    connection = Tortoise.get_connection("default")
    raw_dialect = str(
        getattr(getattr(connection, "capabilities", None), "dialect", "") or ""
    ).lower()
    if not raw_dialect.startswith("sqlite"):
        return

    columns = await connection.execute_query_dict("PRAGMA table_info(chat_history)")
    if not columns:
        return
    nullable_columns = {"group_id", "text", "plain_text"}
    if not any(
        str(row.get("name")) in nullable_columns and int(row.get("notnull") or 0)
        for row in columns
    ):
        return

    column_names = [str(row["name"]) for row in columns if row.get("name")]
    definitions: list[str] = []
    for row in columns:
        name = str(row["name"])
        sql_type = str(row.get("type") or "TEXT")
        definition = f'"{name.replace(chr(34), chr(34) + chr(34))}" {sql_type}'
        if int(row.get("pk") or 0) == 1:
            definition += " PRIMARY KEY"
            if name == "id" and sql_type.upper() == "INTEGER":
                definition += " AUTOINCREMENT"
        elif int(row.get("notnull") or 0) and name not in nullable_columns:
            definition += " NOT NULL"
        if row.get("dflt_value") is not None:
            definition += f" DEFAULT {row['dflt_value']}"
        definitions.append(definition)

    indexes = await connection.execute_query_dict("PRAGMA index_list(chat_history)")
    index_definitions: list[tuple[str, tuple[str, ...], bool]] = []
    for index in indexes:
        name = str(index.get("name") or "")
        if not name or name.startswith("sqlite_autoindex"):
            continue
        info = await connection.execute_query_dict(
            f'PRAGMA index_info("{name.replace(chr(34), chr(34) + chr(34))}")'
        )
        index_columns = tuple(
            str(item["name"])
            for item in sorted(info, key=lambda item: int(item.get("seqno") or 0))
            if item.get("name")
        )
        if index_columns:
            index_definitions.append((name, index_columns, bool(index.get("unique"))))

    temporary_table = f"chat_history__nullable_{os.getpid()}"

    def quote(value: str) -> str:
        return f'"{value.replace(chr(34), chr(34) + chr(34))}"'

    table_sql = quote("chat_history")
    temporary_sql = quote(temporary_table)
    columns_sql = ", ".join(quote(name) for name in column_names)
    statements = [
        "PRAGMA foreign_keys=OFF",
        f"DROP TABLE IF EXISTS {temporary_sql}",
        f"CREATE TABLE {temporary_sql} ({', '.join(definitions)})",
        f"INSERT INTO {temporary_sql} ({columns_sql}) "
        f"SELECT {columns_sql} FROM {table_sql}",
        f"DROP TABLE {table_sql}",
        f"ALTER TABLE {temporary_sql} RENAME TO {table_sql}",
    ]
    statements.extend(
        "CREATE "
        f"{'UNIQUE ' if unique else ''}INDEX {quote(name)} ON {table_sql}"
        f"({', '.join(quote(column) for column in index_columns)})"
        for name, index_columns, unique in index_definitions
    )
    statements.append("PRAGMA foreign_keys=ON")

    try:
        async with in_transaction(connection_name="default") as transaction:
            for statement in statements:
                await transaction.execute_query(statement)
    except Exception:
        with suppress(Exception):
            await connection.execute_query("PRAGMA foreign_keys=ON")
        raise


class ChatHistory(Model):
    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    """自增id"""
    user_id = fields.CharField(255)
    """用户id"""
    group_id = fields.CharField(255, null=True)
    """群聊id"""
    text = fields.TextField(null=True)
    """文本内容"""
    plain_text = fields.TextField(null=True)
    """纯文本"""
    create_time = fields.DatetimeField(auto_now_add=True)
    """创建时间"""
    bot_id = fields.CharField(255, null=True)
    """bot记录id"""
    platform = fields.CharField(255, null=True)
    """平台"""
    message_id = fields.CharField(255, null=True)
    """消息id"""
    segments = fields.JSONField(null=True)
    """结构化消息段，媒体只存地址"""
    is_bot = fields.BooleanField(default=False)
    """是否为 Bot 自己的发言"""

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chat_history"
        table_description = "聊天记录数据表"
        indexes: ClassVar = [
            ("user_id", "create_time"),
            ("group_id", "create_time"),
            ("user_id", "group_id"),
            ("bot_id", "group_id", "create_time"),
            ("bot_id", "create_time"),
        ]

    @staticmethod
    def dump_segments(message: "UniMessage") -> list[dict[str, Any]]:
        """把消息转为可存储的结构化消息段，去掉内嵌的二进制和本地路径

        参数:
            message: 消息

        返回:
            list[dict[str, Any]]: 消息段
        """
        segments = []
        for seg in message.dump():
            if not isinstance(seg, dict):
                continue
            item = {}
            for key, value in seg.items():
                if key in ("raw", "path") or value is None:
                    continue
                if isinstance(value, str):
                    if value.startswith(("base64://", "file://")):
                        continue
                    value = value.replace("\x00", "")
                item[key] = value
            segments.append(item)
        return segments

    @classmethod
    async def cleanup(
        cls, segment_days: int, retention_days: int, *, progress: dict | None = None
    ) -> CleanupResult:
        """按主键分批清理过期记录和消息段，数据库繁忙或关闭时暂停。

        参数:
            segment_days: 结构化消息段保留天数
            retention_days: 记录保留天数
            progress: 每批提交确认后更新的计数映射

        返回:
            CleanupResult: 清空数、删除数及暂停原因、批次和扫描数
        """
        from zhenxun.services.lifecycle import lifecycle_kernel
        from zhenxun.services.message_load import should_pause_db_tasks

        now = datetime.now()
        cleared = deleted = batches = scanned = 0
        if progress is not None:
            progress.update(processed=0, cleared=0, deleted=0, batches=0, scanned=0)
        pause_reason = ""
        if segment_days <= 0 and retention_days <= 0:
            return CleanupResult(0, 0, "disabled", 0, 0)
        if lifecycle_kernel.shutdown_requested:
            return CleanupResult(0, 0, "shutdown", 0, 0)
        if should_pause_db_tasks():
            return CleanupResult(0, 0, "db_pressure", 0, 0)
        last = await cls.all().order_by("-id").limit(1).values_list("id", flat=True)
        high_water = last[0] if last else 0
        cursor = 0
        delete_cutoff = now - timedelta(days=retention_days)
        segment_cutoff = now - timedelta(days=segment_days)
        while cursor < high_water:
            if lifecycle_kernel.shutdown_requested:
                pause_reason = "shutdown"
                break
            if should_pause_db_tasks():
                pause_reason = "db_pressure"
                break
            ids = (
                await cls.filter(id__gt=cursor, id__lte=high_water)
                .order_by("id")
                .limit(500)
                .values_list("id", flat=True)
            )
            if not ids:
                break
            if lifecycle_kernel.shutdown_requested:
                pause_reason = "shutdown"
                break
            if should_pause_db_tasks():
                pause_reason = "db_pressure"
                break
            async with in_transaction(
                connection_name=cls._meta.default_connection
            ) as tx:
                if retention_days > 0:
                    deleted += (
                        await cls.filter(id__in=ids, create_time__lt=delete_cutoff)
                        .using_db(tx)
                        .delete()
                    )
                if segment_days > 0:
                    cleared += (
                        await cls.filter(
                            id__in=ids,
                            create_time__lt=segment_cutoff,
                            segments__isnull=False,
                        )
                        .using_db(tx)
                        .update(segments=None)
                    )
            cursor = ids[-1]
            scanned += len(ids)
            batches += 1
            if progress is not None:
                progress.update(
                    processed=cleared + deleted,
                    cleared=cleared,
                    deleted=deleted,
                    batches=batches,
                    scanned=scanned,
                )
            await asyncio.sleep(0)
        return CleanupResult(cleared, deleted, pause_reason, batches, scanned)

    @classmethod
    def _platform_from_scope(cls, platform_scope: str | None) -> str | None:
        """Map the new fine-grained scope back to the legacy platform column."""
        if not platform_scope:
            return None
        scope = str(platform_scope).lower()
        if scope in {"qq", "qq_client", "qq_api"} or scope.startswith("qq_"):
            return "qq"
        if "onebot" in scope:
            return "qq"
        return scope

    @classmethod
    def scoped_query(cls, platform_scope: str | None = None, **filters: Any):
        """Return a chat-history query compatible with platform_scope callers.

        chat_history currently stores the coarse legacy ``platform`` column rather
        than a dedicated ``platform_scope`` column, so this method intentionally
        stays as a thin compatibility shim.
        """
        query = cls.filter(**filters)
        if not platform_scope:
            return query
        if "platform" in filters or any(k.startswith("platform__") for k in filters):
            return query

        platform = cls._platform_from_scope(platform_scope)
        if not platform:
            return query
        if platform == "qq" and str(platform_scope).lower() in {"qq", "qq_client"}:
            return query.filter(
                Q(platform=platform) | Q(platform__isnull=True) | Q(platform="")
            )
        return query.filter(platform=platform)

    @classmethod
    async def get_group_msg_rank(
        cls,
        gid: str | None,
        limit: int = 10,
        order: str = "DESC",
        date_scope: tuple[datetime, datetime] | None = None,
    ) -> list[tuple[str, int]]:
        """获取排行数据

        参数:
            gid: 群号
            limit: 获取数量
            order: 排序类型，desc，des
            date_scope: 日期范围
        """
        from zhenxun.services.hot_query_cache import get_chat_history_rank_cached

        return await get_chat_history_rank_cached(cls, gid, limit, order, date_scope)

    @classmethod
    async def get_group_first_msg_datetime(
        cls, group_id: str | None
    ) -> datetime | None:
        """获取群第一条记录消息时间

        参数:
            group_id: 群组id
        """
        from zhenxun.services.hot_query_cache import (
            get_chat_history_first_msg_datetime_cached,
        )

        return await get_chat_history_first_msg_datetime_cached(cls, group_id)

    @classmethod
    async def get_message(
        cls,
        uid: str | None,
        gid: str | None,
        type_: Literal["user", "group"],
        msg_type: Literal["private", "group"] | None = None,
        days: int | tuple[datetime, datetime] | None = None,
        platform_scope: str | None = None,
    ) -> list[Self]:
        """获取消息查询query

        参数:
            uid: 用户id
            gid: 群聊id
            type_: 类型，私聊或群聊
            msg_type: 消息类型，用户或群聊
            days: 限制日期
            platform_scope: 兼容细粒度平台作用域
        """
        if type_ == "user":
            query = cls.scoped_query(platform_scope=platform_scope, user_id=uid)
            if msg_type == "private":
                query = query.filter(group_id__isnull=True)
            elif msg_type == "group":
                query = query.filter(group_id__not_isnull=True)
        else:
            query = cls.scoped_query(
                platform_scope=platform_scope, group_id=gid, is_bot=False
            )
            if uid:
                query = query.filter(user_id=uid)
        if days:
            if isinstance(days, int):
                query = query.filter(
                    create_time__gte=datetime.now() - timedelta(days=days)
                )
            elif isinstance(days, tuple):
                query = query.filter(create_time__range=days)
        return await query.all()  # type: ignore

    @classmethod
    async def _run_script(cls):
        return [
            AlterColumnType(
                "chat_history",
                "message_id",
                {"postgres": "character varying(255)", "mysql": "VARCHAR(255)"},
                nullable=True,
                risk=SchemaOpRisk.SAFE,
            ),
            # 允许 group_id 为空
            SetColumnNullable("chat_history", "group_id", {"mysql": "VARCHAR(255)"}),
            # 允许 text 为空
            SetColumnNullable("chat_history", "text", {"mysql": "LONGTEXT"}),
            # 允许 plain_text 为空
            SetColumnNullable("chat_history", "plain_text", {"mysql": "LONGTEXT"}),
            # 将user_id改为user_id
            RenameColumn("chat_history", "user_qq", "user_id"),
            AlterColumnType(
                "chat_history",
                "user_id",
                {"postgres": "character varying(255)", "mysql": "VARCHAR(255)"},
            ),
            AlterColumnType(
                "chat_history",
                "group_id",
                {"postgres": "character varying(255)", "mysql": "VARCHAR(255)"},
            ),
        ]
