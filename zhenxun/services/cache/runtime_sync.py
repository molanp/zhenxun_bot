"""运行时缓存的注册、跨进程同步、变更记录与刷新调度"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextvars import ContextVar
import inspect
import json
import time
from typing import Any, ClassVar
import uuid

from zhenxun.services.cache.config import CacheMode
from zhenxun.services.db_context.utils import db_dialect, with_db_timeout
from zhenxun.services.db_context.write import (
    CachePublicationError,
    CacheUnavailable,
    on_publication_failed,
)
from zhenxun.services.log import logger
from zhenxun.services.message_load import is_db_unhealthy

LOG_COMMAND = "RuntimeCache"

RUNTIME_CACHE_SYNC_ENABLED = True
RUNTIME_CACHE_SYNC_CHANNEL = "ZHENXUN_RUNTIME_CACHE_SYNC"
RUNTIME_CACHE_REVISION_PREFIX = f"{RUNTIME_CACHE_SYNC_CHANNEL}:REVISION"
RUNTIME_CACHE_LOAD_RETRY_SECONDS = 1.0
RUNTIME_CACHE_STARTUP_REFRESH_SKIP_SECONDS = 5.0
RUNTIME_CACHE_DB_TIMEOUT_SECONDS = 3.0
RUNTIME_CACHE_REVISION_AUDIT_SECONDS = 30.0

INSTANCE_ID = uuid.uuid4().hex
_LIFECYCLE_CONTEXT: Any | None = None
_APPLYING_REMOTE_CACHE_EVENT: ContextVar[bool] = ContextVar(
    "APPLYING_REMOTE_RUNTIME_CACHE_EVENT",
    default=False,
)
RUNTIME_CACHES: dict[str, type[RuntimeMemoryCache]] = {}


def _redis_enabled() -> bool:
    from zhenxun.services.cache import cache_config

    return cache_config.cache_mode == CacheMode.REDIS and bool(cache_config.redis_host)


def set_lifecycle_context(context: Any | None) -> None:
    """设置运行时缓存后台任务所属的生命周期上下文

    参数:
        context: 生命周期上下文，为空时使用普通 asyncio 任务
    """
    global _LIFECYCLE_CONTEXT
    _LIFECYCLE_CONTEXT = context


def _spawn_runtime_task(
    coroutine,
    *,
    name: str,
    degrade_on_error: bool = True,
) -> asyncio.Task[Any]:
    if _LIFECYCLE_CONTEXT is not None:
        return _LIFECYCLE_CONTEXT.spawn_task(
            coroutine,
            name=name,
            degrade_on_error=degrade_on_error,
        )
    return asyncio.create_task(coroutine, name=name)


class RuntimeMemoryCache:
    """运行时内存缓存基类，子类定义时按 label 注册到 RUNTIME_CACHES"""

    label: ClassVar[str]
    """同步与刷新使用的缓存名"""
    table: ClassVar[str]
    """对应的数据表"""
    refresh_interval: ClassVar[int] = 0
    """兜底全量刷新间隔（秒）"""
    indexes: ClassVar[tuple[str, ...]] = ()
    """用于统计条目数的索引属性名"""
    _lock: ClassVar[asyncio.Lock]
    _loaded: ClassVar[bool] = False
    _last_refresh: ClassVar[float] = 0.0
    _last_error: ClassVar[str | None] = None
    _publication_failed: ClassVar[bool] = False
    _publication_generation: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._lock = asyncio.Lock()
        RUNTIME_CACHES[cls.label] = cls

    @classmethod
    async def _refresh_snapshot(cls) -> None:
        raise NotImplementedError

    @classmethod
    async def apply_sync_event(cls, action: str, data: dict[str, Any]) -> None:
        raise NotImplementedError

    @classmethod
    async def refresh(cls) -> None:
        """与定时、按需请求共享的全量刷新"""
        await runtime_cache_refresh_coordinator.refresh_one(cls)

    @classmethod
    async def reconcile(cls) -> None:
        """批量写入后全量刷新，并通知其他实例刷新"""
        RuntimeCacheSync.note_local_mutation(cls.label)
        await cls.refresh()
        if cls._last_error:
            raise RuntimeError(f"Cache reconciliation failed: {cls.label}")
        RuntimeCacheMutation.publish(cls.label, "refresh", {})

    @classmethod
    async def ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        await RuntimeCacheMutation.ensure_loaded(cls)
        RuntimeCacheMutation.require_fresh(cls)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def health(cls) -> dict[str, Any]:
        """获取缓存健康状态，不访问数据库

        返回:
            dict[str, Any]: 加载状态、条目数、最近刷新时间与错误
        """
        return {
            "loaded": bool(cls._loaded),
            "entry_count": sum(len(getattr(cls, name)) for name in cls.indexes),
            "last_refresh": float(cls._last_refresh or 0.0),
            "last_error": cls._last_error,
        }


class RuntimeCacheSync:
    _redis: ClassVar[Any | None] = None
    _pubsub: ClassVar[Any | None] = None
    _task: ClassVar[asyncio.Task | None] = None
    _publish_tasks: ClassVar[set[asyncio.Task]] = set()
    _ready: ClassVar[bool] = False
    _channel: ClassVar[str] = ""
    _publish_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _last_revision: ClassVar[dict[str, int]] = {}
    _local_mutation: ClassVar[dict[str, int]] = {}
    _revision_reconcile: ClassVar[set[str]] = set()
    _audited_revision: ClassVar[dict[str, int]] = {}

    @classmethod
    def _sync_enabled(cls) -> bool:
        enabled = RUNTIME_CACHE_SYNC_ENABLED
        return enabled and _redis_enabled()

    @classmethod
    async def start(cls) -> None:
        if cls._ready:
            return
        if not cls._sync_enabled():
            return
        try:
            import redis.asyncio as redis_async
        except ImportError:
            logger.warning(
                "redis not installed, runtime cache sync disabled", LOG_COMMAND
            )
            return

        from zhenxun.services.cache import cache_config

        host = cache_config.redis_host
        if not host:
            return
        port = cache_config.redis_port or 6379
        password = cache_config.redis_password
        cls._channel = RUNTIME_CACHE_SYNC_CHANNEL
        try:
            cls._redis = redis_async.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
            )
            cls._pubsub = cls._redis.pubsub()
            if cls._pubsub is None:
                return
            await cls._pubsub.subscribe(cls._channel)
            # Seed the watermark before consuming deltas. A worker joining an
            # active cluster must not mistake its first event for full history.
            for cache_type in (*RUNTIME_CACHES, "permission"):
                try:
                    value = await cls._redis.get(
                        f"{RUNTIME_CACHE_REVISION_PREFIX}:{cache_type}"
                    )
                except Exception as error:
                    logger.warning(
                        f"runtime cache revision seed failed: {cache_type}",
                        LOG_COMMAND,
                        e=error,
                    )
                    continue
                if value is not None:
                    with suppress(TypeError, ValueError):
                        cls._last_revision[cache_type] = int(value)
            cls._task = _spawn_runtime_task(
                cls._listen_loop(), name="runtime-cache-sync"
            )
            cls._ready = True
            logger.info("runtime cache sync enabled", LOG_COMMAND)
        except Exception as exc:
            logger.error("runtime cache sync init failed", LOG_COMMAND, e=exc)
            await cls.stop()

    @classmethod
    async def stop(cls) -> None:
        cls._ready = False
        if cls._publish_tasks:
            tasks = list(cls._publish_tasks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
            finally:
                cls._publish_tasks.difference_update(tasks)
        task, cls._task = cls._task, None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            if cls._pubsub is not None:
                await cls._pubsub.close()
        except Exception:
            pass
        cls._pubsub = None
        try:
            if cls._redis is not None:
                await cls._redis.close()
        except Exception:
            pass
        cls._redis = None
        cls._last_revision.clear()
        cls._local_mutation.clear()
        cls._revision_reconcile.clear()
        cls._audited_revision.clear()

    @classmethod
    def refresh_revision(cls, cache_type: str) -> tuple[int | None, int]:
        return cls._last_revision.get(cache_type), cls._local_mutation.get(
            cache_type, 0
        )

    @classmethod
    def note_local_mutation(cls, cache_type: str) -> None:
        cls._local_mutation[cache_type] = cls._local_mutation.get(cache_type, 0) + 1

    @classmethod
    def refresh_is_stable(
        cls, cache_type: str, revision: tuple[int | None, int]
    ) -> bool:
        """Finish a DB refresh only if no cache mutation arrived during it."""
        current = cls.refresh_revision(cache_type)
        if revision != current:
            cls._revision_reconcile.add(cache_type)
            return False
        cls._revision_reconcile.discard(cache_type)
        return True

    @classmethod
    def publish_event(cls, cache_type: str, action: str, data: dict[str, Any]) -> None:
        if not cls._ready:
            return
        task = _spawn_runtime_task(
            cls._publish(cache_type, action, data),
            name="runtime-cache-publish",
            degrade_on_error=False,
        )
        cls._publish_tasks.add(task)
        task.add_done_callback(cls._publish_tasks.discard)

    @classmethod
    async def _publish(cls, cache_type: str, action: str, data: dict[str, Any]) -> None:
        if not cls._ready or cls._redis is None:
            return
        try:
            # Redis Pub/Sub is unordered across concurrent publish tasks. Keep
            # revision assignment and publication in one critical section so a
            # receiver can detect both reordering and dropped messages.
            async with cls._publish_lock:
                revision = await cls._redis.incr(
                    f"{RUNTIME_CACHE_REVISION_PREFIX}:{cache_type}"
                )
                payload = {
                    "source": INSTANCE_ID,
                    "type": cache_type,
                    "action": action,
                    "revision": int(revision),
                    "data": data,
                }
                await cls._redis.publish(cls._channel, json.dumps(payload))
        except Exception as exc:
            logger.error("runtime cache sync publish failed", LOG_COMMAND, e=exc)

    @classmethod
    def _reconcile(cls, cache_type: str, reason: str) -> None:
        """缓存事件缺失或无法应用时，改为从数据库重新加载该缓存

        权限策略将已加载的群聊与私聊策略标记为待刷新，使用前逐个重新读取。

        参数:
            cache_type: 缓存类型
            reason: 原因
        """
        if cache_type == "permission":
            from zhenxun.services.bot_group_policy import bot_group_policy_service

            bot_group_policy_service.mark_all_stale()
            RuntimeCacheMutation.revision += 1
        elif cache_cls := RUNTIME_CACHES.get(cache_type):
            cls._revision_reconcile.add(cache_type)
            RuntimeCacheMutation.mark_error(cache_cls, RuntimeError(reason))
            runtime_cache_refresh_coordinator.request_refresh(cache_cls)

    @classmethod
    async def _accept_revision(cls, cache_type: str, revision: Any) -> bool:
        """Reject stale events and reconcile a detected Pub/Sub gap."""
        cache_cls = RUNTIME_CACHES.get(cache_type)
        try:
            current = int(revision)
        except (TypeError, ValueError):
            # Older workers did not publish revisions. A refresh is safer than
            # applying an unversioned snapshot over a versioned one.
            cls._reconcile(cache_type, "runtime_cache_legacy_event")
            return False
        if current <= 0:
            return False
        previous = cls._last_revision.get(cache_type)
        if previous is not None and current <= previous:
            return False
        cls._last_revision[cache_type] = current
        if cache_type in cls._revision_reconcile:
            if cache_cls is not None:
                runtime_cache_refresh_coordinator.request_refresh(cache_cls)
            return False
        # A worker that has not completed its authoritative initial load must
        # not apply a single Pub/Sub delta as if it had seen the preceding
        # history. The refresh will establish a complete snapshot first.
        if cache_cls is not None and not getattr(cache_cls, "_loaded", False):
            cls._revision_reconcile.add(cache_type)
            runtime_cache_refresh_coordinator.request_refresh(cache_cls)
            return False
        if previous is not None and current > previous + 1:
            cls._reconcile(
                cache_type, f"runtime_cache_revision_gap:{previous}->{current}"
            )
            return False
        return True

    @classmethod
    async def _audit_revisions(cls) -> None:
        """核对 Redis 中的修订号，发现事件丢失或计数器回退时全量对账

        上一轮核对时已分配的修订号到本轮仍未收到，说明事件在断线或发布失败时丢失；
        Redis 中的修订号小于已收到的修订号，说明计数器被重置，以 Redis 中的值为准。
        """
        if cls._redis is None:
            return
        cache_types = (*RUNTIME_CACHES, "permission")
        values = await cls._redis.mget(
            [f"{RUNTIME_CACHE_REVISION_PREFIX}:{name}" for name in cache_types]
        )
        for cache_type, value in zip(cache_types, values):
            try:
                current = int(value or 0)
            except (TypeError, ValueError):
                continue
            received = cls._last_revision.get(cache_type, 0)
            audited = cls._audited_revision.get(cache_type, 0)
            if current < received:
                cls._last_revision[cache_type] = current
                cls._reconcile(
                    cache_type, f"runtime_cache_revision_reset:{received}->{current}"
                )
            elif received < audited:
                cls._last_revision[cache_type] = audited
                cls._reconcile(
                    cache_type, f"runtime_cache_event_lost:{received}->{audited}"
                )
            cls._audited_revision[cache_type] = current

    @classmethod
    async def _listen_loop(cls) -> None:
        """接收其他实例的缓存事件并定期核对修订号，连接中断时等待自动重连"""
        if cls._pubsub is None:
            return
        failed = False
        audit_at = time.monotonic() + RUNTIME_CACHE_REVISION_AUDIT_SECONDS
        while True:
            try:
                if time.monotonic() >= audit_at:
                    audit_at = time.monotonic() + RUNTIME_CACHE_REVISION_AUDIT_SECONDS
                    await cls._audit_revisions()
                message = await cls._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not failed:
                    logger.error(
                        "runtime cache sync listener failed", LOG_COMMAND, e=exc
                    )
                failed = True
                await asyncio.sleep(1.0)
                continue
            if failed:
                failed = False
                logger.info("runtime cache sync listener recovered", LOG_COMMAND)
            if not message:
                await asyncio.sleep(0.05)
                continue
            await cls._handle_message(message.get("data"))

    @classmethod
    async def _handle_message(cls, raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, bytes | bytearray):
            try:
                raw = raw.decode()
            except Exception:
                return
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            return
        cache_type = payload.get("type")
        action = payload.get("action")
        data = payload.get("data") or {}
        if not isinstance(cache_type, str) or not await cls._accept_revision(
            cache_type, payload.get("revision")
        ):
            return
        if payload.get("source") == INSTANCE_ID:
            return
        token = _APPLYING_REMOTE_CACHE_EVENT.set(True)
        try:
            if cache_cls := RUNTIME_CACHES.get(cache_type):
                await cache_cls.apply_sync_event(action, data)
            elif cache_type == "permission":
                if action == "policy_changed":
                    from zhenxun.services.bot_group_policy import (
                        bot_group_policy_service,
                        group_key,
                    )

                    await bot_group_policy_service.refresh_policy(
                        group_key(
                            data.get("bot_id"),
                            data.get("platform_scope"),
                            data.get("group_id"),
                            data.get("channel_id"),
                        )
                    )
                RuntimeCacheMutation.revision += 1
        except Exception as exc:
            logger.warning(
                f"runtime cache sync event apply failed: {cache_type}",
                LOG_COMMAND,
                e=exc,
            )
            cls._reconcile(cache_type, "runtime_cache_sync_apply_failed")
        finally:
            _APPLYING_REMOTE_CACHE_EVENT.reset(token)


class RuntimeCacheMutation:
    """Small helpers for runtime cache mutation bookkeeping.

    Cache classes still own their storage layout. This helper centralizes the
    shared mutation side effects: health markers and cross-process publish.
    """

    # 运行态缓存每次发布、刷新或失效时递增，权限结果缓存据此失效
    revision: ClassVar[int] = 0

    _load_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _retry_after: ClassVar[dict[str, float]] = {}

    @classmethod
    def publication_failed(cls, key: tuple | None) -> None:
        # Legacy/custom effects do not identify a cache. Preserve their existing
        # conservative repair; known runtime effects stay targeted.
        if key and key[0] not in {
            "runtime",
            "bulk",
            "group_settings",
            "group_settings_all",
        }:
            key = None
        changed = False
        for label, cache_cls in RUNTIME_CACHES.items():
            if key is not None and not (
                (key[0] == "runtime" and key[1] == cache_cls.__name__)
                or (key[0] == "bulk" and key[1] == cache_cls.table)
            ):
                continue
            cache_cls._publication_failed = True
            cls.mark_error(cache_cls, RuntimeError("cache_publication_failed"))
            cache_cls._publication_generation = (
                getattr(cache_cls, "_publication_generation", 0) + 1
            )
            cache_cls._loaded = False
            cls._retry_after.pop(label, None)
            runtime_cache_refresh_coordinator.request_refresh(cache_cls)
            changed = True
        if changed:
            cls.revision += 1

    @staticmethod
    def require_fresh(cache_cls: type, *, allow_unloaded: bool = False) -> None:
        if getattr(cache_cls, "_publication_failed", False):
            raise CachePublicationError(
                f"runtime_cache_reconciliation_pending:{cache_cls.__name__}"
            )

        if not allow_unloaded and not getattr(cache_cls, "_loaded", False):
            raise CacheUnavailable(f"runtime_cache_unavailable:{cache_cls.__name__}")

    @classmethod
    async def ensure_loaded(cls, cache_cls: type[RuntimeMemoryCache]) -> None:
        label = cache_cls.label
        if getattr(cache_cls, "_loaded", False):
            return
        if is_db_unhealthy():
            cls.mark_error(cache_cls, RuntimeError("database unhealthy"))
            return
        now = time.monotonic()
        if cls._retry_after.get(label, 0.0) > now:
            return
        lock = cls._load_locks.setdefault(label, asyncio.Lock())
        async with lock:
            if getattr(cache_cls, "_loaded", False):
                return
            if is_db_unhealthy():
                cls.mark_error(cache_cls, RuntimeError("database unhealthy"))
                return
            now = time.monotonic()
            if cls._retry_after.get(label, 0.0) > now:
                return
            try:
                refreshed = await runtime_cache_refresh_coordinator.refresh_one(
                    cache_cls
                )
                if refreshed:
                    return
                cls._retry_after[label] = (
                    time.monotonic() + RUNTIME_CACHE_LOAD_RETRY_SECONDS
                )
            except Exception as exc:
                cls.mark_error(cache_cls, exc)
                cls._retry_after[label] = (
                    time.monotonic() + RUNTIME_CACHE_LOAD_RETRY_SECONDS
                )

    @staticmethod
    async def read_db(cache_cls: type, coro, *, operation: str):
        if is_db_unhealthy():
            # 数据库降级时跳过本次刷新，已加载的数据没有过期证据，继续使用
            if getattr(cache_cls, "_loaded", False):
                setattr(cache_cls, "_last_error", "RuntimeError: database unhealthy")
            else:
                RuntimeCacheMutation.mark_error(
                    cache_cls, RuntimeError("database unhealthy")
                )
            if inspect.iscoroutine(coro):
                coro.close()
            return None
        generation = getattr(cache_cls, "_publication_generation", 0)
        try:
            result = await with_db_timeout(
                coro,
                timeout=RUNTIME_CACHE_DB_TIMEOUT_SECONDS,
                operation=operation,
                source="runtime_cache",
            )
            if generation != getattr(cache_cls, "_publication_generation", 0):
                RuntimeCacheMutation.mark_error(
                    cache_cls, RuntimeError("cache_invalidated_during_refresh")
                )
                return None
            return result
        except Exception as exc:
            RuntimeCacheMutation.mark_error(cache_cls, exc)
            return None

    @staticmethod
    def mark_refreshed(cache_cls: type) -> None:
        RuntimeCacheMutation.revision += 1
        setattr(cache_cls, "_loaded", True)
        setattr(cache_cls, "_last_refresh", time.time())
        setattr(cache_cls, "_last_error", None)
        setattr(cache_cls, "_publication_failed", False)

    @staticmethod
    def mark_error(cache_cls: type, exc: Exception) -> None:
        if getattr(cache_cls, "_loaded", False):
            RuntimeCacheMutation.revision += 1
        setattr(cache_cls, "_last_error", f"{type(exc).__name__}: {exc}")
        setattr(cache_cls, "_loaded", False)

    @staticmethod
    def publish(cache_type: str, action: str, data: dict[str, Any]) -> None:
        RuntimeCacheMutation.revision += 1
        if _APPLYING_REMOTE_CACHE_EVENT.get():
            return
        RuntimeCacheSync.note_local_mutation(cache_type)
        RuntimeCacheSync.publish_event(cache_type, action, data)


class RuntimeCacheRefreshCoordinator:
    """Own all full-table refreshes so SQLite never receives a refresh burst."""

    def __init__(self) -> None:
        self._task: asyncio.Task[Any] | None = None
        self._wake = asyncio.Event()
        self._next_due: dict[str, float] = {}
        self._failure_counts: dict[str, int] = {}
        self._inflight: dict[str, asyncio.Task] = {}
        self._requests: dict[str, int] = {}
        self._semaphore = None

    def request_refresh(self, cache_cls: type[RuntimeMemoryCache]) -> None:
        name = cache_cls.label
        self._requests[name] = self._requests.get(name, 0) + 1
        self._next_due[name] = time.monotonic()
        self._wake.set()
        if (
            (self._task is None or self._task.done())
            and _LIFECYCLE_CONTEXT is not None
            and _LIFECYCLE_CONTEXT.accepting
        ):
            self._task = _spawn_runtime_task(
                self._run(), name="runtime-cache-refresh-coordinator"
            )

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        now = time.monotonic()
        self._next_due = {
            name: now + cache_cls.refresh_interval + min(30.0, (index + 1) * 2.0)
            for index, (name, cache_cls) in enumerate(RUNTIME_CACHES.items())
            if cache_cls.refresh_interval > 0
        }
        self._wake.clear()
        self._task = _spawn_runtime_task(
            self._run(), name="runtime-cache-refresh-coordinator"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        self._wake.set()
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        tasks = list(self._inflight.values())
        for pending in tasks:
            if not pending.done() and not getattr(pending, "cancelling", lambda: 0)():
                pending.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()
        self._semaphore = None

    async def _refresh_entry(
        self, cache_cls: type[RuntimeMemoryCache], *, startup: bool = False
    ) -> tuple[str, bool]:
        name = cache_cls.label
        task = self._inflight.get(name)
        if task is None:
            requested = self._requests.get(name, 0)

            async def run():
                if self._semaphore is None:
                    sqlite = db_dialect() == "sqlite"
                    self._semaphore = asyncio.Semaphore(1 if sqlite else 3)
                try:
                    async with self._semaphore:
                        outcome = await self._perform_refresh(
                            name, cache_cls, startup=startup
                        )
                        if self._requests.get(name, 0) == requested:
                            interval = cache_cls.refresh_interval
                            if interval > 0:
                                self._next_due[name] = (
                                    time.monotonic() + self._retry_delay(name, interval)
                                )
                                self._wake.set()
                            else:
                                self._next_due.pop(name, None)
                        return outcome
                finally:
                    self._inflight.pop(name, None)

            task = _spawn_runtime_task(
                run(), name=f"runtime-cache-refresh:{name}", degrade_on_error=False
            )
            self._inflight[name] = task
        return await asyncio.shield(task)

    async def _perform_refresh(
        self, name: str, cache_cls: type[RuntimeMemoryCache], *, startup: bool = False
    ) -> tuple[str, bool]:
        if startup and getattr(cache_cls, "_loaded", False):
            last_refresh = float(getattr(cache_cls, "_last_refresh", 0.0) or 0.0)
            if time.time() - last_refresh <= RUNTIME_CACHE_STARTUP_REFRESH_SKIP_SECONDS:
                return name, True
        previous_refresh = float(getattr(cache_cls, "_last_refresh", 0.0) or 0.0)
        revision_before = RuntimeCacheSync.refresh_revision(name)
        try:
            await cache_cls._refresh_snapshot()
            if not RuntimeCacheSync.refresh_is_stable(name, revision_before):
                RuntimeCacheMutation.mark_error(
                    cache_cls, RuntimeError("cache_refresh_revision_changed")
                )
                self.request_refresh(cache_cls)
                return name, False
            current_refresh = float(getattr(cache_cls, "_last_refresh", 0.0) or 0.0)
            current_error = getattr(cache_cls, "_last_error", None)
            if current_error and current_refresh <= previous_refresh:
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
                return name, False
            self._failure_counts.pop(name, None)
            return name, True
        except Exception as error:
            RuntimeCacheMutation.mark_error(cache_cls, error)
            self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
            logger.warning(
                f"runtime cache refresh failed: {name}", LOG_COMMAND, e=error
            )
            return name, False

    def _retry_delay(self, name: str, interval: int) -> float:
        """失败退避，与 _run 循环同一公式。"""
        failures = self._failure_counts.get(name, 0)
        if not failures:
            return float(interval)
        return float(min(interval, max(5, 2**failures * 5)))

    async def refresh_one(self, cache_cls: type[RuntimeMemoryCache]) -> bool:
        _, success = await self._refresh_entry(cache_cls)
        return success

    async def refresh_all(self, *, startup: bool = False) -> dict[str, bool]:
        results = await asyncio.gather(
            *(
                self._refresh_entry(cache_cls, startup=startup)
                for cache_cls in RUNTIME_CACHES.values()
            )
        )
        return dict(results)

    async def _run(self) -> None:
        # stop() clears ownership before waking/cancelling the wait. Python 3.10
        # wait_for can consume cancellation when its inner wait just completed.
        while self._task is asyncio.current_task():
            now = time.monotonic()
            active_due = {
                name: due
                for name, due in self._next_due.items()
                if name in RUNTIME_CACHES
            }
            delay = max(0.0, min(active_due.values(), default=now + 60.0) - now)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
                self._wake.clear()
                continue
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            due_names = [name for name, due in active_due.items() if due <= now]
            if not due_names:
                continue
            await asyncio.gather(
                *(self._refresh_entry(RUNTIME_CACHES[name]) for name in due_names)
            )


runtime_cache_refresh_coordinator = RuntimeCacheRefreshCoordinator()
on_publication_failed(RuntimeCacheMutation.publication_failed)
