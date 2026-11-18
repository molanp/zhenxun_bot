"""
密钥与线路健康状态

密钥以哈希标识持久化，不保存原始密钥。
"""

import asyncio
from contextvars import Token
import hashlib
import json
import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.core.exceptions import (
    AuthenticationException,
    QuotaExceededException,
    RateLimitException,
)
from zhenxun.services.ai.utils.logger import log_llm as logger
from zhenxun.services.lifecycle.deadline import current_budget
from zhenxun.utils.pydantic_compat import model_dump, parse_as

HEALTH_FILE = DATA_PATH / "ai" / "api_key.json"


def key_id(api_key: str) -> str:
    """计算密钥的持久化标识

    参数:
        api_key: 原始密钥

    返回:
        str: 密钥的 SHA-256 摘要前 16 位
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def mask_key(api_key: str) -> str:
    """生成用于展示的脱敏密钥

    参数:
        api_key: 原始密钥

    返回:
        str: 仅保留首尾字符的密钥
    """
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


class KeyState(BaseModel):
    """单个密钥的健康状态"""

    status: Literal["healthy", "cooldown", "invalid"] = "healthy"
    successes: int = 0
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None


ROUTE_FAILURE_THRESHOLD = 2
"""线路连续失败多少次后熔断"""


class RouteState(BaseModel):
    """单条线路 (提供商/模型) 的熔断状态"""

    state: Literal["closed", "open", "half_open"] = "closed"
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None


class HealthState(BaseModel):
    """全部健康状态"""

    keys: dict[str, dict[str, KeyState]] = Field(default_factory=dict)
    """提供商名称 -> 密钥标识 -> 状态"""
    routes: dict[str, RouteState] = Field(default_factory=dict)
    """线路全名 -> 状态"""


class HealthManager:
    """密钥轮询、冷却与线路熔断管理"""

    def __init__(self):
        self.state = HealthState()
        self._cursor: dict[str, int] = {}
        self._dirty = False
        self._save_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """加载持久化状态并启动后台保存"""
        if HEALTH_FILE.exists():
            try:
                data = json.loads(HEALTH_FILE.read_text(encoding="utf-8") or "{}")
                self.state = parse_as(HealthState, data)
                self._dirty = "providers" in data
            except Exception as e:
                logger.warning(f"密钥状态文件无法解析，已重新开始记录: {e}")
                self._dirty = True
        if self._task is None or self._task.done():
            token: Token = current_budget.set(None)
            try:
                self._task = asyncio.create_task(self._save_loop())
            finally:
                current_budget.reset(token)

    async def shutdown(self) -> None:
        """停止后台保存并写入最新状态"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.save()

    async def _save_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.save()

    async def save(self) -> None:
        """把脏状态原子写入磁盘"""
        if not self._dirty:
            return
        async with self._save_lock:
            if not self._dirty:
                return
            self._dirty = False
            serialized = json.dumps(model_dump(self.state), ensure_ascii=False)
            temp_path = HEALTH_FILE.with_suffix(".json.tmp")

            def write() -> None:
                HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
                try:
                    temp_path.write_text(serialized, encoding="utf-8")
                    os.replace(temp_path, HEALTH_FILE)
                finally:
                    temp_path.unlink(missing_ok=True)

            writer = asyncio.ensure_future(asyncio.to_thread(write))
            try:
                await asyncio.shield(writer)
            except asyncio.CancelledError:
                await asyncio.wait({writer})
                self._dirty = True
                raise
            except Exception as e:
                self._dirty = True
                logger.error(f"保存密钥状态失败: {e}", e=e)

    def _key_state(self, provider: str, api_key: str) -> KeyState:
        return self.state.keys.setdefault(provider, {}).setdefault(
            key_id(api_key), KeyState()
        )

    def available_keys(
        self, provider: str, api_keys: list[str], exclude: set[str]
    ) -> list[str]:
        """获取未失效、未冷却且不在排除集合中的密钥

        参数:
            provider: 提供商名称
            api_keys: 提供商的全部密钥
            exclude: 本次调用已经失败的密钥标识

        返回:
            list[str]: 可用密钥
        """
        now = time.time()
        return [
            api_key
            for api_key in api_keys
            if key_id(api_key) not in exclude
            and (state := self._key_state(provider, api_key)).status != "invalid"
            and state.cooldown_until <= now
        ]

    def pick_key(
        self,
        provider: str,
        api_keys: list[str],
        exclude: set[str],
        *,
        affinity: str | None = None,
    ) -> str | None:
        """同一会话优先使用相同的健康密钥；无会话请求轮询。"""
        available = self.available_keys(provider, api_keys, exclude)
        if not available:
            return None
        if affinity:
            return max(
                available,
                key=lambda key: hashlib.sha256(
                    f"{provider}\0{affinity}\0{key_id(key)}".encode()
                ).digest(),
            )
        cursor = self._cursor.get(provider, 0)
        self._cursor[provider] = cursor + 1
        return available[cursor % len(available)]

    def key_recovery_time(self, provider: str, api_keys: list[str]) -> float | None:
        """获取最早恢复可用的冷却密钥时间戳，全部失效时为 None"""
        times = [
            state.cooldown_until
            for api_key in api_keys
            if (state := self._key_state(provider, api_key)).status != "invalid"
        ]
        return min(times) if times else None

    def record_key_success(self, provider: str, api_key: str) -> None:
        """记录密钥调用成功"""
        state = self._key_state(provider, api_key)
        state.successes += 1
        state.status = "healthy"
        state.cooldown_until = 0.0
        state.last_error = None
        self._dirty = True

    def record_key_failure(
        self,
        provider: str,
        api_key: str,
        error: Exception,
        cooldowns: dict[str, int],
    ) -> None:
        """记录密钥调用失败，并按错误类型设置冷却或失效

        参数:
            provider: 提供商名称
            api_key: 失败的密钥
            error: 失败原因
            cooldowns: rate_limit / quota 两类冷却秒数
        """
        state = self._key_state(provider, api_key)
        state.failures += 1
        state.last_error = str(error)[:256]
        retry_after = getattr(error, "details", {}).get("retry_after")
        if isinstance(error, AuthenticationException):
            state.status = "invalid"
        elif isinstance(error, QuotaExceededException | RateLimitException):
            kind = "quota" if isinstance(error, QuotaExceededException) else "rate"
            seconds = retry_after or cooldowns[kind]
            if seconds:
                state.status = "cooldown"
                state.cooldown_until = time.time() + float(seconds)
        self._dirty = True
        logger.debug(f"密钥 {mask_key(api_key)} 调用失败: {error}")

    def key_stats(self, provider: str, api_keys: list[str]) -> dict[str, Any]:
        """获取提供商的密钥统计，以脱敏密钥为键"""
        return {
            mask_key(api_key): model_dump(self._key_state(provider, api_key))
            for api_key in api_keys
        }

    def route_available(self, route: str) -> bool:
        """判断线路是否可用，熔断冷却结束后放行一次探测"""
        state = self.state.routes.get(route)
        if state is None or state.state == "closed":
            return True
        if state.state == "open" and time.time() >= state.cooldown_until:
            state.state = "half_open"
            logger.info(f"线路 {route} 熔断冷却结束，放行一次探测请求")
            return True
        return False

    def best_fallback(self, routes: list[str]) -> str:
        """在全部熔断的线路中选择最早恢复的一条"""
        return min(
            routes,
            key=lambda name: (
                state.cooldown_until if (state := self.state.routes.get(name)) else 0
            ),
        )

    def record_route_success(self, route: str) -> None:
        """记录线路成功并关闭熔断"""
        state = self.state.routes.setdefault(route, RouteState())
        state.successes += 1
        if state.state != "closed":
            logger.info(f"线路 {route} 已恢复")
        state.state = "closed"
        state.consecutive_failures = 0
        state.cooldown_until = 0.0
        state.last_error = None
        self._dirty = True

    def record_route_failure(self, route: str, error: Exception, cooldown: int) -> None:
        """记录线路失败，连续失败达到阈值或半开探测失败时打开熔断"""
        state = self.state.routes.setdefault(route, RouteState())
        state.failures += 1
        state.consecutive_failures += 1
        state.last_error = str(error)[:256]
        self._dirty = True
        if cooldown and (
            state.state == "half_open"
            or state.consecutive_failures >= ROUTE_FAILURE_THRESHOLD
        ):
            state.state = "open"
            state.cooldown_until = time.time() + cooldown
            logger.warning(f"线路 {route} 请求失败，熔断 {cooldown} 秒: {error}")
            return
        logger.warning(
            f"线路 {route} 请求失败 ({state.consecutive_failures}"
            f"/{ROUTE_FAILURE_THRESHOLD}): {error}"
        )


health_manager = HealthManager()
