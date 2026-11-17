from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any, Generic, SupportsIndex, TypeVar, cast, overload

from zhenxun.services.ai.core.protocols.tool import ToolExecutable, ToolResolvable
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.tool import FunctionTool
from zhenxun.services.ai.tools.models import ResolvedToolPayload

T = TypeVar("T", bound=ToolExecutable)


class ToolCollection(Generic[T]):
    """不可变的工具集合"""

    def __init__(self, iterable: Iterable[T] | None = None):
        """初始化工具集合。"""
        self._tuple: tuple[T, ...] = tuple(iterable) if iterable else ()
        self._name_cache: dict[str, T] = {}
        self._build_name_cache()

    def _build_name_cache(self) -> None:
        """构建工具名称小写到工具实例的映射缓存。"""
        self._name_cache.clear()
        for tool in self._tuple:
            name = getattr(tool, "name", None)
            if name:
                self._name_cache[name.lower()] = tool

    def __iter__(self):
        """获取工具元组的迭代器。"""
        return iter(self._tuple)

    def __len__(self):
        """获取工具集合中的工具数量。"""
        return len(self._tuple)

    def __bool__(self):
        """检查工具集合是否非空。"""
        return bool(self._tuple)

    @overload
    def __getitem__(self, key: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[T, ...]: ...

    @overload
    def __getitem__(self, key: str) -> T: ...

    def __getitem__(self, key: Any) -> Any:
        """按索引、切片或名称获取工具。"""
        if isinstance(key, str):
            return self._name_cache[key.lower()]
        return self._tuple[key]

    def get(self, key: str, default: Any = None) -> T | Any:
        """通过名称获取工具，若不存在则返回默认值。"""
        return self._name_cache.get(key.lower(), default)

    def keys(self):
        """获取所有工具名称缓存的键。"""
        return self._name_cache.keys()

    def values(self):
        """获取所有已缓存的工具实例。"""
        return self._name_cache.values()

    def items(self):
        """获取所有工具名称与实例的键值对。"""
        return self._name_cache.items()


async def _resolve_one(item: Any, context: RunContext | None) -> ResolvedToolPayload:
    """解析单个工具定义

    参数:
        item: 工具、工具箱、可解析对象或普通函数
        context: 运行上下文

    返回:
        ResolvedToolPayload: 工具载荷
    """
    if hasattr(item, "resolve"):
        return await cast(ToolResolvable, item).resolve(context)
    if callable(item):
        return await FunctionTool(func=item).resolve(context)
    raise TypeError(f"不支持的工具定义: {type(item)}")


async def resolve_tools(
    tool_definitions: Iterable[Any] | None, context: RunContext | None = None
) -> ResolvedToolPayload:
    """解析工具定义并合并为统一载荷

    参数:
        tool_definitions: 工具定义列表，可嵌套列表
        context: 运行上下文

    返回:
        ResolvedToolPayload: 合并后的工具载荷
    """
    defs: list[Any] = []

    def _flatten(items: Iterable[Any]) -> None:
        for item in items:
            if isinstance(item, list):
                _flatten(item)
            else:
                defs.append(item)

    _flatten(tool_definitions or [])
    from zhenxun.services.ai.run.context import RunContext
    from zhenxun.services.ai.run.session import session_manager

    owner = context or RunContext()
    token = owner.run.cancellation_token
    if token:
        token.raise_if_cancelled()
    tasks = [asyncio.create_task(_resolve_one(item, context)) for item in defs]
    session_manager.own_preparations(owner, tasks)
    owner.state["_preparation"] = {"status": "running", "sources": len(tasks)}
    batch = asyncio.gather(*tasks)
    batch.add_done_callback(
        lambda future: None if future.cancelled() else future.exception()
    )
    try:
        payloads = await asyncio.shield(batch)
        if token:
            token.raise_if_cancelled()
    except BaseException as error:
        owner.state["_preparation"] = {
            "status": "cancelled"
            if isinstance(error, asyncio.CancelledError)
            else "failed",
            "reason": type(error).__name__,
        }
        await session_manager.drain_preparations(owner, tasks)
        raise
    owner.state["_preparation"] = {"status": "completed", "sources": len(tasks)}
    final_payload = ResolvedToolPayload()
    for item, payload in zip(defs, payloads):
        if payload:
            if not payload.injection_sources:
                payload.injection_sources.append(
                    {
                        "source": type(item).__name__,
                        "role": "user",
                        "characters": sum(map(len, payload.reminders)),
                        "count": len(payload.reminders),
                        "temporary": True,
                    }
                )
            final_payload.merge(payload)
    return final_payload
