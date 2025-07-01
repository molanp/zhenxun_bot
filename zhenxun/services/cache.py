from collections.abc import Callable
from datetime import datetime
from functools import wraps
import inspect
from typing import Any, ClassVar, Generic, TypeVar

from aiocache import Cache as AioCache

# from aiocache.backends.redis import RedisCache
from aiocache.base import BaseCache
from aiocache.serializers import JsonSerializer
import nonebot
from nonebot.compat import model_dump
from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel

from zhenxun.services.log import logger

__all__ = ["Cache", "CacheData", "CacheRoot"]

T = TypeVar("T")

LOG_COMMAND = "cache"

driver = nonebot.get_driver()


class Config(BaseModel):
    redis_host: str | None = None
    """redis地址"""
    redis_port: int | None = None
    """redis端口"""
    redis_password: str | None = None
    """redis密码"""
    redis_expire: int = 600
    """redis过期时间"""


config = nonebot.get_plugin_config(Config)


class DbCacheException(Exception):
    """缓存相关异常"""

    def __init__(self, info: str):
        self.info = info

    def __str__(self) -> str:
        return self.info


def validate_name(func: Callable):
    """验证缓存名称是否存在的装饰器"""

    def wrapper(self, name: str, *args, **kwargs):
        _name = name.upper()
        if _name not in CacheManager._data:
            raise DbCacheException(f"缓存数据 {name} 不存在")
        return func(self, _name, *args, **kwargs)

    return wrapper


class CacheGetter(BaseModel, Generic[T]):
    """缓存数据获取器"""

    get_func: Callable[..., Any] | None = None
    get_all_func: Callable[..., Any] | None = None

    async def get(self, cache_data: "CacheData", key: str, *args, **kwargs) -> T:
        """获取单个缓存数据"""
        if not self.get_func:
            data = await cache_data.get_key(key)
            if cache_data.result_model:
                return cache_data._deserialize_value(data, cache_data.result_model)
            return data

        if is_coroutine_callable(self.get_func):
            data = await self.get_func(cache_data, key, *args, **kwargs)
        else:
            data = self.get_func(cache_data, key, *args, **kwargs)

        if cache_data.result_model:
            return cache_data._deserialize_value(data, cache_data.result_model)
        return data

    async def get_all(self, cache_data: "CacheData", *args, **kwargs) -> dict[str, T]:
        """获取所有缓存数据"""
        if not self.get_all_func:
            data = await cache_data.get_all_data()
            if cache_data.result_model:
                return {
                    k: cache_data._deserialize_value(v, cache_data.result_model)
                    for k, v in data.items()
                }
            return data

        if is_coroutine_callable(self.get_all_func):
            data = await self.get_all_func(cache_data, *args, **kwargs)
        else:
            data = self.get_all_func(cache_data, *args, **kwargs)

        if cache_data.result_model:
            return {
                k: cache_data._deserialize_value(v, cache_data.result_model)
                for k, v in data.items()
            }
        return data


class CacheData(BaseModel):
    """缓存数据模型"""

    name: str
    func: Callable[..., Any]
    getter: CacheGetter | None = None
    updater: Callable[..., Any] | None = None
    with_refresh: Callable[..., Any] | None = None
    expire: int = 600  # 默认10分钟过期
    reload_count: int = 0
    lazy_load: bool = True  # 默认延迟加载
    result_model: type | None = None
    _keys: set[str] = set()  # 存储所有缓存键
    _cache: BaseCache | AioCache

    class Config:
        arbitrary_types_allowed = True

    def _deserialize_value(self, value: Any, target_type: type | None = None) -> Any:
        """反序列化值，将JSON数据转换回原始类型

        参数:
            value: 需要反序列化的值
            target_type: 目标类型，用于指导反序列化

        返回:
            反序列化后的值
        """
        if value is None:
            return None

        # 如果是字典且指定了目标类型
        if isinstance(value, dict) and target_type:
            # 处理Tortoise-ORM Model
            if hasattr(target_type, "_meta"):
                return self._extracted_from__deserialize_value_19(value, target_type)
            elif hasattr(target_type, "model_validate"):
                return target_type.model_validate(value)
            elif hasattr(target_type, "from_dict"):
                return target_type.from_dict(value)
            elif hasattr(target_type, "parse_obj"):
                return target_type.parse_obj(value)
            else:
                return target_type(**value)

        # 处理列表类型
        if isinstance(value, list):
            if not value:
                return value
            if (
                target_type
                and hasattr(target_type, "__origin__")
                and target_type.__origin__ is list
            ):
                item_type = target_type.__args__[0]
                return [self._deserialize_value(item, item_type) for item in value]
            return [self._deserialize_value(item) for item in value]

        # 处理字典类型
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}

        return value

    def _extracted_from__deserialize_value_19(self, value, target_type):
        # 处理字段值
        processed_value = {}
        for field_name, field_value in value.items():
            if field := target_type._meta.fields_map.get(field_name):
                # 跳过反向关系字段
                if hasattr(field, "_related_name"):
                    continue
                processed_value[field_name] = field_value

        logger.debug(f"处理后的值: {processed_value}")

        # 创建模型实例
        instance = target_type()
        # 设置字段值
        for field_name, field_value in processed_value.items():
            if field_name in target_type._meta.fields_map:
                field = target_type._meta.fields_map[field_name]
                # 设置字段值
                try:
                    if hasattr(field, "to_python_value"):
                        if not field.field_type:
                            logger.debug(f"字段 {field_name} 类型为空")
                            continue
                        field_value = field.to_python_value(field_value)
                    setattr(instance, field_name, field_value)
                except Exception as e:
                    logger.warning(f"设置字段 {field_name} 失败", e=e)

        # 设置 _saved_in_db 标志
        instance._saved_in_db = True
        return instance

    async def get_data(self) -> Any:
        """从缓存获取数据"""
        try:
            data = await self._cache.get(self.name)
            logger.debug(f"获取缓存 {self.name} 数据: {data}")

            # 如果数据为空，尝试重新加载
            # if data is None:
            #     logger.debug(f"缓存 {self.name} 数据为空，尝试重新加载")
            #     try:
            #         if self.has_args():
            #             new_data = (
            #                 await self.func()
            #                 if is_coroutine_callable(self.func)
            #                 else self.func()
            #             )
            #         else:
            #             new_data = (
            #                 await self.func()
            #                 if is_coroutine_callable(self.func)
            #                 else self.func()
            #             )

            #         await self.set_data(new_data)
            #         self.reload_count += 1
            #         logger.info(f"重新加载缓存 {self.name} 完成")
            # return new_data
            # except Exception as e:
            #     logger.error(f"重新加载缓存 {self.name} 失败: {e}")
            #     return None

            # 使用 result_model 进行反序列化
            if self.result_model:
                return self._deserialize_value(data, self.result_model)

            return data
        except Exception as e:
            logger.error(f"获取缓存 {self.name} 失败: {e}")
            return None

    def _serialize_value(self, value: Any) -> Any:
        """序列化值，将数据转换为JSON可序列化的格式

        参数:
            value: 需要序列化的值

        返回:
            JSON可序列化的值
        """
        if value is None:
            return None

        # 处理datetime
        if isinstance(value, datetime):
            return value.isoformat()

        # 处理Tortoise-ORM Model
        if hasattr(value, "_meta") and hasattr(value, "__dict__"):
            result = {}
            for field in value._meta.fields:
                try:
                    field_value = getattr(value, field)
                    # 跳过反向关系字段
                    if isinstance(field_value, list | set) and hasattr(
                        field_value, "_related_name"
                    ):
                        continue
                    # 跳过外键关系字段
                    if hasattr(field_value, "_meta"):
                        field_value = getattr(
                            field_value, value._meta.fields[field].related_name or "id"
                        )
                    result[field] = self._serialize_value(field_value)
                except AttributeError:
                    continue
            return result

        # 处理Pydantic模型
        elif isinstance(value, BaseModel):
            return model_dump(value)
        elif isinstance(value, dict):
            # 处理字典
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple | set):
            # 处理列表、元组、集合
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, int | float | str | bool):
            # 基本类型直接返回
            return value
        else:
            # 其他类型转换为字符串
            return str(value)

    async def set_data(self, value: Any):
        """设置缓存数据"""
        try:
            # 1. 序列化数据
            serialized_value = self._serialize_value(value)
            logger.debug(f"设置缓存 {self.name} 原始数据: {value}")
            logger.debug(f"设置缓存 {self.name} 序列化后数据: {serialized_value}")

            # 2. 删除旧数据
            await self._cache.delete(self.name)
            logger.debug(f"删除缓存 {self.name} 旧数据")

            # 3. 设置新数据
            await self._cache.set(self.name, serialized_value, ttl=self.expire)
            logger.debug(f"设置缓存 {self.name} 新数据完成")

        except Exception as e:
            logger.error(f"设置缓存 {self.name} 失败: {e}")
            raise  # 重新抛出异常，让上层处理

    async def delete_data(self):
        """删除缓存数据"""
        try:
            await self._cache.delete(self.name)
        except Exception as e:
            logger.error(f"删除缓存 {self.name}", e=e)

    async def get(self, key: str, *args, **kwargs) -> Any:
        """获取缓存"""
        if not self.reload_count and not self.lazy_load:
            await self.reload(*args, **kwargs)

        if not self.getter:
            return await self.get_key(key)

        return await self.getter.get(self, key, *args, **kwargs)

    async def get_all(self, *args, **kwargs) -> dict[str, Any]:
        """获取所有缓存数据"""
        if not self.reload_count and not self.lazy_load:
            await self.reload(*args, **kwargs)

        if not self.getter:
            return await self.get_all_data()

        return await self.getter.get_all(self, *args, **kwargs)

    async def update(self, key: str, value: Any = None, *args, **kwargs):
        """更新单个缓存项"""
        if not self.updater:
            logger.warning(f"缓存 {self.name} 未配置更新方法")
            return

        current_data = await self.get_key(key) or {}
        if is_coroutine_callable(self.updater):
            await self.updater(current_data, key, value, *args, **kwargs)
        else:
            self.updater(current_data, key, value, *args, **kwargs)

        await self.set_key(key, current_data)
        logger.debug(f"更新缓存 {self.name}.{key}")

    async def refresh(self, *args, **kwargs):
        """刷新缓存数据"""
        if not self.with_refresh:
            return await self.reload(*args, **kwargs)

        current_data = await self.get_data()
        if current_data:
            if is_coroutine_callable(self.with_refresh):
                await self.with_refresh(current_data, *args, **kwargs)
            else:
                self.with_refresh(current_data, *args, **kwargs)
            await self.set_data(current_data)
            logger.debug(f"刷新缓存 {self.name}")

    async def reload(self, *args, **kwargs):
        """重新加载全部数据"""
        try:
            if self.has_args():
                new_data = (
                    await self.func(*args, **kwargs)
                    if is_coroutine_callable(self.func)
                    else self.func(*args, **kwargs)
                )
            else:
                new_data = (
                    await self.func()
                    if is_coroutine_callable(self.func)
                    else self.func()
                )

            # 如果是字典，则分别存储每个键值对
            if isinstance(new_data, dict):
                for key, value in new_data.items():
                    await self.set_key(key, value)
            else:
                # 如果不是字典，则存储为单个键值对
                await self.set_key("default", new_data)

            self.reload_count += 1
            logger.info(f"重新加载缓存 {self.name} 完成")
        except Exception as e:
            logger.error(f"重新加载缓存 {self.name} 失败: {e}")
            raise

    def has_args(self) -> bool:
        """检查函数是否需要参数"""
        sig = inspect.signature(self.func)
        return any(
            param.kind
            in (
                param.POSITIONAL_OR_KEYWORD,
                param.POSITIONAL_ONLY,
                param.VAR_POSITIONAL,
            )
            for param in sig.parameters.values()
        )

    async def get_key(self, key: str) -> Any:
        """获取缓存中指定键的数据

        参数:
            key: 要获取的键名

        返回:
            键对应的值，如果不存在返回None
        """
        cache_key = self._get_cache_key(key)
        try:
            data = await self._cache.get(cache_key)
            logger.debug(f"获取缓存 {cache_key} 数据: {data}")

            if self.result_model:
                return self._deserialize_value(data, self.result_model)
            return data
        except Exception as e:
            logger.error(f"获取缓存 {cache_key} 失败: {e}")
            return None

    async def get_keys(self, keys: list[str]) -> dict[str, Any]:
        """获取缓存中多个键的数据

        参数:
            keys: 要获取的键名列表

        返回:
            包含所有请求键值的字典，不存在的键值为None
        """
        try:
            data = await self.get_data()
            if isinstance(data, dict):
                return {key: data.get(key) for key in keys}
            return dict.fromkeys(keys)
        except Exception as e:
            logger.error(f"获取缓存 {self.name} 的多个键失败: {e}")
            return dict.fromkeys(keys)

    def _get_cache_key(self, key: str) -> str:
        """获取缓存键名"""
        return f"{self.name}:{key}"

    async def get_all_data(self) -> dict[str, Any]:
        """获取所有缓存数据"""
        try:
            result = {}
            for key in self._keys:
                # 提取原始键名（去掉前缀）
                original_key = key.split(":", 1)[1]
                data = await self._cache.get(key)
                if self.result_model:
                    result[original_key] = self._deserialize_value(
                        data, self.result_model
                    )
                else:
                    result[original_key] = data
            return result
        except Exception as e:
            logger.error(f"获取所有缓存数据失败: {e}")
            return {}

    async def set_key(self, key: str, value: Any):
        """设置指定键的缓存数据"""
        cache_key = self._get_cache_key(key)
        try:
            serialized_value = self._serialize_value(value)
            await self._cache.set(cache_key, serialized_value, ttl=self.expire)
            self._keys.add(cache_key)  # 添加到键列表
            logger.debug(f"设置缓存 {cache_key} 数据完成")
        except Exception as e:
            logger.error(f"设置缓存 {cache_key} 失败: {e}")
            raise

    async def delete_key(self, key: str):
        """删除指定键的缓存数据"""
        cache_key = self._get_cache_key(key)
        try:
            await self._cache.delete(cache_key)
            self._keys.discard(cache_key)  # 从键列表中移除
            logger.debug(f"删除缓存 {cache_key} 完成")
        except Exception as e:
            logger.error(f"删除缓存 {cache_key} 失败: {e}")

    async def clear(self):
        """清除所有缓存数据"""
        try:
            for key in list(self._keys):  # 使用列表复制避免在迭代时修改
                await self._cache.delete(key)
            self._keys.clear()
            logger.debug(f"清除缓存 {self.name} 完成")
        except Exception as e:
            logger.error(f"清除缓存 {self.name} 失败: {e}")


class CacheManager:
    """全局缓存管理器"""

    _cache_instance: BaseCache | AioCache | None = None
    _data: ClassVar[dict[str, CacheData]] = {}

    @property
    def _cache(self) -> BaseCache | AioCache:
        """获取aiocache实例"""
        if self._cache_instance is None:
            if config.redis_host:
                self._cache_instance = AioCache(
                    AioCache.REDIS,
                    serializer=JsonSerializer(),
                    namespace="zhenxun_cache",
                    timeout=30,  # 操作超时时间
                    ttl=config.redis_expire,  # 设置默认过期时间
                    endpoint=config.redis_host,
                    port=config.redis_port,
                    password=config.redis_password,
                )
            else:
                self._cache_instance = AioCache(
                    AioCache.MEMORY,
                    serializer=JsonSerializer(),
                    namespace="zhenxun_cache",
                    timeout=30,  # 操作超时时间
                    ttl=config.redis_expire,  # 设置默认过期时间
                )
            logger.info("初始化缓存完成...", LOG_COMMAND)
        return self._cache_instance

    async def close(self):
        if self._cache_instance:
            await self._cache_instance.close()

    async def verify_connection(self):
        """连接测试"""
        try:
            await self._cache.get("__test__")
        except Exception as e:
            logger.error("连接失败", LOG_COMMAND, e=e)
            raise

    async def init_non_lazy_caches(self):
        """初始化所有非延迟加载的缓存"""
        await self.verify_connection()
        for name, cache in self._data.items():
            cache._cache = self._cache
            if not cache.lazy_load:
                try:
                    await cache.reload()
                    logger.info(f"初始化缓存 {name} 完成")
                except Exception as e:
                    logger.error(f"初始化缓存 {name} 失败: {e}")

    def new(self, name: str, lazy_load: bool = True, expire: int = 600):
        """注册新缓存

        参数:
            name: 缓存名称
            lazy_load: 是否延迟加载，默认为True。为False时会在程序启动时自动加载
            expire: 过期时间（秒）
        """

        def wrapper(func: Callable):
            _name = name.upper()
            if _name in self._data:
                raise DbCacheException(f"缓存 {name} 已存在")

            self._data[_name] = CacheData(
                name=_name,
                func=func,
                expire=expire,
                lazy_load=lazy_load,
                _cache=self._cache,
            )
            return func

        return wrapper

    def listener(self, name: str):
        """创建缓存监听器"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return (
                        await func(*args, **kwargs)
                        if is_coroutine_callable(func)
                        else func(*args, **kwargs)
                    )
                finally:
                    cache = self._data.get(name.upper())
                    if cache and cache.with_refresh:
                        await cache.refresh()
                        logger.debug(f"监听器触发缓存 {name} 刷新")

            return wrapper

        return decorator

    @validate_name
    def updater(self, name: str):
        """设置缓存更新方法"""

        def wrapper(func: Callable):
            self._data[name].updater = func
            return func

        return wrapper

    @validate_name
    def getter(self, name: str, result_model: type):
        """设置缓存获取方法"""

        def wrapper(func: Callable):
            self._data[name].getter = CacheGetter[result_model](get_func=func)
            self._data[name].result_model = result_model
            return func

        return wrapper

    @validate_name
    def with_refresh(self, name: str):
        """设置缓存刷新方法"""

        def wrapper(func: Callable):
            self._data[name].with_refresh = func
            return func

        return wrapper

    async def get_cache_data(self, name: str) -> Any | None:
        """获取缓存数据"""
        cache = await self.get_cache(name.upper())
        return await cache.get_data() if cache else None

    async def get_cache(self, name: str) -> CacheData | None:
        """获取缓存对象"""
        return self._data.get(name.upper())

    async def get(self, name: str, *args, **kwargs) -> Any:
        """获取缓存内容"""
        cache = await self.get_cache(name.upper())
        return await cache.get(*args, **kwargs) if cache else None

    async def update(self, name: str, key: str, value: Any = None, *args, **kwargs):
        """更新缓存项"""
        cache = await self.get_cache(name.upper())
        if cache:
            await cache.update(key, value, *args, **kwargs)

    async def reload(self, name: str, *args, **kwargs):
        """重新加载缓存"""
        cache = await self.get_cache(name.upper())
        if cache:
            await cache.reload(*args, **kwargs)


# 全局缓存管理器实例
CacheRoot = CacheManager()


class Cache(Generic[T]):
    """类型化缓存访问接口"""

    def __init__(self, module: str):
        self.module = module.upper()

    async def get(self, *args, **kwargs) -> T | None:
        """获取缓存"""
        return await CacheRoot.get(self.module, *args, **kwargs)

    async def update(self, key: str, value: Any = None, *args, **kwargs):
        """更新缓存项"""
        await CacheRoot.update(self.module, key, value, *args, **kwargs)

    async def reload(self, *args, **kwargs):
        """重新加载缓存"""
        await CacheRoot.reload(self.module, *args, **kwargs)


@driver.on_shutdown
async def _():
    await CacheRoot.close()
