from typing import Any, ClassVar, Generic, TypeVar, cast

from zhenxun.services.cache import Cache, CacheRoot, cache_config
from zhenxun.services.cache.config import COMPOSITE_KEY_SEPARATOR, CacheMode
from zhenxun.services.db_context import Model, with_db_timeout
from zhenxun.services.log import logger

T = TypeVar("T", bound=Model)

cache = CacheRoot.cache_dict("DB_TEST_BAN", 10, int)


class DataAccess(Generic[T]):
    """数据访问层，根据配置决定是否使用缓存

    使用示例:
    ```python
    from zhenxun.services import DataAccess
    from zhenxun.models.plugin_info import PluginInfo

    # 创建数据访问对象
    plugin_dao = DataAccess(PluginInfo)

    # 获取单个数据
    plugin = await plugin_dao.get(module="example_module")

    # 获取所有数据
    all_plugins = await plugin_dao.all()

    # 筛选数据
    enabled_plugins = await plugin_dao.filter(status=True)

    # 创建数据
    new_plugin = await plugin_dao.create(
        module="new_module",
        name="新插件",
        status=True
    )
    ```
    """

    # 添加缓存统计信息
    _cache_stats: ClassVar[dict] = {}

    _NULL_RESULT = "__NULL_RESULT_PLACEHOLDER__"

    def __init__(
        self, model_cls: type[T], key_field: str = "id", cache_type: str | None = None
    ):
        """初始化数据访问对象

        参数:
            model_cls: 模型类
            key_field: 主键字段
        """
        self.model_cls = model_cls
        self.key_field = getattr(model_cls, "cache_key_field", key_field)
        self.cache_type = getattr(model_cls, "cache_type", cache_type)

        if not self.cache_type:
            raise ValueError("缓存类型不能为空")
        self.cache = Cache(self.cache_type)

        # 初始化缓存统计
        if self.cache_type not in self._cache_stats:
            self._cache_stats[self.cache_type] = {
                "hits": 0,  # 缓存命中次数
                "misses": 0,  # 缓存未命中次数
                "sets": 0,  # 缓存设置次数
                "deletes": 0,  # 缓存删除次数
            }

    @classmethod
    def get_cache_stats(cls):
        """获取缓存统计信息"""
        result = []
        for cache_type, stats in cls._cache_stats.items():
            hits = stats["hits"]
            misses = stats["misses"]
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            result.append(
                {
                    "cache_type": cache_type,
                    "hits": hits,
                    "misses": misses,
                    "sets": stats["sets"],
                    "deletes": stats["deletes"],
                    "hit_rate": f"{hit_rate:.2f}%",
                }
            )
        return result

    @classmethod
    def reset_cache_stats(cls):
        """重置缓存统计信息"""
        for stats in cls._cache_stats.values():
            stats["hits"] = 0
            stats["misses"] = 0
            stats["sets"] = 0
            stats["deletes"] = 0

    def _build_cache_key_from_kwargs(self, **kwargs) -> str | None:
        """从关键字参数构建缓存键

        参数:
            **kwargs: 关键字参数

        返回:
            str | None: 缓存键，如果无法构建则返回None
        """
        if isinstance(self.key_field, tuple):
            # 多字段主键
            key_parts = []
            for field in self.key_field:
                if value := kwargs.get(field):
                    key_parts.append(f"{field}:{value}")
            return COMPOSITE_KEY_SEPARATOR.join(key_parts) if key_parts else None
        elif self.key_field in kwargs:
            # 单字段主键
            value = kwargs[self.key_field]
            return f"{self.key_field}:{value}"
        return None

    async def _with_cache(self, db_query_func, *args, **kwargs) -> Any:
        """带缓存的通用获取方法

        参数:
            db_query_func: 数据库查询函数
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            Optional[T]: 查询结果，如果不存在返回None
        """
        # 如果没有缓存类型，直接从数据库获取
        if not self.cache_type or cache_config.cache_mode == CacheMode.NONE:
            logger.debug(f"{self.model_cls.__name__} 直接从数据库获取数据: {kwargs}")
            return await with_db_timeout(
                db_query_func(*args, **kwargs),
                operation=f"{self.model_cls.__name__}.{db_query_func.__name__}",
                source="DataAccess",
            )

        # 尝试从缓存获取
        cache_key = self._build_cache_key_from_kwargs(**kwargs)

        try:
            # 如果成功构建缓存键，尝试从缓存获取
            if cache_key is not None:
                data = await self.cache.get(cache_key)
                logger.debug(
                    f"{self.model_cls.__name__}  key: {cache_key}"
                    f" 从缓存获取到的数据 {type(data)}: {data}"
                )
                if not data:
                    # 缓存未命中
                    self._cache_stats[self.cache_type]["misses"] += 1
                    logger.debug(
                        f"{self.model_cls.__name__} 缓存未命中: {cache_key}"
                        f", kwargs: {kwargs}"
                    )
                elif data == self._NULL_RESULT:
                    # 缓存命中，但结果为空
                    self._cache_stats[self.cache_type]["hits"] += 1
                    logger.debug(
                        f"{self.model_cls.__name__} 从缓存获取数据成功: {cache_key}"
                        f", kwargs: {kwargs}"
                    )
                    return None
                else:
                    # 缓存命中
                    self._cache_stats[self.cache_type]["hits"] += 1
                    logger.debug(
                        f"{self.model_cls.__name__} 从缓存获取数据成功: {cache_key}"
                        f", kwargs: {kwargs}"
                    )
                    return cast(T, data)
        except Exception as e:
            logger.error(f"{self.model_cls.__name__} 从缓存获取数据失败: {kwargs}", e=e)

        # 如果缓存中没有，从数据库获取
        logger.debug(f"{self.model_cls.__name__} 从数据库获取数据: {kwargs}")
        data = await db_query_func(*args, **kwargs)

        # 存入缓存
        try:
            # 生成缓存键
            if cache_key is not None:
                # 存入缓存
                await self.cache.set(cache_key, data or self._NULL_RESULT)
                self._cache_stats[self.cache_type]["sets"] += 1
                logger.debug(f"{self.model_cls.__name__} 数据已存入缓存: {cache_key}")
        except Exception as e:
            logger.error(f"{self.model_cls.__name__} 存入缓存失败，参数: {kwargs}", e=e)
        return data

    async def get_or_none(
        self, allow_not_exist: bool = True, *args, **kwargs
    ) -> T | None:
        """获取单条数据

        参数:
            allow_not_exist: 已弃用
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            Optional[T]: 查询结果，如果不存在返回None
        """
        return await self._with_cache(self.model_cls.get_or_none, *args, **kwargs)

    async def safe_get_or_none(
        self, allow_not_exist: bool = True, *args, **kwargs
    ) -> T | None:
        """安全的获取单条数据

        参数:
            allow_not_exist: 已弃用
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            Optional[T]: 查询结果，如果不存在返回None
        """
        return await self._with_cache(self.model_cls.safe_get_or_none, *args, **kwargs)

    async def get_by_func_or_none(
        self, func, allow_not_exist: bool = True, *args, **kwargs
    ) -> T | None:
        """根据函数获取数据

        参数:
            func: 函数
            allow_not_exist: 已弃用
            *args: 查询参数
            **kwargs: 查询参数
        """
        return await self._with_cache(func, *args, **kwargs)

    async def clear_cache(self, **kwargs) -> bool:
        """只清除缓存，不影响数据库数据

        参数:
            **kwargs: 查询参数，必须包含主键字段

        返回:
            bool: 是否成功清除缓存
        """
        # 如果没有缓存类型，直接返回True
        if not self.cache_type or cache_config.cache_mode == CacheMode.NONE:
            return True

        try:
            # 构建缓存键
            cache_key = self._build_cache_key_from_kwargs(**kwargs)
            if cache_key is None:
                if isinstance(self.key_field, tuple):
                    # 如果是复合键，检查缺少哪些字段
                    missing_fields = [
                        field for field in self.key_field if field not in kwargs
                    ]
                    logger.error(
                        f"清除{self.model_cls.__name__}缓存失败: "
                        f"缺少主键字段 {', '.join(missing_fields)}"
                    )
                else:
                    logger.error(
                        f"清除{self.model_cls.__name__}缓存失败: "
                        f"缺少主键字段 {self.key_field}"
                    )
                return False

            # 删除缓存
            await self.cache.delete(cache_key)
            self._cache_stats[self.cache_type]["deletes"] += 1
            logger.debug(f"已清除{self.model_cls.__name__}缓存: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"清除{self.model_cls.__name__}缓存失败", e=e)
            return False

    async def filter(self, *args, **kwargs) -> list[T]:
        """筛选数据

        参数:
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            List[T]: 查询结果列表
        """
        # 从数据库获取数据
        logger.debug(f"{self.model_cls.__name__} filter: 从数据库查询, 参数: {kwargs}")
        data_list = await self._with_cache(self.model_cls.filter, *args, **kwargs)
        logger.debug(
            f"{self.model_cls.__name__} filter: 查询结果数量: {len(data_list)}"
        )

        return data_list

    async def all(self) -> list[T]:
        """获取所有数据

        返回:
            List[T]: 所有数据列表
        """
        # 直接从数据库获取
        logger.debug(f"{self.model_cls.__name__} all: 从数据库查询所有数据")
        data_list = await self.model_cls.all()
        logger.debug(f"{self.model_cls.__name__} all: 查询结果数量: {len(data_list)}")

        return data_list

    async def count(self, *args, **kwargs) -> int:
        """获取数据数量

        参数:
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            int: 数据数量
        """
        # 直接从数据库获取数量
        return len(await self.filter(*args, **kwargs))

    async def exists(self, *args, **kwargs) -> bool:
        """判断数据是否存在

        参数:
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            bool: 是否存在
        """
        # 直接从数据库判断是否存在
        return bool(await self.filter(*args, **kwargs))

    async def create(self, **kwargs) -> T:
        """创建数据

        参数:
            **kwargs: 创建参数

        返回:
            T: 创建的数据
        """
        # 创建数据
        logger.debug(f"{self.model_cls.__name__} create: 创建数据, 参数: {kwargs}")
        data = await self.model_cls.create(**kwargs)

        # 如果有缓存类型，将数据存入缓存
        if self.cache_type and cache_config.cache_mode != CacheMode.NONE:
            try:
                # 生成缓存键
                cache_key = self._build_cache_key_from_kwargs(**kwargs)
                if cache_key is not None:
                    # 存入缓存
                    await self.cache.set(cache_key, data)
                    self._cache_stats[self.cache_type]["sets"] += 1
                    logger.debug(
                        f"{self.model_cls.__name__} create: "
                        f"新创建的数据已存入缓存: {cache_key}"
                    )
            except Exception as e:
                logger.error(
                    f"{self.model_cls.__name__} create: 存入缓存失败，参数: {kwargs}",
                    e=e,
                )

        return data

    async def update_or_create(
        self, defaults: dict[str, Any] | None = None, **kwargs
    ) -> tuple[T, bool]:
        """更新或创建数据

        参数:
            defaults: 默认值
            **kwargs: 查询参数

        返回:
            tuple[T, bool]: (数据, 是否创建)
        """
        # 更新或创建数据
        data, created = await self.model_cls.update_or_create(
            defaults=defaults, **kwargs
        )

        # 如果有缓存类型，将数据存入缓存
        if self.cache_type and cache_config.cache_mode != CacheMode.NONE:
            try:
                # 生成缓存键
                cache_key = self._build_cache_key_from_kwargs(**kwargs)
                if cache_key is not None:
                    # 存入缓存
                    await self.cache.set(cache_key, data)
                    self._cache_stats[self.cache_type]["sets"] += 1
                    logger.debug(f"更新或创建的数据已存入缓存: {cache_key}")
            except Exception as e:
                logger.error(f"存入缓存失败，参数: {kwargs}", e=e)

        return data, created

    async def delete(self, *args, **kwargs) -> int:
        """删除数据

        参数:
            *args: 查询参数
            **kwargs: 查询参数

        返回:
            int: 删除的数据数量
        """
        logger.debug(f"{self.model_cls.__name__} delete: 删除数据, 参数: {kwargs}")

        # 如果有缓存类型且有key_field参数，先尝试删除缓存
        if self.cache_type and cache_config.cache_mode != CacheMode.NONE:
            try:
                # 尝试构建缓存键
                cache_key = self._build_cache_key_from_kwargs(**kwargs)

                if cache_key is not None:
                    # 如果成功构建缓存键，直接删除缓存
                    await self.cache.delete(cache_key)
                    self._cache_stats[self.cache_type]["deletes"] += 1
                    logger.debug(
                        f"{self.model_cls.__name__} delete: 已删除缓存: {cache_key}"
                    )
            except Exception as e:
                logger.error(f"{self.model_cls.__name__} delete: 删除缓存失败", e=e)

        # 删除数据
        result = await self.model_cls.filter(*args, **kwargs).delete()
        logger.debug(
            f"{self.model_cls.__name__} delete: 已从数据库删除 {result} 条数据"
        )
        return result
