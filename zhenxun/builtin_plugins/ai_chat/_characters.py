"""二次元角色识别与资料检索

本地打标模型、角色库与云端识别来源并发执行，候选按标签合并后加权融合；
模型通过 `recognize_character` 按需识别图片，`character_info` 从 Bangumi 查询角色资料。
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import hashlib
import io
import re
import time
from typing import Any, ClassVar

import httpx
from PIL import Image

from zhenxun.configs.config import Config
from zhenxun.services.ai.core.templates import wrap_untrusted
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ResolvedToolPayload, ToolResult
from zhenxun.services.ai.utils.utils import MediaFetchError
from zhenxun.services.log import logger
from zhenxun.utils.http_utils import get_async_client

from ._char_local import TAGGERS, Candidate, LocalModels, normalize
from ._data_source import MODULE
from ._tools import fetch_image, locate_images, pick_images

TIERS = ("off", "cloud", "lite", "full")
"""识别档位"""
LITE_TAGGER = 0.8
"""低配档打标模型的权重折扣"""
CLOUD_TIMEOUT = 5
"""云端来源单次超时秒数"""
LOCAL_TIMEOUT = 60
"""本地来源超时秒数（含首次加载模型）"""
MIN_SIZE = 256
"""长边小于该像素的图片不识别"""
IMAGE_MAX_BYTES = 10 * 1024 * 1024
"""识别图片的最大字节数"""
RECOGNIZE_CONCURRENCY = 2
"""一次识别多张图片时同时识别的张数"""
BANGUMI = "https://api.bgm.tv"
BANGUMI_TTL = 7 * 86400
"""Bangumi 资料缓存秒数"""
USER_AGENT = "zhenxun_bot/ai_chat (https://github.com/zhenxun-org/zhenxun_bot)"


@dataclass(frozen=True)
class Source:
    title: str
    """名称"""
    weight: float
    """默认权重"""
    enabled: bool = True
    """默认是否启用"""
    local: bool = False
    """是否为本地来源"""


SOURCES: dict[str, Source] = {
    "tagger": Source("打标模型", 1.0, local=True),
    "library": Source("角色库比对", 0.9, local=True),
    "saucenao": Source("SauceNAO 以图搜图", 1.0),
    "dio": Source("DIO 动漫人脸识别", 0.9),
    "animetrace": Source("AnimeTrace", 0.6, enabled=False),
}
"""识别来源与默认设置；lite 档的打标模型（swinv2 量化版）默认权重为 0.8"""


@dataclass
class Fused:
    key: str
    """角色标签或“名字（作品）”"""
    work: str | None
    """作品"""
    score: float
    """融合分"""
    sources: list[tuple[str, float]]
    """(来源, 原始分数)"""


@dataclass
class Recognition:
    candidates: list[Fused]
    """达到最低融合分的前 3 个候选"""
    sources: list[dict[str, Any]] = field(default_factory=list)
    """各来源的结果、耗时与错误"""

    @property
    def ran(self) -> bool:
        """是否至少有一个来源成功执行"""
        return any(s["error"] is None for s in self.sources)


def tier() -> str:
    """当前识别档位"""
    value = str(Config.get_config(MODULE, "CHARACTER_RECOGNITION") or "off")
    return value if value in TIERS else "off"


def settings() -> dict[str, dict[str, Any]]:
    """各来源的启用状态与权重，合并默认值与配置

    返回:
        dict[str, dict[str, Any]]: 来源 -> {enabled, weight, title, local}
    """
    custom = Config.get_config(MODULE, "CHARACTER_SOURCES") or {}
    result = {}
    for key, source in SOURCES.items():
        value = custom.get(key)
        value = value if isinstance(value, dict) else {}
        result[key] = {
            "title": source.title,
            "local": source.local,
            "enabled": bool(value.get("enabled", source.enabled)),
            "weight": float(value.get("weight", source.weight)),
        }
    return result


def _jpeg(data: bytes) -> bytes:
    """压缩为长边不超过 1280 的 JPEG，用于上传云端"""
    image = Image.open(io.BytesIO(data))
    image.seek(0)
    image = image.convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _tag(name: str) -> str:
    """把“hu tao (genshin impact)”写法转为 danbooru 标签"""
    return name.strip().lower().replace(" ", "_")


async def _saucenao(client: httpx.AsyncClient, jpeg: bytes) -> list[Candidate]:
    key = str(Config.get_config(MODULE, "CHARACTER_SAUCENAO_KEY") or "")
    response = await client.post(
        "https://saucenao.com/search.php",
        params={"output_type": 2, "api_key": key, "db": 999, "numres": 6},
        files={"file": ("image.jpg", jpeg, "image/jpeg")},
    )
    body = response.json()
    status = (body.get("header") or {}).get("status", 0)
    if status < 0 or response.status_code >= 400:
        message = (body.get("header") or {}).get("message") or response.status_code
        raise RuntimeError(f"SauceNAO 返回错误：{str(message)[:120]}")
    best: dict[str, Candidate] = {}
    for item in body.get("results") or []:
        similarity = float(item["header"]["similarity"])
        info = item.get("data") or {}
        if similarity < 50 or not info.get("characters"):
            continue
        score = normalize(similarity, 50, 100) * (0.8 if similarity < 60 else 1)
        work = info.get("material") or info.get("source")
        for name in str(info["characters"]).split(","):
            tag = _tag(name)
            if tag and (tag not in best or best[tag].raw < similarity):
                best[tag] = Candidate(
                    tag, "saucenao", round(score, 4), similarity, work
                )
    return list(best.values())


async def _dio(client: httpx.AsyncClient, jpeg: bytes) -> list[Candidate]:
    response = await client.post(
        "https://dio.jite.me/api/recognize",
        files={"file": ("image.jpg", jpeg, "image/jpeg")},
    )
    response.raise_for_status()
    result = []
    for face in response.json().get("faces") or []:
        if face.get("accepted") and face.get("name"):
            score = float(face.get("score") or 0)
            result.append(
                Candidate(
                    f"{face['name']}（{face.get('anime') or '未知作品'}）",
                    "dio",
                    normalize(score, 0.7, 1.0),
                    score,
                    face.get("anime"),
                )
            )
    return result


async def _animetrace(client: httpx.AsyncClient, jpeg: bytes) -> list[Candidate]:
    response = await client.post(
        "https://api.animetrace.com/v1/search",
        data={"is_multi": 1, "ai_detect": 0, "model": "animetrace-yuri-4.2"},
        files={"file": ("image.jpg", jpeg, "image/jpeg")},
    )
    body = response.json()
    if body.get("code") not in (0, 200, 17720):
        raise RuntimeError(f"AnimeTrace 返回错误码 {body.get('code')}")
    result = []
    for box in body.get("data") or []:
        found = box.get("character") or []
        if found and not box.get("not_confident"):
            name, work = found[0].get("character"), found[0].get("work")
            if name:
                result.append(
                    Candidate(
                        f"{name}（{work or '未知作品'}）", "animetrace", 0.8, 1, work
                    )
                )
    return result


CLOUD = {"saucenao": _saucenao, "dio": _dio, "animetrace": _animetrace}
"""云端来源，参数为客户端与压缩后的 JPEG"""


def fuse(
    candidates: list[Candidate], weights: dict[str, float], min_score: float
) -> list[Fused]:
    """按标签合并候选，融合分 = Σ 权重 × 来源分数

    参数:
        candidates: 各来源的候选
        weights: 来源权重
        min_score: 最低融合分

    返回:
        list[Fused]: 达到最低融合分的前 3 个候选，按融合分降序
    """
    library = LocalModels.library()
    best: dict[tuple[str, str], Candidate] = {}
    for c in candidates:
        key = library.alias.get(c.key, c.key) if library else c.key
        if (key, c.source) not in best or best[(key, c.source)].score < c.score:
            best[(key, c.source)] = replace(c, key=key)
    groups: dict[str, Fused] = {}
    for (key, source), c in best.items():
        group = groups.setdefault(key, Fused(key, c.work, 0, []))
        group.score += weights.get(source, 0) * c.score
        group.sources.append((source, c.raw))
        group.work = group.work or c.work
    ranked = sorted(groups.values(), key=lambda f: -f.score)
    for f in ranked:
        f.score = round(f.score, 3)
    return [f for f in ranked if f.score >= min_score][:3]


class CharacterRecognizer:
    """角色识别入口：按档位调度各来源、融合结果并按图片缓存"""

    _cache: ClassVar[OrderedDict[str, Recognition]] = OrderedDict()
    _info: ClassVar[dict[tuple[str, str], tuple[float, str]]] = {}

    @classmethod
    async def _run(
        cls, key: str, call, timeout: float
    ) -> tuple[str, list[Candidate] | dict[str, list[Candidate]], dict[str, Any]]:
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(call, timeout)
            error = None
        except asyncio.TimeoutError:
            result, error = [], f"超过 {timeout:g} 秒未返回"
        except Exception as e:
            result, error = [], str(e) or type(e).__name__
            logger.debug(f"角色识别来源 {key} 失败: {error}", "AI角色识别")
        elapsed = round((time.monotonic() - started) * 1000)
        return key, result, {"ms": elapsed, "error": error}

    @classmethod
    async def recognize(
        cls, data: bytes, level: str | None = None, use_cache: bool = True
    ) -> Recognition:
        """识别图片中的角色，上传云端的 JPEG 在线程中压缩一次后供各云端来源共用

        参数:
            data: 图片
            level: 档位，为空时使用当前档位
            use_cache: 是否使用按图片缓存的结果

        返回:
            Recognition: 融合后的候选与各来源结果
        """
        level = level or tier()
        config = settings()
        min_score = float(Config.get_config(MODULE, "CHARACTER_MIN_SCORE") or 0.5)
        digest = hashlib.md5(data).hexdigest()
        cache_key = f"{digest}:{level}:{min_score}:{sorted(config.items())}"
        if use_cache and (cached := cls._cache.get(cache_key)):
            cls._cache.move_to_end(cache_key)
            return cached
        if level == "off":
            return Recognition([])
        has_key = bool(Config.get_config(MODULE, "CHARACTER_SAUCENAO_KEY"))
        cloud = [
            key
            for key in CLOUD
            if config[key]["enabled"] and (key != "saucenao" or has_key)
        ]
        failed: list[tuple[str, list[Candidate], dict[str, Any]]] = []
        jpeg = b""
        if cloud:
            try:
                jpeg = await asyncio.to_thread(_jpeg, data)
            except Exception as e:
                error = str(e) or type(e).__name__
                failed = [(key, [], {"ms": 0, "error": error}) for key in cloud]
                cloud = []
        async with get_async_client(
            headers={"User-Agent": USER_AGENT}, timeout=CLOUD_TIMEOUT + 2
        ) as client:
            jobs = [
                cls._run(key, CLOUD[key](client, jpeg), CLOUD_TIMEOUT) for key in cloud
            ]
            local = level in TAGGERS and (
                config["tagger"]["enabled"] or config["library"]["enabled"]
            )
            if local:
                jobs.append(
                    cls._run("local", LocalModels.recognize(data, level), LOCAL_TIMEOUT)
                )
            results = failed + list(await asyncio.gather(*jobs))
        candidates: list[Candidate] = []
        sources = []
        for key, result, info in results:
            if isinstance(result, dict):
                groups = result
            elif key == "local":
                groups = {"tagger": result, "library": []}
            else:
                groups = {key: result}
            for name, items in groups.items():
                if not config[name]["enabled"]:
                    continue
                candidates += items
                sources.append(
                    {
                        "key": name,
                        "title": config[name]["title"],
                        **info,
                        "candidates": [
                            {
                                "key": c.key,
                                "score": c.score,
                                "raw": c.raw,
                                "work": c.work,
                            }
                            for c in items
                        ],
                    }
                )
        weights = {k: v["weight"] for k, v in config.items()}
        if level == "lite":
            weights["tagger"] *= LITE_TAGGER
        recognition = Recognition(fuse(candidates, weights, min_score), sources)
        if recognition.ran:
            cls._cache[cache_key] = recognition
            while len(cls._cache) > 512:
                cls._cache.popitem(last=False)
        return recognition

    @staticmethod
    def note(recognition: Recognition) -> str:
        """识别结果整理为说明文字

        参数:
            recognition: 识别结果

        返回:
            str: 说明文字
        """
        if not recognition.candidates:
            return "没有识别到角色。除非非常确定，不要断言图中是某个具体角色。"
        items = []
        for c in recognition.candidates:
            basis = "、".join(
                f"{SOURCES[s].title} {raw:g}" for s, raw in c.sources if s in SOURCES
            )
            work = f"作品 {c.work}；" if c.work and c.work not in c.key else ""
            items.append(f"{c.key}（{work}{basis}）")
        return (
            "角色识别候选（自动识别，可能有误，与画面明显不符时以画面为准；"
            "需要角色资料时可用 character_info 查询）：" + "；".join(items)
        )

    @classmethod
    async def character_info(cls, name: str, work: str = "") -> str | None:
        """从 Bangumi 查询角色资料，按作品过滤同名角色，结果缓存 7 天，
        写入缓存时丢弃过期的资料

        参数:
            name: 角色名
            work: 作品名，可为空

        返回:
            str | None: 资料文本，查不到时为 None
        """
        key = (name.strip(), work.strip())
        if (cached := cls._info.get(key)) and time.time() - cached[0] < BANGUMI_TTL:
            return cached[1] or None
        async with get_async_client(
            base_url=BANGUMI, headers={"User-Agent": USER_AGENT}, timeout=10
        ) as client:
            response = await client.post(
                "/v0/search/characters", params={"limit": 10}, json={"keyword": key[0]}
            )
            response.raise_for_status()
            found = None
            for item in (response.json().get("data") or [])[:5]:
                if not _named(item, key[0]) or (
                    not key[1] and not (item.get("summary") or "").strip()
                ):
                    continue
                r = await client.get(f"/v0/characters/{item['id']}/subjects")
                subjects = r.json() if r.status_code == 200 else []
                works = [s.get("name_cn") or s.get("name") or "" for s in subjects]
                names = [
                    n for s in subjects for n in (s.get("name"), s.get("name_cn")) if n
                ]
                if not key[1] or any(
                    key[1].lower() in n.lower() or n.lower() in key[1].lower()
                    for n in names
                ):
                    found = (item, works)
                    break
        text = _format_character(*found) if found else ""
        cls._info.pop(key, None)
        cls._info[key] = (now := time.time(), text)
        while next(iter(cls._info.values()))[0] < now - BANGUMI_TTL:
            del cls._info[next(iter(cls._info))]
        return text or None


def _named(item: dict[str, Any], name: str) -> bool:
    """Bangumi 角色的名字、中文名或别名与查询名互相包含"""

    def clean(value: str) -> str:
        return re.sub(r"[\s·・=＝]", "", value).lower()

    names = [str(item.get("name") or "")]
    for entry in item.get("infobox") or []:
        if "名" not in str(entry.get("key")):
            continue
        value = entry.get("value")
        if isinstance(value, list):
            names += [str(v.get("v", "")) for v in value if isinstance(v, dict)]
        elif value:
            names.append(str(value))
    query = clean(name)
    return any(
        n and (query in n or (len(n) >= 2 and n in query)) for n in map(clean, names)
    )


def _format_character(item: dict[str, Any], works: list[str]) -> str:
    """整理 Bangumi 角色资料"""
    fields = []
    for entry in item.get("infobox") or []:
        value = entry.get("value")
        if isinstance(value, list):
            value = "、".join(str(v.get("v", "")) for v in value if isinstance(v, dict))
        if value and len(fields) < 10:
            fields.append(f"{entry.get('key')}：{value}")
    lines = [f"名字：{item.get('name')}"]
    if works:
        lines.append(f"作品：{'、'.join(dict.fromkeys(w for w in works if w))[:200]}")
    lines += fields
    if summary := (item.get("summary") or "").strip():
        lines.append(f"简介：{summary[:600]}")
    lines.append(f"来源：https://bgm.tv/character/{item.get('id')}")
    return "\n".join(lines)


def _large(data: bytes) -> bool:
    """图片长边是否不小于 MIN_SIZE"""
    try:
        return max(Image.open(io.BytesIO(data)).size) >= MIN_SIZE
    except Exception:
        return False


class CharacterToolkit(BaseToolkit):
    """AI 聊天的角色识别与资料工具，开启角色识别时提供"""

    default_instructions = (
        "## 角色识别\n别人问图里的动漫、游戏角色是谁时，先用 `recognize_character` "
        "识别，不要凭画风猜；结果只是候选，按把握程度说“可能是”。问到角色的作品、"
        "设定、声优、生日等资料时用 `character_info` 查，不要凭印象回答。"
    )

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        if tier() == "off":
            return ResolvedToolPayload()
        return await super().resolve(context)

    @tool(
        name="recognize_character",
        description=(
            "识别图片里的动漫 / 游戏角色。只在需要知道图中是谁时调用，"
            "message_id 为空时取当前消息（或它引用的消息）里的图片；"
            "indexes 是要识别第几张图片（从 1 开始），为空时识别全部，"
            "多张图片一次调用即可。结果是候选列表，可能有误。"
        ),
    )
    async def recognize_character(
        self,
        context: RunContext,
        message_id: str = "",
        indexes: list[int] | None = None,
    ) -> ToolResult:
        located = await locate_images(context, message_id.strip().lstrip("#"))
        if isinstance(located, str):
            return ToolResult(output=located).as_error()
        picked = pick_images(located[0], indexes)
        if isinstance(picked, str):
            return ToolResult(output=picked).as_error()
        semaphore = asyncio.Semaphore(RECOGNIZE_CONCURRENCY)

        async def run(segment) -> tuple[str, bool]:
            async with semaphore:
                return await self._recognize_one(segment, context.get_bot())

        results = await asyncio.gather(*(run(segment) for _, segment in picked))
        lines = [
            f"第 {index} 张：{text}" if len(picked) > 1 else text
            for (index, _), (text, _) in zip(picked, results)
        ]
        result = ToolResult(output="\n".join(lines))
        return result if any(ok for _, ok in results) else result.as_error(False)

    @staticmethod
    async def _recognize_one(segment, bot=None) -> tuple[str, bool]:
        """识别一张图片

        参数:
            segment: 图片消息段
            bot: Bot，图片地址无效时用于向协议端查询

        返回:
            tuple[str, bool]: 识别说明或失败原因、是否完成识别
        """
        try:
            raw = await fetch_image(bot, segment, IMAGE_MAX_BYTES)
        except MediaFetchError as e:
            logger.debug(f"下载识别图片失败: {e}", "AI角色识别")
            return ("图片太大。" if e.too_large else "图片下载失败。"), False
        if not _large(raw):
            return "图片太小，无法识别。", False
        result = await CharacterRecognizer.recognize(raw)
        if not result.ran:
            return "角色识别暂时不可用。", False
        return CharacterRecognizer.note(result), True

    @tool(
        name="character_info",
        description=(
            "查询动漫 / 游戏角色的资料（来自 Bangumi）。name 用角色的中文名或原名，"
            "work 为作品名，可留空；识别候选中的 danbooru 标签请先换成角色名。"
        ),
    )
    async def character_info(
        self, context: RunContext, name: str, work: str = ""
    ) -> ToolResult:
        try:
            text = await CharacterRecognizer.character_info(name, work)
        except Exception as e:
            logger.warning(f"Bangumi 查询失败: {e}", "AI角色识别")
            return ToolResult(output=f"Bangumi 查询失败：{e}")
        if not text:
            return ToolResult(
                output="Bangumi 没有收录该角色。如果有联网搜索工具可以改用搜索，"
                "否则如实说不知道，不要编造。"
            )
        return ToolResult(output=wrap_untrusted(text, "bangumi"))
