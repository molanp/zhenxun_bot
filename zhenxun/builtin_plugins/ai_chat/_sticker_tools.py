"""AI 聊天的表情包工具：收藏、搜索、修改、合并、删除、发送"""

from nonebot_plugin_alconna import Segment
from pydantic import BaseModel, Field

from zhenxun.configs.config import Config
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import (
    ResolvedToolPayload,
    ToolOptions,
    ToolResult,
)
from zhenxun.services.ai.utils.utils import MediaFetchError
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_validate

from ._data_source import MODULE
from ._send import Scene, SendQuota
from ._stickers import STICKER_MAX_BYTES, Stickers
from ._tools import (
    GroupHistory,
    fetch_image,
    image_source,
    locate_images,
    pick_images,
)

MANAGE_TOOLS = {"delete_stickers", "merge_stickers", "update_sticker"}
"""只在超级用户的对话中提供的表情整理工具"""


class StickerNote(BaseModel):
    """收藏一张表情时的说明"""

    index: int = Field(
        default=0, description="第几张图片，从 1 开始，为 0 时按列表顺序"
    )
    remark: str = Field(default="", description="几个字写表情的意思")
    tags: list[str] | None = Field(default=None, description="关键词")
    usage: str = Field(default="", description="适合在什么场合用")


class StickerToolkit(BaseToolkit):
    """AI 聊天的表情包工具：收藏、搜索、发送，超级用户还能修改说明、删除与合并"""

    manage_instructions = (
        "## 表情库整理\n当前对话者是超级用户，可以整理表情库：说明不准的用 "
        "`update_sticker` 修改，重复的用 `merge_stickers` 合并，不好用的用 "
        "`delete_stickers` 删除，删除前先用 `find_stickers` 确认 id。"
    )

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        payload = await super().resolve(context)
        hidden = set()
        if not Config.get_config(MODULE, "SEND_TOOLS", True):
            hidden.add("send_sticker")
        if not Config.get_config(MODULE, "STICKER_COLLECT", True):
            hidden.add("collect_sticker")
        manage = bool(context) and await self._manager(context)
        if not manage:
            hidden |= MANAGE_TOOLS
        top = await Stickers.candidates()
        if context:
            context.state.setdefault("_sticker_candidates", {}).update(
                {s.id: s.md5 for s in top}
            )
        if not top and "collect_sticker" in hidden:
            return ResolvedToolPayload()
        frequency = Config.get_config(MODULE, "STICKER_FREQUENCY", "balanced")
        hints = {
            "high": (
                "聊天场景合适时积极使用贴切的表情接梗、表达情绪，"
                "但不要机械地每次都发。"
            ),
            "balanced": (
                "自然穿插表情；通常用文字即可，表情能增添表达时再发，不要每次都用。"
            ),
            "low": "大多数回复只用文字；仅在表情特别贴切或用户明确要求时发送。",
        }
        payload.tools = [t for t in payload.tools if t.name not in hidden]
        if top and "send_sticker" not in hidden:
            payload.reminders.append(
                "本轮候选表情：" + "；".join(Stickers.describe(s) for s in top)
            )
        if "send_sticker" not in hidden:
            hint = hints.get(frequency, hints["balanced"])
            payload.instruction_builders.append(
                lambda names: [
                    "## 表情包\n回复末尾可以写一个 [表情:#id] 发送表情，id 是候选表情"
                    "或 `find_stickers` 结果里 # 后的数字。聊天记录里的 [表情: …]、"
                    "[表情包: …] 是别人发的表情的文字描述，不要照抄进回复。" + hint
                ]
                if "send_sticker" in names
                else []
            )
        if manage:
            payload.instruction_builders.append(
                lambda names: [self.manage_instructions]
                if MANAGE_TOOLS <= names
                else []
            )
        return payload

    @tool(
        name="collect_sticker",
        description=(
            "把图片或表情收藏进你的表情库，一次可收藏同一条消息里的多张，"
            "不要为每张图片分别调用。message_id 为空时取当前消息（或它引用的消息）"
            "里的图片；images 每项对应一张图片：index 是第几张（从 1 开始），"
            "remark 用几个字写表情的意思，tags 是关键词，usage 写适合在什么场合用。"
        ),
    )
    async def collect_sticker(
        self,
        context: RunContext,
        images: list[StickerNote],
        message_id: str = "",
    ) -> ToolResult:
        scene = Scene.of(context)
        if scene is None:
            return ToolResult(output="当前不在会话中。").as_error()
        if not Config.get_config(MODULE, "STICKER_COLLECT", True):
            return ToolResult(output="表情收藏已关闭。").as_error(False)
        located = await locate_images(context, message_id.strip().lstrip("#"))
        if isinstance(located, str):
            return ToolResult(output=located).as_error()
        segments, owner = located
        notes = []
        for i, item in enumerate(images or [StickerNote()], 1):
            note = model_validate(StickerNote, item)
            note.index = note.index or i
            notes.append(note)
        picked = pick_images(segments, [note.index for note in notes])
        if isinstance(picked, str):
            return ToolResult(output=picked).as_error()
        by_index = {note.index: note for note in notes}
        lines, collected, limited = [], 0, ""
        for index, segment in picked:
            line, ok, limited = await self._collect_one(
                scene, segment, by_index[index], owner, len(picked) > 1
            )
            if limited:
                lines.append(limited)
                break
            lines.append(f"第 {index} 张：{line}" if len(picked) > 1 else line)
            collected += ok
        result = ToolResult(output="\n".join(lines))
        return result if collected or not limited else result.as_error(False)

    @staticmethod
    async def _collect_one(
        scene: Scene, segment: Segment, note: StickerNote, owner: str | None, many: bool
    ) -> tuple[str, bool, str]:
        """收藏一张图片

        参数:
            scene: 当前会话
            segment: 图片消息段
            note: 备注、标签与适用场合
            owner: 原图发送者
            many: 是否一次收藏多张

        返回:
            tuple[str, bool, str]: 结果说明、是否新收藏、达到收藏上限时的原因
        """
        raw, url, summary = image_source(segment)
        if raw is None and not url:
            return "读取不到这张图片。", False, ""
        key = ("collect", scene.key[1])
        if reason := SendQuota.take(
            key,
            1,
            int(Config.get_config(MODULE, "STICKER_PER_MINUTE") or 0),
            int(Config.get_config(MODULE, "STICKER_PER_HOUR") or 0),
            unit="张",
        ):
            return "", False, f"剩下的图片没有收藏：{reason}" if many else reason
        if raw is None:
            try:
                raw = await fetch_image(scene.bot, segment, STICKER_MAX_BYTES)
            except MediaFetchError as e:
                SendQuota.refund(key, 1)
                logger.debug(f"下载表情失败: {e}", MODULE)
                reason = "图片超过 2MB。" if e.too_large else "图片下载失败。"
                return reason, False, ""
        try:
            sticker, created = await Stickers.add(
                raw,
                note.remark or summary,
                note.tags,
                note.usage,
                group_id=scene.group_id,
                user_id=owner,
            )
        except ValueError as e:
            SendQuota.refund(key, 1)
            return str(e), False, ""
        if not created:
            SendQuota.refund(key, 1)
            return f"已经收藏过了：{Stickers.describe(sticker)}", False, ""
        await Stickers.push_qq(scene.bot, raw, sticker.remark)
        return f"已收藏：{Stickers.describe(sticker)}", True, ""

    @tool(
        name="find_stickers",
        rules=ToolOptions(metadata={"read_only": True}),
        description="按关键词搜索表情库（备注、标签、适用场合），为空时列出最常用的。",
    )
    async def find_stickers(self, context: RunContext, query: str = "") -> ToolResult:
        stickers = await Stickers.search(query)
        context.state.setdefault("_sticker_candidates", {}).update(
            {s.id: s.md5 for s in stickers}
        )
        if not stickers:
            return ToolResult(output="没有找到合适的表情。")
        return ToolResult(output="\n".join(Stickers.describe(s) for s in stickers))

    @tool(
        name="update_sticker",
        description="修改表情的备注、标签或适用场合，不填的不修改。",
    )
    async def update_sticker(
        self,
        context: RunContext,
        sticker_id: int,
        remark: str | None = None,
        tags: list[str] | None = None,
        usage: str | None = None,
    ) -> ToolResult:
        if not await self._manager(context):
            return ToolResult(output="只有超级用户可以修改全局表情说明。").as_error(
                False
            )
        sticker = await Stickers.update(sticker_id, remark, tags, usage)
        if sticker is None:
            return ToolResult(output="没有这个表情。").as_error()
        return ToolResult(output=f"已修改：{Stickers.describe(sticker)}")

    @tool(
        name="send_sticker",
        description=(
            "立即向当前会话发送表情库中的一个表情，reply_to 可引用本群一条消息；"
            "不需要引用时在回复末尾写 [表情:#id] 即可。"
        ),
    )
    async def send_sticker(
        self, context: RunContext, sticker_id: int, reply_to: str = ""
    ) -> ToolResult:
        scene = Scene.of(context)
        if scene is None:
            return ToolResult(output="当前不在会话中。").as_error()
        reply_to = reply_to.strip().lstrip("#")
        if reply_to and (
            not scene.group_id
            or await GroupHistory.message(scene.bot, scene.group_id, reply_to) is None
        ):
            return ToolResult(output="只能引用当前群里的消息。").as_error()
        expected = context.state.get("_sticker_candidates", {}).get(sticker_id)
        if expected is None:
            return ToolResult(output="请先检索本轮可用的表情。").as_error(False)
        if reason := await Stickers.send(
            scene, sticker_id, reply_to, expected_md5=expected
        ):
            return ToolResult(output=reason).as_error(reason == "没有这个表情。")
        return ToolResult(output="已发送。")

    @tool(
        name="delete_stickers",
        description=(
            "从表情库删除表情（只在超级用户要求或确认时使用）。"
            "block 为 true 时同时设为不再收集，之后不会再收藏这些图片。"
        ),
    )
    async def delete_stickers(
        self, context: RunContext, sticker_ids: list[int], block: bool = False
    ) -> ToolResult:
        if not await self._manager(context):
            return ToolResult(output="只有超级用户可以删除表情。").as_error(False)
        count = await Stickers.remove_many(sticker_ids[:50], block)
        if not count:
            return ToolResult(output="没有这些表情。").as_error()
        return ToolResult(output=f"已删除 {count} 个表情。")

    @tool(
        name="merge_stickers",
        description=(
            "把重复的表情合并到 keep_id：标签取并集、使用次数相加，"
            "other_ids 中的表情删除并设为不再收集。"
        ),
    )
    async def merge_stickers(
        self, context: RunContext, keep_id: int, other_ids: list[int]
    ) -> ToolResult:
        if not await self._manager(context):
            return ToolResult(output="只有超级用户可以合并表情。").as_error(False)
        sticker = await Stickers.merge(keep_id, other_ids[:50])
        if sticker is None:
            return ToolResult(output="没有要保留的这个表情。").as_error()
        return ToolResult(output=f"已合并：{Stickers.describe(sticker)}")

    @staticmethod
    async def _manager(context: RunContext) -> bool:
        """当前对话者是否为超级用户

        参数:
            context: 运行上下文

        返回:
            bool: 是否为超级用户
        """
        from zhenxun.services.ai.utils.runtime import PermissionUtils

        from ._agent import source_policy

        return bool(
            source_policy(context)["privileged"]
            and await PermissionUtils.check_superuser(context)
        )
