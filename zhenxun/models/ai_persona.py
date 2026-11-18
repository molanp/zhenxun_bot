from collections.abc import Iterable
from typing import ClassVar

from tortoise import fields
from tortoise.backends.base.client import BaseDBAsyncClient

from zhenxun.services.db_context import Model


class AIPersona(Model):
    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    """自增id"""
    name = fields.CharField(64, unique=True, description="人设名称")
    """人设名称"""
    prompt = fields.TextField(description="人设提示词")
    """人设提示词"""
    style = fields.TextField(default="", description="说话风格")
    """说话风格"""
    tone_examples = fields.JSONField(default=list, description="语气示例")
    """语气示例"""
    preset_dialogues = fields.JSONField(default=list, description="开场对话")
    """开场对话"""
    tools = fields.JSONField(null=True, description="工具组白名单，为空不限制")
    """可用的工具组（见 ai_chat 的 TOOL_GROUPS），为空不限制，空列表为不提供工具"""
    knowledge_bases = fields.JSONField(null=True, description="绑定的知识库")
    """人设绑定的知识库 id，群有独立限制时取交集"""
    skills = fields.JSONField(null=True, description="技能白名单，为空不限制")
    """可用的技能名，为空时可用全部已启用技能，空列表为不提供技能"""
    subagents = fields.JSONField(null=True, description="子代理")
    """可转交任务的子代理人设 id，为空不提供子代理"""
    model = fields.CharField(255, null=True, description="指定模型")
    """指定模型，为空使用默认模型"""
    group_ids = fields.JSONField(default=list, description="绑定的群")
    """绑定的群"""
    user_ids = fields.JSONField(null=True, description="绑定的私聊用户")
    """绑定的私聊用户"""
    error_message = fields.TextField(null=True, description="自定义出错提示")
    """对话出错时发送的提示，为空使用默认提示"""
    is_default = fields.BooleanField(default=False, description="是否默认人设")
    """是否默认人设"""
    enabled = fields.BooleanField(default=True, description="是否启用")
    """是否启用"""
    updated_at = fields.DatetimeField(auto_now=True, description="更新时间")
    """更新时间"""

    _enabled: ClassVar[list["AIPersona"] | None] = None

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "ai_persona"
        table_description = "AI 聊天人设"

    @classmethod
    def clear_cache(cls) -> None:
        """清空启用人设的缓存"""
        cls._enabled = None

    async def save(
        self,
        using_db: BaseDBAsyncClient | None = None,
        update_fields: Iterable[str] | None = None,
        force_create: bool = False,
        force_update: bool = False,
    ):
        await super().save(
            using_db=using_db,
            update_fields=update_fields,
            force_create=force_create,
            force_update=force_update,
        )
        type(self).clear_cache()

    async def delete(self, using_db: BaseDBAsyncClient | None = None):
        await super().delete(using_db=using_db)
        type(self).clear_cache()

    @classmethod
    async def get_for_scene(
        cls, group_id: str | None, user_id: str | None = None
    ) -> "AIPersona | None":
        """获取场景使用的人设，群聊取群绑定、私聊取用户绑定的人设，其次为默认人设；
        启用的人设缓存在内存中，修改后清空

        参数:
            group_id: 群号，私聊为空
            user_id: 私聊用户id

        返回:
            AIPersona | None: 人设
        """
        personas = cls._enabled
        if personas is None:
            personas = list(await AIPersona.filter(enabled=True).order_by("id"))
            cls._enabled = personas
        for persona in personas:
            if group_id and group_id in (persona.group_ids or []):
                return persona
            if not group_id and user_id and user_id in (persona.user_ids or []):
                return persona
        return next((p for p in personas if p.is_default), None)

    @classmethod
    async def set_default(cls, persona_id: int) -> None:
        """设为默认人设，同时取消其他人设的默认标记

        参数:
            persona_id: 人设id
        """
        await (
            cls.filter(is_default=True).exclude(id=persona_id).update(is_default=False)
        )
        await cls.filter(id=persona_id).update(is_default=True)
        cls.clear_cache()
