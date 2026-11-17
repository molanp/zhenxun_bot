from tortoise import fields

from zhenxun.services.db_context import Model


class AIChatScene(Model):
    id = fields.IntField(pk=True)
    scene_id = fields.CharField(255, unique=True)
    current_id = fields.CharField(255, null=True)
    revision = fields.IntField(default=0)
    settings = fields.JSONField(default=dict)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "ai_chat_scene"
        table_description = "AI 聊天场景及当前对话"
