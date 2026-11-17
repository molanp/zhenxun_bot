from tortoise import fields

from zhenxun.services.db_context import Model


class AIChatConversation(Model):
    id = fields.IntField(pk=True)
    session_id = fields.CharField(255, unique=True)
    scene_id = fields.CharField(255, index=True)
    name = fields.CharField(80, default="默认对话")
    archived = fields.BooleanField(default=False)
    options = fields.JSONField(default=dict)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "ai_chat_conversation"
        table_description = "AI 本地对话及会话级设置"
