from datetime import datetime, timedelta, timezone
from typing import ClassVar

from tortoise import fields
from tortoise.timezone import localtime

from zhenxun.services.db_context import Model


class AIAgentRun(Model):
    id = fields.BigIntField(pk=True, generated=True, auto_increment=True)
    """自增id"""
    run_id = fields.CharField(32, unique=True, description="运行id")
    """运行id，工具调用记录按它关联"""
    source = fields.CharField(16, description="来源")
    """来源：qq、webui、interject 插话、task 任务、listen 监听、steal 偷表情、
    memory 记忆整理"""
    session_id = fields.CharField(255, description="会话id")
    """会话id"""
    user_id = fields.CharField(255, description="发起人")
    """发起人"""
    group_id = fields.CharField(255, null=True, description="群组id")
    """群组id"""
    persona = fields.CharField(64, null=True, description="人设")
    """人设名称，为空时为默认人设"""
    model = fields.CharField(255, null=True, description="模型")
    """使用的模型"""
    input = fields.TextField(default="", description="输入摘要")
    """输入摘要"""
    output = fields.TextField(default="", description="输出摘要")
    """输出摘要"""
    tools = fields.IntField(default=0, description="工具调用次数")
    """工具调用次数"""
    tokens = fields.IntField(default=0, description="token 用量")
    """token 用量"""
    status = fields.CharField(16, description="结果")
    """结果：ok 完成，error 失败，stopped 中止"""
    error = fields.TextField(default="", description="错误信息")
    """错误信息"""
    duration = fields.IntField(default=0, description="耗时（毫秒）")
    """耗时（毫秒）"""
    details = fields.JSONField(null=True, description="投递与结算详情")

    created_at = fields.DatetimeField(auto_now_add=True, description="开始时间")
    """开始时间"""

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "ai_agent_run"
        table_description = "AI Agent 运行记录"
        indexes: ClassVar = [("created_at",)]

    @classmethod
    async def cleanup(cls, keep_days: int = 30) -> int:
        """删除过期的运行记录

        参数:
            keep_days: 保留天数

        返回:
            int: 删除条数
        """
        cutoff = localtime(datetime.now(timezone.utc) - timedelta(days=keep_days))
        return (
            await cls.filter(created_at__lt=cutoff)
            .exclude(
                status__in=[
                    "queued",
                    "running",
                    "delivering",
                    "settling",
                    "stopping",
                    "unknown",
                    "interrupted",
                ]
            )
            .delete()
        )
