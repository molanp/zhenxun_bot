from typing import Any

from zhenxun.services.ai.capabilities.base import AbstractCapability
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.utils.runtime import PermissionUtils


class SuperuserCapability(AbstractCapability):
    """仅当前具有超管授权的调用者可见并执行。"""

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        from zhenxun.services.ai.tools.models import ToolResult

        if not await PermissionUtils.check_superuser(context):
            context.run.operations["permission_reason"] = context.state.get(
                "_permission_reason", "缺少超管授权"
            )
            return ToolResult(output="当前没有超级用户授权，操作未执行。").as_error(
                False
            )
        return await handler(arguments)

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        if await PermissionUtils.check_superuser(context):
            return tool_defs
        return []


class AdminLevelCapability(AbstractCapability):
    """需要群主或管理员权限"""

    def __init__(self, min_level: int):
        """
        初始化群管理权限限制能力。

        参数:
            min_level: 权限等级，大于 0 时要求超级用户或当前群的群主、管理员。
        """
        self.min_level = min_level

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        if await PermissionUtils.check_admin_level(context, self.min_level):
            return tool_defs
        return []
