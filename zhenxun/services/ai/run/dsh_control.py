"""DSH 会话选择与工作区展示；不替换桌面 Agent 的模型或执行权限。"""

import re

from zhenxun.services.ai.run.dsh_desktop import DesktopClient, DshUnavailable


class DshControl:
    @staticmethod
    def describe(row: dict) -> dict:
        """从被动投影取得可展示的会话设置，不返回消息或凭据。"""
        values = (row.get("projections") or {}).get("values") or {}
        selection = values.get("modelSelection") or {}
        return {
            "remote_id": row["sessionId"],
            "title": values.get("title") or row["sessionId"],
            "cwd": row.get("cwd"),
            "preset": values.get("agentPreset"),
            "model": selection.get("next") or selection.get("lastUsed"),
            "plan": values.get("plan"),
            "permissions": values.get("permissions"),
            "running": bool(row.get("running")),
            "format": row.get("formatStatus"),
            "subagent": bool(values.get("subagent")),
            "revision": (row.get("projections") or {}).get("asOfSeq"),
        }

    @classmethod
    async def list(cls, settings, owner: str) -> list[dict]:
        """列出可供当前所有者接入的桌面会话，排除其他用户已认领会话。"""
        from .dsh import DshSessions

        records = DshSessions._read()
        claims = {
            row["remote_id"]: row["owner"]
            for row in [*records["archives"], *records["owners"].values()]
        }
        client = await DesktopClient.open(settings.desktop_url, settings.home)
        try:
            rows = (await client.rpc("session/list", {})).get("items", [])
            workspaces = await client.workspace_snapshot()
            archived = set(workspaces.get("archivedSessionIds") or [])
            result = []
            for row in rows:
                session_id = row.get("sessionId", "")
                if (
                    not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id)
                    or claims.get(session_id, owner) != owner
                ):
                    continue
                item = cls.describe(row)
                if item["subagent"] or item["format"] != "current":
                    continue
                item["archived"] = session_id in archived
                result.append(item)
            return result
        finally:
            await client.close()

    @classmethod
    async def choose(
        cls, caller, settings, remote_id: str | None, expected_revision: int | None
    ):
        """为指定超管显式新开或接入原会话；不复制历史、不更改原模型、模式或目录。"""
        from .dsh import DshSessions
        from .session import session_manager

        owner = DshSessions.owner_key(caller, "agent")
        before = DshSessions._read()["owners"].get(owner)
        if before and any(
            active.session_id == before["local_id"] and task and not task.done()
            for active, task in session_manager.runs.values()
        ):
            raise DshUnavailable("当前会话仍有运行，请先停止或等待完成后切换。")
        if before and before.get("inflight"):
            client = await DesktopClient.open(settings.desktop_url, settings.home)
            try:
                state = await client.rpc(
                    "session/state", {"sessionId": before["remote_id"]}
                )
                if (
                    state.get("running")
                    or state.get("queue")
                    or state.get("pendingInput")
                ):
                    raise DshUnavailable("原会话仍有桌面任务，未切换或中断原任务。")
                DshSessions.update(before, inflight=None)
            finally:
                await client.close()
        revision = before["revision"] if before else 0
        if expected_revision is not None and revision != expected_revision:
            raise DshUnavailable("当前会话已变化，请刷新后重新选择。")
        selected = None
        if remote_id:
            matches = [
                item
                for item in await cls.list(settings, owner)
                if item["remote_id"] == remote_id
            ]
            if len(matches) != 1:
                raise DshUnavailable("会话不存在、由其他用户使用或不能独立接入。")
            selected = matches[0]
            if selected["running"]:
                raise DshUnavailable("目标会话正在执行，未接管或中断桌面任务。")
            if selected["preset"] in {"zhenxun-chat", "zhenxun-agent"}:
                selected["native"] = False
            else:
                selected["native"] = True
        item = DshSessions.switch(caller, selected, revision)
        if before:
            session_manager.invalidate(before["local_id"])
        return item

    @staticmethod
    async def chat_workspace(client) -> dict:
        """普通聊天在同一空目录展示为真寻分组，历史仍按会话隔离。"""
        from .dsh import DSH_PATH

        path = DSH_PATH / "chat"
        path.mkdir(parents=True, exist_ok=True)
        result = await client.rpc("workspace/create", {"path": str(path.resolve())})
        workspace = result["workspace"]
        if workspace.get("title") != "真寻":
            changed = await client.rpc(
                "workspace/rename",
                {"workspaceId": workspace["workspaceId"], "title": "真寻"},
            )
            workspace = changed["workspace"]
        return workspace
