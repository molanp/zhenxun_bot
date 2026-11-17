"""DSH 会话选择与工作区展示；不替换桌面 Agent 的模型或执行权限。"""

import re

from zhenxun.services.ai.run.dsh_desktop import DesktopClient, DshUnavailable


class DshControl:
    @staticmethod
    async def client(settings, mode="agent"):
        if settings.transport == "sdk":
            from .dsh_sdk import SdkClient

            return await SdkClient.open(settings, mode)
        return await DesktopClient.open(settings.desktop_url, settings.home)

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
    async def list(cls, settings, owner: str, *, mode: str = "agent") -> list[dict]:
        """列出可供当前所有者接入的桌面会话，排除其他用户已认领会话。"""
        from .dsh import DshSessions

        records = DshSessions.for_settings(settings)._read()
        if mode == "chat":
            items = {}
            for row in [*records["archives"], *records["owners"].values()]:
                if (
                    row["owner"] == owner
                    and row["mode"] == "chat"
                    and not row.get("native")
                ):
                    items[row["remote_id"]] = {
                        **{
                            key: row.get(key)
                            for key in (
                                "remote_id",
                                "model",
                                "cwd",
                                "plan",
                                "preset",
                                "created",
                            )
                        },
                        "native": False,
                        "title": row.get("title") or row["remote_id"],
                        "running": bool(row.get("inflight")),
                        "quarantined": bool(row.get("quarantined")),
                        "subagent": False,
                        "format": "current",
                    }
            return list(items.values())
        claims = {
            row["remote_id"]: row["owner"]
            for row in [*records["archives"], *records["owners"].values()]
        }
        client = await cls.client(settings)
        try:
            rows = (await client.rpc("session/list", {})).get("items", [])
            workspaces = (
                {} if settings.transport == "sdk" else await client.workspace_snapshot()
            )
            archived = set(workspaces.get("archivedSessionIds") or [])
            result = []
            for row in rows:
                session_id = row.get("remote_id", row.get("sessionId", ""))
                if (
                    not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id)
                    or claims.get(session_id, owner) != owner
                ):
                    continue
                item = row if settings.transport == "sdk" else cls.describe(row)
                if item["subagent"] or item["format"] != "current":
                    continue
                item["archived"] = session_id in archived
                result.append(item)
            return result
        finally:
            await client.close()

    @classmethod
    async def choose(
        cls,
        caller,
        settings,
        remote_id: str | None,
        expected_revision: int | None,
        *,
        mode: str = "agent",
    ):
        """为指定超管显式新开或接入原会话；不复制历史、不更改原模型、模式或目录。"""
        from .dsh import DshSessions
        from .session import session_manager

        records = DshSessions.for_settings(settings)
        owner = records.owner_key(caller, mode)
        before = records._read()["owners"].get(owner)
        if before and any(
            active.session_id == before["local_id"] and task and not task.done()
            for active, task in session_manager.runs.values()
        ):
            raise DshUnavailable("当前会话仍有运行，请先停止或等待完成后切换。")
        if before and before.get("inflight"):
            client = await cls.client(settings, mode)
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
                if settings.transport == "sdk" and not state.get("loaded"):
                    if remote_id is not None:
                        raise DshUnavailable(
                            "SDK 原运行停止不可确认；请先显式新建会话，"
                            "旧结果保留待核验。"
                        )
                    records.update(
                        before,
                        unresolved_run=before["inflight"],
                        quarantined="previous_run_unknown",
                    )
                records.update(before, inflight=None)
            finally:
                await client.close()
        revision = before["revision"] if before else 0
        if expected_revision is not None and revision != expected_revision:
            raise DshUnavailable("当前会话已变化，请刷新后重新选择。")
        selected = None
        if remote_id:
            matches = [
                item
                for item in await cls.list(settings, owner, mode=mode)
                if item["remote_id"] == remote_id
            ]
            if len(matches) != 1:
                raise DshUnavailable("会话不存在、由其他用户使用或不能独立接入。")
            selected = matches[0]
            if selected["running"] or selected.get("quarantined"):
                raise DshUnavailable("目标会话正在执行，未接管或中断桌面任务。")
            if mode == "chat" and selected.get("created"):
                client = await cls.client(settings, mode)
                try:
                    state = await client.rpc("session/state", {"sessionId": remote_id})
                    if (
                        state.get("running")
                        or state.get("queue")
                        or state.get("pendingInput")
                    ):
                        raise DshUnavailable("目标聊天仍有执行，不能切换。")
                finally:
                    await client.close()
            if selected["preset"] in {"zhenxun-chat", "zhenxun-agent"}:
                selected["native"] = False
            else:
                selected["native"] = True
        item = records.switch(caller, selected, revision, mode)
        if before:
            session_manager.invalidate(before["local_id"])
        return item

    @classmethod
    async def release(cls, settings, item, *, archive=False):
        """停止前台后解除关联，远端归档与永久删除明确区分。"""
        from zhenxun.builtin_plugins.ai_chat._data_source import AIChatManager

        from .dsh import DshSessions
        from .session import session_manager

        records = DshSessions.for_binding(item)
        session_manager.invalidate(item["local_id"])
        await AIChatManager.stop(item["local_id"])
        records.refresh(item)
        if not item.get("created") and not item.get("inflight"):
            records.reset(item["local_id"])
            return
        client = await cls.client(settings, item["mode"])
        try:
            state = await client.rpc("session/state", {"sessionId": item["remote_id"]})
            if (
                state.get("running")
                or state.get("queue")
                or state.get("pendingInput")
                or (
                    item.get("inflight")
                    and not state.get("loaded", settings.transport == "desktop")
                )
            ):
                raise DshUnavailable("远端运行尚未确认停止，未解除关联。")
            if archive and settings.transport == "desktop":
                await client.rpc(
                    "workspace/archiveSession", {"sessionId": item["remote_id"]}
                )
            records.update(item, inflight=None)
            records.reset(item["local_id"])
        finally:
            await client.close()

    @classmethod
    async def stop_owned(cls, settings, item, run_id: str | None = None) -> dict:
        """依据持久收据查询并停止所属请求，不重新附着转发或重放输入。"""
        import asyncio

        from zhenxun.services.ai.core.exceptions import RunInterrupted

        from .dsh import DshSessions

        records = DshSessions.for_binding(item)
        result = {"state": "unknown", "foreground_only": bool(item.get("native"))}
        client = None

        def receipt(value):
            flight = value.get("inflight") or {}
            return (
                value["local_id"],
                value["remote_id"],
                value["revision"],
                flight.get("run_id"),
                flight.get("request_id"),
            )

        expected = receipt(item)
        try:
            records.refresh(item)
            if receipt(item) != expected:
                return {**result, "reason": "会话或运行已变化，未执行停止"}
            inflight = item.get("inflight") or {}
            if run_id and inflight.get("run_id") != run_id:
                return {**result, "reason": "运行已变化，未停止后续任务"}
            client = await cls.client(settings, item["mode"])
            records.refresh(item)
            if receipt(item) != expected:
                return {**result, "reason": "连接期间运行已变化，未执行停止"}
            if not inflight:
                state = await client.rpc(
                    "session/state", {"sessionId": item["remote_id"]}
                )
                if (
                    state.get("running")
                    or state.get("queue")
                    or state.get("pendingInput")
                ):
                    return {**result, "reason": "存在非本次转发的原生活动，未中断"}
                return {
                    **result,
                    "state": "foreground_idle"
                    if state.get("loaded", settings.transport == "desktop")
                    else "unknown",
                }
            if settings.transport == "sdk":
                result = await client.call(
                    "stop_run",
                    {"run_id": inflight["run_id"], "session_id": item["remote_id"]},
                    timeout=10,
                )
            else:
                result["state"] = await client.stop(
                    item["remote_id"], inflight["request_id"]
                )
                if item.get("native") and result["state"] == "confirmed":
                    result["state"] = "foreground_confirmed"
            if result.get("state") in {
                "confirmed",
                "foreground_confirmed",
                "foreground_idle",
                "target_removed",
            }:
                records.refresh(item)
                if receipt(item) == expected:
                    records.update(item, inflight=None, stop=result["state"])
            return result
        except asyncio.CancelledError:
            raise
        except (Exception, RunInterrupted) as error:
            return {**result, "reason": f"停止结果待核验（{type(error).__name__}）"}
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    from zhenxun.services.log import logger

                    logger.warning("DSH 停止查询连接关闭失败", "AI")

    @classmethod
    async def parameter(cls, settings, item, parameter, value):
        """查询原生选择；SDK 仅在空闲时修改当前会话的模型参数。"""
        from .dsh import DshSessions

        rows = await cls.list(settings, item["owner"])
        selected = next((r for r in rows if r["remote_id"] == item["remote_id"]), None)
        if not selected:
            raise DshUnavailable("会话未找到，请刷新后选择。")
        if not value:
            return str(
                selected.get("preset")
                if parameter == "mode"
                else (selected.get("model") or {}).get("reasoningEffort")
                if parameter == "thinking"
                else selected.get("model")
            )
        if settings.transport != "sdk" or parameter == "mode":
            raise DshUnavailable(
                "此运行时未提供已核验的会话级修改接口；请在 DSH 原生界面处理，"
                "不会改默认值或作为 prompt 发送。"
            )
        if selected.get("running") or not selected.get("loaded"):
            raise DshUnavailable("仅能修改已加载且空闲的 SDK 会话。")
        model = dict(selected.get("model") or {})
        if not model:
            raise DshUnavailable("无法确认原模型选择。")
        if parameter == "model":
            from .dsh_sdk import SdkClient

            SdkClient.credentials(settings.model_copy(update={"sdk_model": value}))
            model["model"] = value
        else:
            if len(value) > 32 or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
                raise DshUnavailable("无效的思考等级。")
            model["reasoningEffort"] = value
        client = await cls.client(settings)
        try:
            await client.call(
                "control",
                {
                    "method": "prepare",
                    "args": {"sessionId": item["remote_id"], "model": model},
                },
            )
            confirmed = await client.call(
                "control",
                {"method": "describe", "args": {"sessionId": item["remote_id"]}},
            )
            if confirmed.get("model") != model:
                raise DshUnavailable("模型选择回读不一致，请重新查询。")
            DshSessions.for_binding(item).update(item, model=model)
            return "已修改当前 SDK 会话；其他会话与桌面默认值不变。"
        finally:
            await client.close()

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
