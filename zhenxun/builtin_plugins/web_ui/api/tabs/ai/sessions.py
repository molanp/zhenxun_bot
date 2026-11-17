"""AI 聊天会话与用量"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from tortoise.expressions import Q
from tortoise.functions import Count, Max, Sum

from zhenxun.services.log import logger

from ....base_model import Result
from ....utils import authentication

router = APIRouter()

from .console import CONSOLE_PREFIX

SESSION_PREFIX = "ai_chat:"


class SceneSettingsUpdate(BaseModel):
    expected_revision: int = Field(ge=0)
    overrides: dict[str, Any]


def _require_scene(scene_id: str) -> None:
    from zhenxun.builtin_plugins.ai_chat._data_source import parse_session
    from zhenxun.builtin_plugins.ai_chat._sessions import scene_key

    if (
        not scene_id.startswith(SESSION_PREFIX)
        or len(scene_id) > 255
        or scene_key(scene_id) != scene_id
    ):
        raise HTTPException(422, detail={"code": "scene_invalid"})
    parts = scene_id.split(":", 2)
    if len(parts) != 3 or parts[1] == "dsh" or parts[2].startswith("task:"):
        raise HTTPException(422, detail={"code": "scene_invalid"})
    bot_id, group_id, user_id = parse_session(scene_id)
    if not bot_id or not (group_id or user_id):
        raise HTTPException(422, detail={"code": "scene_invalid"})


@router.get(
    "/chat/scenes/{scene_id}/settings",
    dependencies=[authentication()],
    response_model=Result,
)
async def scene_settings(scene_id: str) -> Result:
    """返回指定账号场景的有效设置、继承来源和按任务筛选的模型。"""
    from zhenxun.builtin_plugins.ai_chat._sessions import Conversations

    _require_scene(scene_id)
    state = await Conversations.resolve(scene_id, inspect_policy=True)
    state["models"] = {
        task: Conversations.models(task) for task in ("chat", "tts", "stt")
    }
    return Result.ok(state)


@router.put(
    "/chat/scenes/{scene_id}/settings",
    dependencies=[authentication()],
    response_model=Result,
)
async def update_scene_settings(scene_id: str, payload: SceneSettingsUpdate) -> Result:
    """比较场景修订后保存显式覆盖，null 恢复继承。"""
    from zhenxun.builtin_plugins.ai_chat._sessions import (
        ConversationConflict,
        Conversations,
    )
    from zhenxun.services.ai.core.exceptions import LLMException

    _require_scene(scene_id)
    try:
        await Conversations.change(
            scene_id,
            "configure",
            settings=payload.overrides,
            expected_revision=payload.expected_revision,
        )
    except ConversationConflict as error:
        raise HTTPException(409, detail={"message": str(error)}) from error
    except (ValueError, LLMException) as error:
        raise HTTPException(422, detail={"message": str(error)}) from error
    result = await scene_settings(scene_id)
    result.info = (
        "已保存此账号场景设置；切换历史不会改变配置，目标场景的 AI 计划共同使用。"
    )
    return result


def _parse_session(session_id: str) -> dict[str, Any]:
    """拆分聊天会话id，群聊会话没有用户（旧版按群和用户区分的会话仍带用户）

    参数:
        session_id: 会话id

    返回:
        dict[str, Any]: Bot id、群号（私聊为空）、用户id
    """
    from zhenxun.builtin_plugins.ai_chat._data_source import parse_session

    if session_id.startswith("ai_chat:dsh:"):
        from zhenxun.services.ai.run.dsh import DshSessions, SdkSessions

        records = [source._read() for source in (DshSessions, SdkSessions)]
        item = next(
            (
                item
                for data in records
                for item in [*data["owners"].values(), *data["archives"]]
                if item["local_id"] == session_id
            ),
            {},
        )
        return {
            **item.get("identity", {}),
            "dsh": {
                key: item.get(key)
                for key in (
                    "remote_id",
                    "mode",
                    "revision",
                    "quarantined",
                    "archive_reason",
                )
            },
        }

    bot_id, group_id, user_id = parse_session(session_id)
    return {"bot_id": bot_id, "group_id": group_id, "user_id": user_id or ""}


def _message_text(content: Any) -> str:
    """把短期记忆中的消息内容转为可读文本，媒体用占位表示，略去思考过程

    参数:
        content: 消息内容

    返回:
        str: 文本
    """
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", ""))
        if part.get("type") == "text"
        else f"[{part.get('type')}]"
        for part in content or ()
        if isinstance(part, dict) and part.get("type") != "thought"
    )


def _require_session(session_id: str) -> str:
    if not session_id.startswith((SESSION_PREFIX, CONSOLE_PREFIX)):
        raise HTTPException(422, detail={"code": "session_invalid"})
    return session_id


@router.get("/chat/sessions", dependencies=[authentication()], response_model=Result)
async def list_sessions(
    keyword: str = Query(default="", max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Result:
    from zhenxun.builtin_plugins.ai_chat._sessions import scene_key
    from zhenxun.models.ai_chat_conversation import AIChatConversation
    from zhenxun.models.ai_chat_message import AIChatMessage
    from zhenxun.models.ai_chat_scene import AIChatScene

    metadata = AIChatConversation.all()
    term = keyword.strip()
    if term:
        metadata = metadata.filter(
            Q(session_id__contains=term) | Q(name__contains=term)
        )
    conversations = await metadata
    query = AIChatMessage.filter(session_id__startswith=SESSION_PREFIX)
    if term:
        query = query.filter(
            Q(session_id__contains=term)
            | Q(session_id__in=[item.session_id for item in conversations])
        )
    rows = (
        await query.annotate(messages=Count("id"), last_at=Max("created_at"))
        .group_by("session_id")
        .order_by("-last_at")
        .values("session_id", "messages", "last_at")
    )
    items = {
        row["session_id"]: {
            "session_id": row["session_id"],
            **_parse_session(row["session_id"]),
            "messages": row["messages"],
            "last_at": row["last_at"].isoformat() if row["last_at"] else None,
        }
        for row in rows
    }
    scenes = {
        row.scene_id: row
        for row in await AIChatScene.filter(
            scene_id__in={item.scene_id for item in conversations}
        )
    }
    for item in conversations:
        row = items.setdefault(
            item.session_id,
            {
                "session_id": item.session_id,
                **_parse_session(item.session_id),
                "messages": 0,
                "last_at": None,
            },
        )
        scene = scenes.get(item.scene_id)
        row.update(
            name=item.name,
            scene_id=item.scene_id,
            archived=item.archived,
            active=bool(scene and scene.current_id == item.session_id),
            revision=scene.revision if scene else 0,
            settings={
                key: value
                for key, value in (scene.settings or {}).items()
                if not key.startswith("_")
            }
            if scene
            else {},
            options={},
            created_at=item.created_at.isoformat(),
            local=True,
        )
    for row in items.values():
        if "local" not in row:
            row.update(
                local=not row["session_id"].startswith("ai_chat:dsh:"),
                scene_id=scene_key(row["session_id"]),
                revision=0,
                name="历史对话",
                active=True,
                archived=False,
            )
    ordered = sorted(
        items.values(),
        key=lambda row: row.get("last_at") or row.get("created_at") or "",
        reverse=True,
    )
    start = (page - 1) * page_size
    return Result.ok(
        {"total": len(ordered), "items": ordered[start : start + page_size]}
    )


class SessionAction(BaseModel):
    scene_id: str = Field(min_length=1, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)
    action: Literal[
        "new", "switch", "rename", "archive", "unarchive", "delete", "reset"
    ]
    expected_revision: int = Field(ge=0)
    name: str | None = Field(default=None, min_length=1, max_length=80)


@router.post("/chat/sessions", dependencies=[authentication()], response_model=Result)
async def change_session(payload: SessionAction) -> Result:
    """比较场景修订后管理本地对话，沿用管理面认证。"""
    from zhenxun.builtin_plugins.ai_chat._sessions import (
        ConversationConflict,
        Conversations,
        scene_key,
    )

    scene = _require_session(payload.scene_id)
    if (
        not scene.startswith(SESSION_PREFIX)
        or scene.startswith("ai_chat:dsh:")
        or scene_key(scene) != scene
    ):
        raise HTTPException(422, detail={"message": "此入口只管理本地聊天场景。"})
    try:
        result = await Conversations.change(
            scene,
            payload.action,
            target=payload.session_id,
            name=payload.name,
            expected_revision=payload.expected_revision,
        )
    except ConversationConflict as error:
        raise HTTPException(409, detail={"message": str(error)}) from error
    except ValueError as error:
        raise HTTPException(422, detail={"message": str(error)}) from error
    return Result.ok(result, info="对话操作已完成，长期记忆与用量记录保留。")


@router.get(
    "/chat/sessions/messages", dependencies=[authentication()], response_model=Result
)
async def session_messages(
    session_id: str = Query(min_length=1, max_length=255),
    raw: bool = False,
) -> Result:
    from zhenxun.models.ai_chat_message import AIChatMessage

    rows = await AIChatMessage.filter(session_id=_require_session(session_id)).order_by(
        "id"
    )
    return Result.ok(
        {
            "session_id": session_id,
            **_parse_session(session_id),
            "messages": [
                {
                    "role": row.role,
                    "text": _message_text(row.content),
                    **(
                        {"content": row.content, "metadata": row.metadata}
                        if raw
                        else {}
                    ),
                    "is_summary": row.is_summary,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ],
        }
    )


@router.delete("/chat/sessions", dependencies=[authentication()], response_model=Result)
async def reset_session(
    session_id: str = Query(min_length=1, max_length=255),
    action: Literal["reset", "new", "archive"] = "reset",
    expected_revision: int | None = None,
) -> Result:
    from zhenxun.builtin_plugins.ai_chat._data_source import AIChatManager

    _require_session(session_id)
    if session_id.startswith("ai_chat:dsh:"):
        from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
        from zhenxun.services.ai.core.exceptions import RunInterrupted
        from zhenxun.services.ai.run.dsh import DshSessions, SdkSessions
        from zhenxun.services.ai.run.dsh_desktop import DesktopClient
        from zhenxun.services.ai.run.session import session_manager

        item = next(
            (
                i
                for source in (DshSessions, SdkSessions)
                for i in source._read()["owners"].values()
                if i["local_id"] == session_id
            ),
            None,
        )
        if not item or (
            expected_revision is not None and item["revision"] != expected_revision
        ):
            raise HTTPException(409, detail={"code": "dsh_session_changed"})
        if item.get("transport") == "sdk":
            from zhenxun.services.ai.run.dsh_control import DshControl
            from zhenxun.services.ai.run.dsh_desktop import DshUnavailable

            settings = dsh_settings()
            if settings.transport != "sdk":
                raise HTTPException(
                    409, detail={"message": "请先选择 SDK 模式再操作此会话。"}
                )
            try:
                await DshControl.release(settings, item, archive=action == "archive")
                if action == "reset":
                    await AIChatManager.reset(session_id)
            except DshUnavailable as error:
                raise HTTPException(409, detail={"message": str(error)}) from error
            return Result.ok(
                {"archived": False, "history_deleted": False},
                info="已解除 SDK 关联；远端历史保留，未执行远端归档或删除。",
            )
        session_manager.invalidate(session_id)
        await AIChatManager.stop(session_id)
        DshSessions.refresh(item)
        if item.get("inflight"):
            raise HTTPException(409, detail={"code": "dsh_stop_unconfirmed"})
        archived = False
        client = None
        try:
            settings = dsh_settings()
            client = await DesktopClient.open(settings.desktop_url, settings.home)
            if item["created"] and (not item.get("native") or action == "archive"):
                await client.rpc(
                    "workspace/archiveSession", {"sessionId": item["remote_id"]}
                )
                archived = True
        except (Exception, RunInterrupted):
            pass
        finally:
            if client:
                await client.close()
        if action == "reset":
            await AIChatManager.reset(session_id)
        else:
            DshSessions.reset(session_id)
        return Result.ok(
            {"archived": archived, "history_deleted": False},
            info="已解除关联；原生 DSH 会话、历史和设置均保留。"
            if item.get("native") and action != "archive"
            else "已解除会话映射；桌面历史已归档但未删除。"
            if archived
            else "已解除会话映射；桌面归档暂未确认，原历史仍保留。",
        )
    if session_id.startswith(CONSOLE_PREFIX):
        if action != "reset":
            raise HTTPException(422, detail={"code": "console_reset_only"})
        await AIChatManager.reset(session_id)
    else:
        from zhenxun.builtin_plugins.ai_chat._sessions import Conversations, scene_key

        try:
            await Conversations.change(
                scene_key(session_id),
                action,
                target=session_id,
                expected_revision=expected_revision,
            )
        except ValueError as error:
            raise HTTPException(409, detail={"message": str(error)}) from error
    logger.info(f"WebUI 重置了 AI 聊天会话 {session_id}", "WebUi")
    return Result.ok(info="会话记忆已清空。")


@router.get("/chat/usage", dependencies=[authentication()], response_model=Result)
async def chat_usage(limit: int = Query(default=20, ge=1, le=100)) -> Result:
    from zhenxun.models.ai_token_usage import AITokenUsage

    today = AITokenUsage.filter(day=date.today())
    rows = await today.order_by("-tokens").limit(limit).values("user_id", "tokens")
    total = (await today.annotate(total=Sum("tokens")).values("total"))[0]["total"]
    return Result.ok(
        {
            "day": date.today().isoformat(),
            "total": int(total or 0),
            "users": await today.count(),
            "top": rows,
        }
    )
