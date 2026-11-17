"""内置 AI 聊天的开关、人设、按群设置与 Agent 设置"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zhenxun.services.log import logger

MODULE = "ai_chat"

from ....base_model import Result
from ....utils import authentication

router = APIRouter()


class ChatSwitchRequest(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)
    enabled: bool


@router.get("/chat/switch", dependencies=[authentication()], response_model=Result)
async def chat_switch_state() -> Result:
    from zhenxun.configs.config import Config
    from zhenxun.services.plugin_policy import plugin_policy_service

    state = await plugin_policy_service.global_state()
    return Result.ok(
        {
            "enabled": state["switches"].get(MODULE, False),
            "revision": state["revision"],
            "executor": Config.get_config(MODULE, "AGENT_RUNNER", "local"),
        }
    )


@router.put("/chat/switch", dependencies=[authentication()], response_model=Result)
async def update_chat_switch(payload: ChatSwitchRequest) -> Result:
    from zhenxun.services.plugin_policy import (
        PluginPolicyConflict,
        plugin_policy_service,
    )

    try:
        state = await plugin_policy_service.set_global(
            MODULE,
            payload.enabled,
            expected_revision=payload.expected_revision,
        )
    except PluginPolicyConflict as error:
        raise HTTPException(
            409, detail={"code": error.code, **error.details}
        ) from error
    logger.info(f"WebUI {'启用' if payload.enabled else '停用'}了内置 AI 聊天", "WebUi")
    return Result.ok(
        {
            "enabled": state["switches"].get(MODULE, False),
            "revision": state["revision"],
        },
        info="内置 AI 聊天已启用。" if payload.enabled else "内置 AI 聊天已停用。",
    )


class PersonaPayload(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=20000)
    style: str = Field(default="", max_length=2000)
    tone_examples: list[str] = Field(default_factory=list, max_length=12)
    preset_dialogues: list[str] = Field(default_factory=list, max_length=12)
    tools: list[str] | None = Field(default=None, max_length=50)
    skills: list[str] | None = Field(default=None, max_length=100)
    subagents: list[int] = Field(default_factory=list, max_length=10)
    model: str | None = Field(default=None, max_length=255)
    group_ids: list[str] = Field(default_factory=list, max_length=500)
    user_ids: list[str] = Field(default_factory=list, max_length=500)
    error_message: str = Field(default="", max_length=500)
    enabled: bool = True
    expected_updated_at: str | None = None


class GroupSettingPayload(BaseModel):
    active_probability: float | None = Field(default=None, ge=0, le=1)
    wake_words: list[str] | None = Field(default=None, max_length=20)
    model: str | None = Field(default=None, max_length=255)
    voice_reply: Literal["off", "voice", "always"] | None = None
    agent_runner: Literal["local", "dsh"] | None = None
    knowledge_bases: list[int] | None = Field(default=None, max_length=20)
    persona_id: int | None = None
    enabled: bool = True


def _clean(items: list[str] | None) -> list[str]:
    """去除空白项与重复项，保持顺序

    参数:
        items: 字符串列表

    返回:
        list[str]: 整理后的列表
    """
    return list(dict.fromkeys(item.strip() for item in items or () if item.strip()))


def _persona_item(persona: Any) -> dict[str, Any]:
    """人设表记录转为接口数据

    参数:
        persona: AIPersona

    返回:
        dict[str, Any]: 人设
    """
    return {
        "id": persona.id,
        "name": persona.name,
        "prompt": persona.prompt,
        "style": persona.style,
        "tone_examples": list(persona.tone_examples or []),
        "preset_dialogues": list(persona.preset_dialogues or []),
        "tools": persona.tools,
        "skills": persona.skills,
        "subagents": list(persona.subagents or []),
        "model": persona.model,
        "group_ids": list(persona.group_ids or []),
        "user_ids": list(persona.user_ids or []),
        "error_message": persona.error_message or "",
        "is_default": persona.is_default,
        "enabled": persona.enabled,
        "updated_at": persona.updated_at.isoformat(),
    }


def _persona_error(status: int, code: str, message: str, **details) -> HTTPException:
    return HTTPException(
        status_code=status, detail={"code": code, "message": message, **details}
    )


async def _persona_data(payload: PersonaPayload, persona_id: int | None) -> dict:
    """校验人设并整理为表字段：名称唯一，一个群或私聊用户只能绑定一个启用的人设

    参数:
        payload: 人设
        persona_id: 修改的人设id，新建时为空

    返回:
        dict: 表字段

    异常:
        HTTPException: 名称重复或群、用户已绑定其他人设
    """
    from zhenxun.builtin_plugins.ai_chat._data_source import TOOL_GROUPS
    from zhenxun.models.ai_persona import AIPersona

    tools = None
    if payload.tools is not None:
        tools = [t for t in _clean(payload.tools) if t in TOOL_GROUPS]
    data = {
        "name": payload.name.strip(),
        "prompt": payload.prompt.strip(),
        "style": payload.style.strip(),
        "tone_examples": _clean(payload.tone_examples),
        "preset_dialogues": _clean(payload.preset_dialogues),
        "tools": tools,
        "skills": None if payload.skills is None else _clean(payload.skills),
        "subagents": [i for i in dict.fromkeys(payload.subagents) if i != persona_id]
        or None,
        "model": (payload.model or "").strip() or None,
        "group_ids": _clean(payload.group_ids),
        "user_ids": _clean(payload.user_ids) or None,
        "error_message": payload.error_message.strip() or None,
        "enabled": payload.enabled,
    }
    if not data["name"] or not data["prompt"]:
        raise _persona_error(422, "persona_invalid", "人设名称和提示词不能为空。")
    others = AIPersona.exclude(id=persona_id) if persona_id else AIPersona.all()
    for other in await others:
        if other.name == data["name"]:
            raise _persona_error(422, "persona_name_duplicate", "人设名称已存在。")
        bound = set(other.group_ids or []) & set(data["group_ids"])
        if data["enabled"] and other.enabled and bound:
            raise _persona_error(
                422,
                "persona_group_conflict",
                f"群 {min(bound)} 已绑定人设“{other.name}”。",
                group_id=min(bound),
                persona=other.name,
            )
        users = set(other.user_ids or []) & set(data["user_ids"] or [])
        if data["enabled"] and other.enabled and users:
            raise _persona_error(
                422,
                "persona_user_conflict",
                f"用户 {min(users)} 已绑定人设“{other.name}”。",
                user_id=min(users),
                persona=other.name,
            )
    subagents = data["subagents"] or []
    if len(subagents) != await AIPersona.filter(id__in=subagents).count():
        raise _persona_error(422, "persona_subagent_missing", "子代理人设不存在。")
    return data


@router.get("/personas", dependencies=[authentication()], response_model=Result)
async def list_personas() -> Result:
    from zhenxun.models.ai_persona import AIPersona

    personas = await AIPersona.all().order_by("id")
    return Result.ok([_persona_item(persona) for persona in personas])


@router.post("/personas", dependencies=[authentication()], response_model=Result)
async def create_persona(payload: PersonaPayload) -> Result:
    from zhenxun.models.ai_persona import AIPersona

    data = await _persona_data(payload, None)
    is_default = not await AIPersona.filter(is_default=True).exists()
    persona = await AIPersona.create(**data, is_default=is_default)
    return Result.ok(_persona_item(persona), info="人设已创建。")


@router.put(
    "/personas/{persona_id}", dependencies=[authentication()], response_model=Result
)
async def update_persona(persona_id: int, payload: PersonaPayload) -> Result:
    from zhenxun.models.ai_persona import AIPersona

    persona = await AIPersona.get_or_none(id=persona_id)
    if persona is None:
        raise _persona_error(404, "persona_not_found", "人设不存在。")
    if payload.expected_updated_at not in (None, persona.updated_at.isoformat()):
        raise _persona_error(
            409, "persona_revision_conflict", "人设已被其他操作修改，请重新加载。"
        )
    await persona.update_from_dict(await _persona_data(payload, persona_id)).save()
    return Result.ok(_persona_item(persona), info="人设已保存，下一条消息生效。")


@router.delete(
    "/personas/{persona_id}", dependencies=[authentication()], response_model=Result
)
async def delete_persona(persona_id: int) -> Result:
    from zhenxun.models.ai_persona import AIPersona

    deleted = await AIPersona.filter(id=persona_id).delete()
    AIPersona.clear_cache()
    if not deleted:
        raise _persona_error(404, "persona_not_found", "人设不存在。")
    return Result.ok(info="人设已删除。")


@router.post(
    "/personas/{persona_id}/default",
    dependencies=[authentication()],
    response_model=Result,
)
async def set_default_persona(persona_id: int) -> Result:
    from zhenxun.models.ai_persona import AIPersona

    if not await AIPersona.exists(id=persona_id):
        raise _persona_error(404, "persona_not_found", "人设不存在。")
    await AIPersona.set_default(persona_id)
    return Result.ok(info="已设为默认人设。")


@router.get("/chat/tools", dependencies=[authentication()], response_model=Result)
async def list_chat_tools() -> Result:
    from zhenxun.builtin_plugins.ai_chat._data_source import TOOL_GROUPS

    return Result.ok([{"name": k, "label": v} for k, v in TOOL_GROUPS.items()])


class AgentSettingsPayload(BaseModel):
    enabled: bool


def _run_item(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "source": run.source,
        "session_id": run.session_id,
        "user_id": run.user_id,
        "group_id": run.group_id,
        "persona": run.persona,
        "model": run.model,
        "input": run.input,
        "output": run.output,
        "tools": run.tools,
        "tokens": run.tokens,
        "status": run.status,
        "error": run.error,
        "duration": run.duration,
        "details": run.details or {},
        "created_at": run.created_at.isoformat(),
    }


@router.get("/chat/agent", dependencies=[authentication()], response_model=Result)
async def agent_state(
    limit: int = Query(50, ge=1, le=200),
    source: str | None = Query(None, max_length=16),
    status: str | None = Query(None, max_length=16),
) -> Result:
    from zhenxun.builtin_plugins.ai_chat._agent import agent_enabled
    from zhenxun.models.ai_agent_run import AIAgentRun
    from zhenxun.services.ai.sandbox.manager import SandboxManager

    query = AIAgentRun.all()
    if source:
        query = query.filter(source=source)
    if status:
        query = query.filter(status=status)
    runs = await query.order_by("-id").limit(limit)
    sandbox_ready, sandbox_reason = await SandboxManager.ready()
    return Result.ok(
        {
            "enabled": agent_enabled(),
            "sandbox": {"ready": sandbox_ready, "reason": sandbox_reason},
            "runs": [_run_item(run) for run in runs],
        }
    )


@router.get(
    "/chat/agent/runs/{run_id}", dependencies=[authentication()], response_model=Result
)
async def agent_run_detail(run_id: str) -> Result:
    from zhenxun.models.ai_agent_call import AIAgentCall
    from zhenxun.models.ai_agent_run import AIAgentRun

    run = await AIAgentRun.get_or_none(run_id=run_id)
    if run is None:
        raise HTTPException(404, detail={"code": "run_not_found"})
    calls = await AIAgentCall.filter(run_id=run_id).order_by("id")
    return Result.ok(
        {
            **_run_item(run),
            "calls": [
                {
                    "id": call.id,
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "decision": call.decision,
                    "approver": call.approver,
                    "result": call.result,
                    "is_error": call.is_error,
                    "duration": call.duration,
                    "created_at": call.created_at.isoformat(),
                }
                for call in calls
            ],
        }
    )


@router.put("/chat/agent", dependencies=[authentication()], response_model=Result)
async def update_agent_settings(payload: AgentSettingsPayload) -> Result:
    from zhenxun.builtin_plugins.ai_chat._data_source import MODULE
    from zhenxun.configs.config import Config

    Config.set_config(MODULE, "AGENT_ENABLED", payload.enabled)
    Config.save(save_simple_data=True)
    logger.info(f"WebUI 修改了 Agent 设置: 开启={payload.enabled}", "WebUi")
    return Result.ok(info="Agent 设置已保存。")


@router.get("/chat/groups", dependencies=[authentication()], response_model=Result)
async def list_chat_groups() -> Result:
    from zhenxun.models.ai_chat_group import AIChatGroup
    from zhenxun.models.ai_persona import AIPersona
    from zhenxun.models.group_console import GroupConsole

    settings = {row.group_id: row for row in await AIChatGroup.all()}
    personas = {
        group_id: persona
        for persona in await AIPersona.filter(enabled=True)
        .only("id", "name", "group_ids")
        .order_by("-id")
        for group_id in persona.group_ids or []
    }
    groups = dict(
        await GroupConsole.filter(channel_id__isnull=True).values_list(
            "group_id", "group_name"
        )
    )
    import nonebot

    from zhenxun.builtin_plugins.ai_chat._data_source import parse_session
    from zhenxun.models.ai_chat_scene import AIChatScene
    from zhenxun.services.platform_identity import native_id, scope_owns
    from zhenxun.utils.platform import PlatformUtils

    targets: dict[str, dict[str, dict]] = {}
    for row in await AIChatScene.all():
        bot_id, group_id, _ = parse_session(row.scene_id)
        if group_id and bot_id != "dsh":
            targets.setdefault(group_id, {})[row.scene_id] = {
                "scene_id": row.scene_id,
                "label": f"Bot {bot_id}",
                "overrides": sorted(
                    key for key in (row.settings or {}) if not key.startswith("_")
                ),
            }
    for bot in nonebot.get_bots().values():
        scope = PlatformUtils.get_platform_scope(bot)
        for group_id in set(groups) | set(settings):
            if scope == "qq_api" and not group_id.startswith(
                f"qq_api:{bot.self_id}:group:"
            ):
                continue
            if not scope_owns(group_id, scope=scope) or not native_id(
                group_id, scope, group=True, bot_id=str(bot.self_id)
            ):
                continue
            scene = f"ai_chat:{bot.self_id}:{group_id}"
            target = targets.setdefault(group_id, {}).setdefault(
                scene, {"scene_id": scene, "overrides": []}
            )
            target["label"] = f"{scope} · Bot {bot.self_id}"
    items = []
    for group_id in sorted(set(groups) | set(settings) | set(targets)):
        setting = settings.get(group_id)
        persona = personas.get(group_id)
        items.append(
            {
                "group_id": group_id,
                "group_name": groups.get(group_id, ""),
                "scene_targets": list(targets.get(group_id, {}).values()),
                "active_probability": setting.active_probability if setting else None,
                "wake_words": setting.wake_words if setting else None,
                "model": setting.model if setting else None,
                "voice_reply": setting.voice_reply if setting else None,
                "agent_runner": setting.agent_runner if setting else None,
                "knowledge_bases": setting.knowledge_bases if setting else None,
                "persona_id": persona.id if persona else None,
                "persona": persona.name if persona else None,
                "enabled": not (setting and setting.disabled),
                "proactive": setting.proactive if setting else None,
            }
        )
    return Result.ok(items)


@router.put(
    "/chat/groups/{group_id}", dependencies=[authentication()], response_model=Result
)
async def update_chat_group(group_id: str, payload: GroupSettingPayload) -> Result:
    from zhenxun.models.ai_chat_group import AIChatGroup
    from zhenxun.models.ai_knowledge_base import AIKnowledgeBase
    from zhenxun.models.ai_persona import AIPersona

    if payload.persona_id is not None and not await AIPersona.exists(
        id=payload.persona_id
    ):
        raise _persona_error(404, "persona_not_found", "人设不存在。")
    words = _clean(payload.wake_words) if payload.wake_words is not None else None
    bases = None
    if payload.knowledge_bases is not None:
        bases = list(dict.fromkeys(payload.knowledge_bases))
        if len(bases) != await AIKnowledgeBase.filter(id__in=bases).count():
            raise _persona_error(422, "knowledge_base_missing", "知识库不存在。")
    await AIChatGroup.save_setting(
        group_id,
        active_probability=payload.active_probability,
        wake_words=words,
        model=(payload.model or "").strip() or None,
        voice_reply=payload.voice_reply,
        agent_runner=payload.agent_runner,
        knowledge_bases=bases,
        disabled=None if payload.enabled else True,
    )
    for persona in await AIPersona.all():
        bound = group_id in (persona.group_ids or [])
        if bound != (persona.id == payload.persona_id):
            groups = [g for g in persona.group_ids or [] if g != group_id]
            persona.group_ids = (
                [*groups, group_id] if persona.id == payload.persona_id else groups
            )
            await persona.save(update_fields=["group_ids", "updated_at"])
    return Result.ok(
        info="群基础设置已保存；账号场景的显式覆盖仍优先，可在场景设置恢复继承。"
    )
