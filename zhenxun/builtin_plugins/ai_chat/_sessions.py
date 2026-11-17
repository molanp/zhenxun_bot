"""本地多对话、场景设置与修订控制，长期记忆仍归属原场景。"""

import asyncio
from contextlib import asynccontextmanager
import re
from typing import ClassVar
import uuid
import weakref

from tortoise.transactions import in_transaction

from zhenxun.models.ai_chat_conversation import AIChatConversation
from zhenxun.models.ai_chat_scene import AIChatScene
from zhenxun.models.ai_persona import AIPersona


def scene_key(session_id: str) -> str:
    """读取对话所属的原始场景标识，不改变平台身份。"""
    return re.sub(r":conv:[0-9a-f]{32}$", "", session_id)


class ConversationConflict(ValueError):
    pass


class Conversations:
    _locks: ClassVar[weakref.WeakValueDictionary[str, asyncio.Lock]] = (
        weakref.WeakValueDictionary()
    )
    _changing: ClassVar[set[str]] = set()

    @staticmethod
    async def disabled(scene: str, setting=None) -> bool:
        """场景是否停用 AI，不加锁；场景未覆盖时继承群设置

        参数:
            scene: 场景标识
            setting: 群设置，为空时按场景读取

        返回:
            bool: 是否停用
        """
        row = await AIChatScene.get_or_none(scene_id=scene)
        value = (row.settings or {}).get("disabled") if row else None
        if value is not None:
            return bool(value)
        if setting is None:
            from zhenxun.models.ai_chat_group import AIChatGroup

            from ._data_source import parse_session

            _, group, _ = parse_session(scene)
            setting = await AIChatGroup.get_setting(group) if group else None
        return bool(setting and setting.disabled)

    @staticmethod
    async def base(scene: str) -> tuple[dict, dict]:
        """读取群、用户绑定及全局默认，不修改共享配置。"""
        from zhenxun.configs.config import Config
        from zhenxun.models.ai_chat_group import AIChatGroup
        from zhenxun.services.ai.llm.manager import get_default_model

        from ._data_source import MODULE, parse_session
        from ._runner import agent_runner

        _, group, user = parse_session(scene)
        setting = await AIChatGroup.get_setting(group) if group else None
        persona = await AIPersona.get_for_scene(group, user)
        values = {
            "model": (setting.model if setting else None)
            or (persona.model if persona else None)
            or get_default_model("chat"),
            "persona_id": persona.id if persona else None,
            "tts": get_default_model("tts"),
            "stt": get_default_model("stt"),
            "disabled": bool(setting and setting.disabled),
            "runner": agent_runner(setting),
            "agent": bool(Config.get_config(MODULE, "AGENT_ENABLED")),
        }
        legacy = {}
        if setting:
            for key, value in {
                "model": setting.model,
                "disabled": setting.disabled,
                "runner": setting.agent_runner,
            }.items():
                if value is not None:
                    legacy[key] = value
        if persona and (
            (group and group in (persona.group_ids or []))
            or (not group and user in (persona.user_ids or []))
        ):
            legacy["persona_id"] = persona.id
        return values, legacy

    @classmethod
    async def resolve(
        cls, scene: str, snapshot: dict | None = None, *, inspect_policy: bool = False
    ) -> dict:
        """返回覆盖、继承和有效值；停用保护由入口另行核验。"""
        snapshot = snapshot or await cls.snapshot(scene, create=False)
        inherited, legacy = await cls.base(scene)
        overrides = {
            key: value
            for key, value in snapshot["settings"].items()
            if key in inherited
        }
        if "persona_id" in overrides and "model" not in legacy:
            from zhenxun.services.ai.llm.manager import get_default_model

            selected = await AIPersona.get_or_none(
                id=overrides["persona_id"], enabled=True
            )
            inherited["model"] = (
                selected.model if selected else None
            ) or get_default_model("chat")
        effective = {**inherited, **overrides}
        blockers = ["场景 AI 已关闭"] if effective["disabled"] else []
        if (
            effective["persona_id"] is not None
            and not await AIPersona.filter(
                id=effective["persona_id"], enabled=True
            ).exists()
        ):
            blockers.append("选定人设已删除或停用")
        if inspect_policy:
            from zhenxun.services.plugin_policy import plugin_policy_service

            try:
                policy = await plugin_policy_service.global_state()
                if not policy["switches"].get("ai_chat", False):
                    blockers.append("AI 聊天插件全局未启用")
            except Exception:
                blockers.append("插件全局状态暂不可确认")
        return {
            "scene_id": scene,
            "revision": snapshot["revision"],
            "execution_revision": snapshot["execution_revision"],
            "overrides": overrides,
            "inherited": inherited,
            "effective": effective,
            "sources": {
                key: "scene" if key in overrides else "inherited" for key in inherited
            },
            "blockers": blockers,
            "scope": "聊天、主动插话及目标场景的 AI 计划",
            "policy_note": "封禁、群插件授权和账号连接在实际执行时继续核验",
            "capabilities": {
                "local_model_selection": effective["runner"] == "local",
                "tts": True,
                "stt": effective["runner"] == "local",
                "native_selection_preserved": effective["runner"] == "dsh",
            },
        }

    @staticmethod
    async def validate(settings: dict) -> None:
        """按实际任务能力校验场景设置，空值恢复继承。"""
        from zhenxun.services.ai.llm.manager import resolve_routes

        allowed = {"model", "persona_id", "tts", "stt", "disabled", "runner", "agent"}
        if set(settings) - allowed:
            raise ValueError("不支持的场景配置字段。")
        for key, value in settings.items():
            if value is None:
                continue
            if key in {"disabled", "agent"}:
                if type(value) is not bool:
                    raise ValueError("开关须为布尔值。")
            elif key == "persona_id":
                if (
                    type(value) is not int
                    or not await AIPersona.filter(id=value, enabled=True).exists()
                ):
                    raise ValueError("人设不存在或未启用。")
            elif key == "runner":
                if not isinstance(value, str) or value not in {"local", "dsh"}:
                    raise ValueError("未知的执行器。")
            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("模型或路由组名称不能为空。")
                resolve_routes(value, "chat" if key == "model" else key)

    @staticmethod
    def models(task: str) -> list[dict]:
        """列出当前任务可用的模型及路由组，不发起网络探测。"""
        from zhenxun.services.ai.config import get_llm_config
        from zhenxun.services.ai.core.exceptions import LLMException
        from zhenxun.services.ai.llm.manager import (
            get_configured_providers,
            get_profile,
            resolve_model_names,
            resolve_routes,
        )

        if task not in {"chat", "tts", "stt"}:
            raise ValueError("不支持的模型任务。")
        items = []
        for provider in get_configured_providers():
            for model in provider.models:
                if model.enabled and get_profile(provider, model).supports_task(task):
                    name = f"{provider.name}/{model.name}"
                    items.append({"id": name, "label": name, "provider": provider.name})
        for name in get_llm_config().model_groups:
            try:
                routes = resolve_routes(name, task)
            except LLMException:
                continue
            members = [f"{r.provider.name}/{r.model.name}" for r in routes]
            excluded = []
            for member in resolve_model_names(name):
                if member not in members:
                    try:
                        resolve_routes(member, task)
                    except LLMException as error:
                        excluded.append({"model": member, "reason": str(error)})
            items.append(
                {
                    "id": f"group:{name}",
                    "value": name,
                    "label": f"路由组 {name}（{len(members)} 条适用线路）",
                    "provider": None,
                    "members": members,
                    "excluded": excluded,
                }
            )
        return items

    @classmethod
    @asynccontextmanager
    async def locked(cls, scene: str):
        """串行处理一个场景的准入和会话变更。"""
        async with cls._locks.setdefault(scene, asyncio.Lock()):
            yield

    @staticmethod
    def _name(value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80 or any(ord(c) < 32 for c in value):
            raise ValueError("对话名称须为 1～80 个可见字符。")
        return value

    @staticmethod
    def _new_id(scene: str) -> str:
        value = f"{scene}:conv:{uuid.uuid4().hex}"
        if len(value) > 255:
            raise ValueError("场景标识过长，不能安全创建对话。")
        return value

    @classmethod
    async def _current(cls, scene: str, *, create: bool, activate: bool = True):
        row = await AIChatScene.get_or_none(scene_id=scene)
        if row is None and create:
            async with in_transaction() as db:
                row, fresh = await AIChatScene.get_or_create(
                    scene_id=scene,
                    defaults={
                        "current_id": scene,
                        "settings": {"_scope_version": 1, "_execution_revision": 0},
                    },
                    using_db=db,
                )
                if fresh:
                    await AIChatConversation.get_or_create(
                        session_id=scene, defaults={"scene_id": scene}, using_db=db
                    )
        if row is None:
            return None, None
        item = (
            await AIChatConversation.get_or_none(
                session_id=row.current_id, scene_id=scene, archived=False
            )
            if row.current_id
            else None
        )
        if row.current_id and item is None:
            # 当前对话已删除或归档时视为没有活跃对话
            changed = await AIChatScene.filter(id=row.id, revision=row.revision).update(
                current_id=None, revision=row.revision + 1
            )
            if not changed:
                raise ConversationConflict("当前对话已变化，请重试。")
            row.current_id, row.revision = None, row.revision + 1
        if (row.settings or {}).get("_scope_version") != 1:
            settings = {
                **(row.settings or {}),
                "_scope_version": 1,
                "_execution_revision": row.revision,
            }
            async with in_transaction() as db:
                changed = (
                    await AIChatScene.filter(id=row.id, revision=row.revision)
                    .using_db(db)
                    .update(settings=settings, revision=row.revision + 1)
                )
                if not changed:
                    raise ConversationConflict("场景迁移期间发生变更，请重试。")
            row.settings, row.revision = settings, row.revision + 1
        if item is None and create and activate:
            async with in_transaction() as db:
                item = await AIChatConversation.create(
                    session_id=cls._new_id(scene), scene_id=scene, using_db=db
                )
                changed = (
                    await AIChatScene.filter(
                        id=row.id, revision=row.revision, current_id=None
                    )
                    .using_db(db)
                    .update(
                        current_id=item.session_id,
                        revision=row.revision + 1,
                        settings={
                            **row.settings,
                            "_execution_revision": row.settings["_execution_revision"]
                            + 1,
                        },
                    )
                )
                if not changed:
                    raise ConversationConflict("当前对话已变化，请重试。")
            row.current_id, row.revision = item.session_id, row.revision + 1
            row.settings["_execution_revision"] += 1
        return row, item

    @classmethod
    async def current(cls, scene: str, *, create: bool = True):
        async with cls.locked(scene):
            return await cls._current(scene, create=create)

    @classmethod
    async def snapshot(cls, scene: str, *, create: bool = True) -> dict:
        row, item = await cls.current(scene, create=create)
        if row is None and not create:
            from zhenxun.models.ai_chat_message import AIChatMessage

            if await AIChatMessage.filter(session_id=scene).exists():
                row, item = await cls.current(scene)
        return {
            "scene_id": scene,
            "session_id": item.session_id if item else None,
            "revision": row.revision if row else 0,
            "name": item.name if item else None,
            "settings": dict(row.settings or {}) if row else {},
            "execution_revision": (row.settings or {}).get("_execution_revision", 0)
            if row
            else 0,
        }

    @staticmethod
    async def check(snapshot: dict) -> None:
        if snapshot["scene_id"] in Conversations._changing:
            raise ConversationConflict("当前对话正在切换或重置，请稍后发送。")
        current = await AIChatScene.get_or_none(scene_id=snapshot["scene_id"])
        if (
            not current
            or current.current_id != snapshot["session_id"]
            or (
                (current.settings or {}).get("_execution_revision", 0)
                != snapshot["execution_revision"]
            )
        ):
            raise ConversationConflict("排队期间对话或设置已变化，请重新发送。")

    @staticmethod
    async def persona(snapshot: dict, group: str | None, user: str | None):
        selected = snapshot.get("settings", {}).get("persona_id")
        if selected is not None:
            persona = await AIPersona.get_or_none(id=selected, enabled=True)
            if persona is None:
                raise ConversationConflict("本对话的人设已停用或删除，请重新选择。")
            return persona
        return await AIPersona.get_for_scene(group, user)

    @staticmethod
    async def drain(scene: str, session_id: str) -> None:
        """停止当前场景已准入任务和整理，保留历史与整理游标。"""
        from zhenxun.services.ai.run.session import session_manager

        from ._data_source import AIChatManager
        from ._memory import Memory

        session_manager.invalidate(scene)
        if session_id != scene:
            session_manager.invalidate(session_id)
        owned = [
            run_id
            for run_id, (active, task) in list(session_manager.runs.items())
            if active.state.get("_conversation", {}).get("scene_id") == scene
            and task
            and task is not asyncio.current_task()
            and not task.done()
        ]
        if owned and not all(
            await asyncio.gather(*(session_manager.stop(run_id) for run_id in owned))
        ):
            raise ConversationConflict("场景仍在清理，未切换配置或历史。")
        await asyncio.wait_for(AIChatManager.stop(scene), 10)
        if session_id != scene:
            await asyncio.wait_for(AIChatManager.stop(session_id), 10)
        await asyncio.wait_for(Memory.cancel_session(session_id), 10)

    @classmethod
    async def change(
        cls,
        scene: str,
        action: str,
        *,
        target: str | None = None,
        name: str | None = None,
        expected_revision: int | None = None,
        settings: dict | None = None,
    ) -> dict:
        """比较场景修订后修改对话；删除仅移除短期数据，保留审计和长期记忆。"""
        if action not in {
            "new",
            "switch",
            "rename",
            "archive",
            "unarchive",
            "delete",
            "reset",
            "configure",
        }:
            raise ValueError("不支持的会话操作。")
        if settings is not None:
            await cls.validate(settings)
        async with cls.locked(scene):
            try:
                row, current = await cls._current(scene, create=True, activate=False)
                if expected_revision is not None and row.revision != expected_revision:
                    raise ConversationConflict("对话已变化，请刷新后重试。")
                item = current
                if target is not None:
                    item = await AIChatConversation.get_or_none(
                        session_id=target, scene_id=scene
                    )
                    if item is None:
                        raise ConversationConflict("对话不存在或不属于当前场景。")
                if item is None and action != "new" and action != "configure":
                    raise ValueError("当前没有活跃对话，请先新建或选择已有对话。")
                if action == "switch" and item.archived:
                    raise ValueError("请先恢复已归档的对话。")
                if action == "rename":
                    name = cls._name(name or "")
                if action == "new":
                    name = cls._name(name or "新对话")
                if action == "configure" and all(
                    (row.settings or {}).get(key) == value
                    for key, value in (settings or {}).items()
                ):
                    return {
                        "scene_id": scene,
                        "session_id": row.current_id,
                        "current_id": row.current_id,
                        "revision": row.revision,
                        "action": action,
                        "history_deleted": False,
                    }
                execution_change = action in {"new", "switch", "configure"} or (
                    current is not None
                    and item is not None
                    and item.session_id == row.current_id
                    and action in {"archive", "delete", "reset"}
                )
                if execution_change:
                    cls._changing.add(scene)
                if current and (
                    action in {"new", "switch", "configure"}
                    or (
                        item.session_id == row.current_id
                        and action in {"archive", "delete", "reset"}
                    )
                ):
                    await cls.drain(scene, current.session_id)
                elif action in {"archive", "delete", "reset"}:
                    await cls.drain(item.session_id, item.session_id)
                if action == "reset":
                    from ._data_source import AIChatManager

                    await AIChatManager.reset(item.session_id)
                async with in_transaction() as db:
                    active = row.current_id
                    if action == "new":
                        item = await AIChatConversation.create(
                            scene_id=scene,
                            session_id=cls._new_id(scene),
                            name=name,
                            using_db=db,
                        )
                        active = item.session_id
                    elif action == "switch":
                        active = item.session_id
                    elif action == "rename":
                        item.name = name
                        await item.save(
                            using_db=db, update_fields=["name", "updated_at"]
                        )
                    elif action in {"archive", "unarchive"}:
                        item.archived = action == "archive"
                        await item.save(
                            using_db=db, update_fields=["archived", "updated_at"]
                        )
                        if item.archived and active == item.session_id:
                            active = None
                    elif action == "delete":
                        from zhenxun.models.ai_chat_message import AIChatMessage
                        from zhenxun.models.ai_memory_cursor import AIMemoryCursor

                        await (
                            AIChatMessage.filter(session_id=item.session_id)
                            .using_db(db)
                            .delete()
                        )
                        await (
                            AIMemoryCursor.filter(session_id=item.session_id)
                            .using_db(db)
                            .delete()
                        )
                        await item.delete(using_db=db)
                        if active == item.session_id:
                            active = None
                    elif action == "configure":
                        if settings is not None:
                            row.settings = {**(row.settings or {}), **settings}
                            row.settings = {
                                k: v for k, v in row.settings.items() if v is not None
                            }
                    if execution_change:
                        row.settings["_execution_revision"] += 1
                    changed = (
                        await AIChatScene.filter(id=row.id, revision=row.revision)
                        .using_db(db)
                        .update(
                            current_id=active,
                            revision=row.revision + 1,
                            settings=row.settings,
                        )
                    )
                    if not changed:
                        raise ConversationConflict("对话已被另一操作修改，请刷新。")
                if action == "new":
                    from ._buffer import ChatBuffer
                    from ._data_source import parse_session

                    bot_id, group_id, _ = parse_session(scene)
                    if group_id:
                        ChatBuffer.reset(bot_id, group_id)
                return {
                    "scene_id": scene,
                    "session_id": item.session_id if item else None,
                    "current_id": active,
                    "revision": row.revision + 1,
                    "action": action,
                    "history_deleted": action in {"reset", "delete"},
                }
            finally:
                cls._changing.discard(scene)
