"""DSH 桌面会话归属、运行授权与收据结算；不启动或终止桌面进程。"""

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import secrets
import time
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname
import uuid

from pydantic import BaseModel, Field, model_validator
import yaml

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.core.exceptions import RunInterrupted
from zhenxun.services.ai.core.messages import ImagePart, TextPart, UsageInfo
from zhenxun.services.ai.run.dsh_desktop import DesktopClient, DshUnavailable
from zhenxun.services.ai.utils.runtime import PermissionUtils
from zhenxun.services.log import logger
from zhenxun.utils.atomic_json import (
    AtomicJsonRecordError,
    mutate_json_locked,
    read_json_locked,
    write_json_locked,
)

LOG_NAME = "dsh"
DSH_PATH = DATA_PATH / "ai" / "dsh"
BOOT_ID = uuid.uuid4().hex
INTEGRATION = Path(__file__).with_name("dsh_guard.mjs")
PRESETS = {"chat": "zhenxun-chat", "agent": "zhenxun-agent"}
TOOLS = {"chat": [], "agent": ["zx_list", "zx_read", "zx_write", "ask_user_question"]}
IMAGE_UNAVAILABLE = (
    "当前 DSH 桌面的图片上传会在内部降级或重发，无法计入统一五次尝试上限；"
    "本接入暂不接受图片，请改发文字。"
)
Approver = Callable[[str, str], Awaitable[bool]]
Asker = Callable[[str], Awaitable[str]]


class DshSettings(BaseModel):
    """显式选择桌面或 SDK；两种后端不自动互相回退。"""

    transport: Literal["desktop", "sdk"] = "desktop"
    desktop_url: str = ""
    home: str = ""
    sdk_bin: str = "dsh"
    sdk_home: str = ""
    sdk_provider: str = ""
    sdk_model: str = ""
    sdk_thinking: str = ""
    timeout: int = Field(default=900, ge=10, le=3600)
    permission: Literal["sandbox", "full"] = "sandbox"
    allow_tools: bool = False
    migration_required: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def legacy(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if value.get("transport") == "sdk":
                for old, new in (
                    ("bin", "sdk_bin"),
                    ("api_provider", "sdk_provider"),
                    ("model", "sdk_model"),
                    ("reasoning_effort", "sdk_thinking"),
                ):
                    if old in value and new not in value:
                        value[new] = value[old]
            value["migration_required"] = (
                value.get("transport", "desktop") == "desktop"
                and not value.get("desktop_url")
                and any(
                    value.get(key)
                    for key in ("bin", "api_provider", "base_url", "provider", "model")
                )
            )
        return value


class DshSessions:
    """按可信所有者保存桌面会话，旧 SDK 文件不参与读取。"""

    FILE = DSH_PATH / "desktop-sessions.json"
    TRANSPORT = "desktop"

    @staticmethod
    def for_settings(settings):
        return SdkSessions if settings.transport == "sdk" else DshSessions

    @staticmethod
    def for_binding(binding):
        return SdkSessions if binding.get("transport") == "sdk" else DshSessions

    @classmethod
    def _read(cls):
        if not cls.FILE.exists():
            return {"version": 1, "owners": {}, "archives": []}
        value = read_json_locked(cls.FILE, None, strict=True)
        cls._validate(value)
        return value

    @staticmethod
    def _validate(value):
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("owners"), dict)
            or not isinstance(value.get("archives"), list)
        ):
            raise AtomicJsonRecordError("桌面会话记录损坏，需要人工处理")
        for key, item in value["owners"].items():
            if (
                not isinstance(item, dict)
                or item.get("owner") != key
                or item.get("mode") not in PRESETS
                or not isinstance(item.get("revision"), int)
                or not re.fullmatch(
                    r"[a-zA-Z0-9_-]{1,128}", str(item.get("remote_id", ""))
                )
                or not item.get("local_id")
            ):
                raise AtomicJsonRecordError("桌面会话归属不可确认")

    @classmethod
    def _mutate(cls, callback):
        cls.FILE.parent.mkdir(parents=True, exist_ok=True)

        def edit(value):
            cls._validate(value)
            return callback(value)

        return mutate_json_locked(
            cls.FILE, {"version": 1, "owners": {}, "archives": []}, edit, strict=True
        )

    @classmethod
    def owner(cls, context, mode: str) -> str:
        return cls.owner_key(cls.mapping_caller(context, mode), mode)

    @staticmethod
    def mapping_caller(context, mode: str):
        """普通定时聊天按目标归属历史，授权与用量仍取真实创建者。"""
        caller = context.run.caller
        if mode == "chat" and caller and caller.source == "task":
            return context.state.get("_dsh_chat_target", caller)
        return caller

    @staticmethod
    def owner_key(caller, mode: str) -> str:
        """按调用者身份与模式计算会话所有者

        参数:
            caller: CallerIdentity
            mode: agent 或 chat

        返回:
            str: 所有者键
        """
        if caller is None or not caller.user_id:
            raise DshUnavailable("无法确认 DSH 会话发起人。")
        scene = (
            ["user", caller.user_id]
            if mode == "agent" or not caller.group_id
            else ["group", caller.group_id]
        )
        key = [
            caller.platform,
            caller.bot_id,
            "webui" if caller.source == "webui" else "bot",
            mode,
            *scene,
        ]
        return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def route(cls, context, mode: str, persona_key: str) -> dict:
        caller = cls.mapping_caller(context, mode)
        owner = cls.owner_key(caller, mode)

        def edit(data):
            item = data["owners"].get(owner)
            if (
                item
                and item.get("inflight")
                and item["inflight"].get("boot") != BOOT_ID
            ):
                item["quarantined"] = "previous_boot_interrupted"
            if item is None:
                remote = "zx-" + uuid.uuid4().hex
                item = {
                    "transport": cls.TRANSPORT,
                    "owner": owner,
                    "mode": mode,
                    "persona": persona_key,
                    "remote_id": remote,
                    "local_id": f"ai_chat:dsh:{owner}:{uuid.uuid4().hex}",
                    "revision": 1,
                    "created": False,
                    "inflight": None,
                    "quarantined": None,
                    "preset": ("sdk-native" if cls.TRANSPORT == "sdk" else "standard")
                    if mode == "agent"
                    else PRESETS[mode],
                    "native": mode == "agent",
                    "identity": {
                        "source": caller.source,
                        "platform": caller.platform,
                        "bot_id": caller.bot_id,
                        "user_id": caller.user_id
                        if mode == "agent" or not caller.group_id
                        else None,
                        "group_id": caller.group_id if mode == "chat" else None,
                    },
                }
                data["owners"][owner] = item
            return dict(item)

        item = cls._mutate(edit)
        context.state["_dsh_binding"] = item
        context.state["_dsh_persona_at_admission"] = persona_key
        context.session_id = context.session.session_id = item["local_id"]
        return item

    @classmethod
    def switch(
        cls, caller, selected: dict | None, expected_revision: int, mode: str = "agent"
    ) -> dict:
        """按可信所有者和模式切换映射，普通聊天只复用自身记录。"""
        if mode not in PRESETS:
            raise DshUnavailable("未知的 DSH 模式。")
        owner = cls.owner_key(caller, mode)
        remote_id = selected["remote_id"] if selected else "zx-" + uuid.uuid4().hex

        def edit(data):
            current = data["owners"].get(owner)
            if (current["revision"] if current else 0) != expected_revision:
                raise DshUnavailable("会话映射已变化，请重新选择。")
            for item in [*data["archives"], *data["owners"].values()]:
                if item["remote_id"] == remote_id and item["owner"] != owner:
                    raise DshUnavailable("该会话已分配给其他用户。")
            owned = next(
                (
                    item
                    for item in [*data["archives"], *data["owners"].values()]
                    if item["remote_id"] == remote_id
                    and item["owner"] == owner
                    and item["mode"] == mode
                ),
                None,
            )
            if (
                mode == "chat"
                and selected
                and (
                    not owned
                    or owned.get("native")
                    or owned.get("quarantined")
                    or owned.get("inflight")
                )
            ):
                raise DshUnavailable("不能接入无可信归属或结果待核验的普通聊天。")
            if current and current.get("inflight"):
                raise DshUnavailable("会话仍在执行，不能切换。")
            if current:
                data["archives"].append(
                    {**current, "archive_reason": "selection_changed"}
                )
            item = {
                "transport": cls.TRANSPORT,
                "owner": owner,
                "mode": mode,
                "persona": owned.get("persona", "") if owned and mode == "chat" else "",
                "remote_id": remote_id,
                "local_id": owned["local_id"]
                if owned and mode == "chat"
                else f"ai_chat:dsh:{owner}:{uuid.uuid4().hex}",
                "revision": expected_revision + 1,
                "created": selected.get("created", True) if selected else False,
                "inflight": None,
                "quarantined": None,
                "preset": selected["preset"]
                if selected
                else PRESETS["chat"]
                if mode == "chat"
                else ("sdk-native" if cls.TRANSPORT == "sdk" else "standard"),
                "native": False
                if mode == "chat"
                else selected["native"]
                if selected
                else True,
                "inherited": bool(selected),
                "cwd": selected["cwd"] if selected else None,
                "model": selected["model"] if selected else None,
                "plan": selected["plan"] if selected else None,
                "title": selected["title"] if selected else "新建桌面会话",
                "identity": {
                    "source": caller.source,
                    "platform": caller.platform,
                    "bot_id": caller.bot_id,
                    "user_id": caller.user_id
                    if mode == "agent" or not caller.group_id
                    else None,
                    "group_id": caller.group_id if mode == "chat" else None,
                },
            }
            if owned and mode == "chat":
                for key in ("preset_revision", "sdk_home"):
                    if key in owned:
                        item[key] = owned[key]
            data["owners"][owner] = item
            return dict(item)

        return cls._mutate(edit)

    @classmethod
    def refresh(cls, item: dict) -> None:
        """取得执行权后读取前一轮结算，不接受已解除的会话映射。"""
        current = cls._read()["owners"].get(item["owner"])
        if not current or current["local_id"] != item["local_id"]:
            raise DshUnavailable("会话已新开或重置，请重新提问。")
        item.update(current)

    @classmethod
    def update(cls, item: dict, **changes) -> dict:
        def edit(data):
            current = data["owners"].get(item["owner"])
            if (
                current is None
                or current["local_id"] != item["local_id"]
                or current["revision"] != item["revision"]
            ):
                raise DshUnavailable("DSH 会话已变化，本次操作已放弃。")
            current.update(changes)
            return dict(current)

        updated = cls._mutate(edit)
        item.update(updated)
        return updated

    @classmethod
    def rotate_remote(cls, item: dict) -> None:
        def edit(data):
            current = data["owners"].get(item["owner"])
            if (
                not current
                or current["local_id"] != item["local_id"]
                or current["revision"] != item["revision"]
            ):
                raise DshUnavailable("会话已变化")
            data["archives"].append(
                {**current, "archive_reason": current.get("quarantined") or "new"}
            )
            current.update(
                remote_id="zx-" + uuid.uuid4().hex,
                revision=current["revision"] + 1,
                created=False,
                inflight=None,
                quarantined=None,
            )
            return dict(current)

        item.update(cls._mutate(edit))

    @classmethod
    def reset(cls, session_id: str) -> None:
        """解除当前映射并保留归属记录；调用方须先等待运行停止。"""
        if not session_id.startswith("ai_chat:dsh:"):
            return

        def edit(data):
            for owner, item in list(data["owners"].items()):
                if item["local_id"] == session_id:
                    if item.get("inflight"):
                        raise DshUnavailable("运行停止尚未确认，不能重置会话。")
                    data["archives"].append({**item, "archive_reason": "unmapped"})
                    del data["owners"][owner]

        cls._mutate(edit)


class SdkSessions(DshSessions):
    FILE = DSH_PATH / "sdk-sessions.json"
    TRANSPORT = "sdk"


@dataclass
class _Turn:
    context: Any
    binding: dict
    client: DesktopClient
    approve: Approver
    ask: Asker
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    proof: dict = field(default_factory=dict)
    usage: UsageInfo = field(default_factory=UsageInfo)
    seen: set[int] = field(default_factory=set)
    responses: dict[str, str] = field(default_factory=dict)
    remote_turn: int | None = None
    admitted: bool = False
    stop_confirmation: str = "not_requested"
    revoked: bool = False
    pending: set = field(default_factory=set)
    event_ids: set[str] = field(default_factory=set)


class DshRuntime:
    """仅管理真寻拥有的桌面轮次；运行撤销与消息投递分别结算。"""

    _turns: ClassVar[dict[str, _Turn]] = {}

    @staticmethod
    def verify_preset(document: dict, mode: str) -> str:
        """核对专用预设的完整组成与守卫文件修订，不接受同名替代能力。"""
        try:
            rows = yaml.safe_load(document["content"])
            wanted = {"native-executor", "zhenxun-guard"}
            if mode == "agent":
                wanted.add("ask-user")
            if (
                not isinstance(rows, list)
                or len(rows) != len(wanted)
                or {r.get("id") for r in rows} != wanted
            ):
                raise ValueError
            for row in rows:
                if set(row) - {"id", "name", "config"}:
                    raise ValueError
                name, config = row["name"], row.get("config") or {}
                if row["id"] == "native-executor":
                    if name != "@deepseek-ai/dsh-tool-executor-native":
                        raise ValueError
                elif row["id"] == "ask-user":
                    if (
                        name != "@deepseek-ai/dsh-tool-ask-user"
                        or config.get("mode", "legacy") != "legacy"
                    ):
                        raise ValueError
                else:
                    uri = urlsplit(name)
                    digest = hashlib.sha256(INTEGRATION.read_bytes()).hexdigest()
                    if (
                        uri.scheme != "file"
                        or uri.netloc
                        or uri.fragment
                        or uri.query != "revision=" + digest
                    ):
                        raise ValueError
                    source = Path(url2pathname(uri.path))
                    if (
                        source.name != "dsh_guard.mjs"
                        or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                    ):
                        raise ValueError
                    if (
                        set(config) != {"root", "mode"}
                        or config["mode"] != mode
                        or Path(config["root"]).resolve() != DSH_PATH.resolve()
                    ):
                        raise ValueError
            return hashlib.sha256(document["content"].encode()).hexdigest()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DshUnavailable("DSH 专用预设或守卫组成不符，未提交问题。") from exc

    @classmethod
    def running(cls):
        return bool(cls._turns)

    @classmethod
    async def status(cls, settings: DshSettings) -> dict:
        if settings.transport == "sdk":
            from .dsh_sdk import SdkClient

            return await SdkClient.status(settings)
        result = {
            "transport": "desktop",
            "connected": False,
            "logged_in": None,
            "guard_ready": None,
            "version": None,
            "presets": {},
            "capabilities": {
                "memory": False,
                "knowledge": False,
                "background": False,
                "shell": False,
                "subagents": False,
                "workspace_tools": True,
                "images": False,
                "native_agent": True,
            },
            "image_limitation": IMAGE_UNAVAILABLE,
            "retry_policy": {"max_attempts": 5, "delays_seconds": [2, 5, 10, 20]},
            "migration_required": settings.migration_required,
        }
        if settings.migration_required:
            result["reason"] = (
                "DSH 设置仍是旧格式：请在聊天设置的 DSH 项中改用 desktop_url 与 home，"
                "无需 API Key。"
            )
            return result
        client = None
        try:
            client = await DesktopClient.open(settings.desktop_url, settings.home)
            presets = await client.rpc("agentPresets/list", args={})
            entries = (
                presets if isinstance(presets, list) else presets.get("presets", [])
            )
            result.update(connected=True, endpoint=client.endpoint)
            result["version"] = client.version
            marker = read_json_locked(
                DSH_PATH / "forwarding-ready.json", {}, strict=True
            )
            result["native_ready"] = bool(
                marker
                and marker.get("pid") == client.identity[0]
                and marker.get("revision")
                == hashlib.sha256(INTEGRATION.read_bytes()).hexdigest()
            )
            result["standard_ready"] = any(p.get("id") == "standard" for p in entries)
            login = await client.rpc("account/getState", args={})
            result["login_state"] = login.get("status")
            result["logged_in"] = False if login.get("status") == "signed-out" else None
            result["presets"] = {
                mode: any(p.get("id") == name for p in entries)
                for mode, name in PRESETS.items()
            }
            result["reason"] = "桌面已连接，每轮执行前会再次校验授权。"
        except (Exception, RunInterrupted) as exc:
            result["reason"] = (
                str(exc)
                if isinstance(exc, DshUnavailable)
                else "DSH 桌面状态不可确认。"
            )
        finally:
            if client:
                await client.close()
        return result

    @classmethod
    async def authorize(cls, token: str, payload: dict) -> dict:
        """按当前运行、会话、预设摘要与真实权限核验一次桌面执行。"""
        turn = next(
            (t for t in cls._turns.values() if secrets.compare_digest(t.token, token)),
            None,
        )
        if turn is None or turn.revoked:
            raise DshUnavailable("运行授权已失效")
        context, binding = turn.context, turn.binding
        PermissionUtils.ensure_current(context)
        current = DshSessions._read()["owners"].get(binding["owner"])
        if (
            not current
            or current["local_id"] != binding["local_id"]
            or current["remote_id"] != binding["remote_id"]
        ):
            raise DshUnavailable("会话映射已失效")
        if (
            payload.get("sessionId") != binding["remote_id"]
            or payload.get("runId") != context.run.run_id
            or payload.get("revision")
            != hashlib.sha256(INTEGRATION.read_bytes()).hexdigest()
        ):
            raise DshUnavailable("桌面执行归属或守卫修订不匹配")
        if binding.get("native"):
            check = context.state.get("_dsh_authorization_check")
            if (
                payload.get("scope") != "retry"
                or payload.get("mode") != "agent"
                or check is None
                or await check() != "agent"
            ):
                raise DshUnavailable("本次转发已结束，不能继续自动重试。")
            return {"allowed": True}
        execution_mode = "chat" if binding["preset"] == PRESETS["chat"] else "agent"
        if payload.get("mode") != execution_mode or sorted(
            payload.get("tools") or []
        ) != sorted(TOOLS[execution_mode]):
            raise DshUnavailable("桌面工具清单不匹配")
        check = context.state.get("_dsh_authorization_check")
        if check is None or await check() != binding["mode"]:
            raise DshUnavailable("DSH 模式或当前授权已变化")
        name = payload.get("tool")
        if name and name not in TOOLS[execution_mode]:
            raise DshUnavailable("工具未授权")
        model = payload.get("model") or {}
        if model.get("provider") not in {"deepseek", "deepseek-account"}:
            raise DshUnavailable("当前桌面模型提供方的中断契约未核验，已阻止执行。")
        turn.proof = {
            "mode": binding["mode"],
            "tools": payload["tools"],
            "model": payload.get("model"),
            "revision": payload["revision"],
        }
        context.run.operations.setdefault("dsh", {}).update(
            {
                "session_id": binding["remote_id"],
                "mode": binding["mode"],
                "guard": "verified",
                "model": payload.get("model"),
                "stop": turn.stop_confirmation,
            }
        )
        if payload.get("model"):
            cls.record_selection(turn, payload["model"])
        return {"allowed": True}

    @classmethod
    async def start(cls, context, binding, settings, approve, ask) -> _Turn:
        if settings.transport != binding.get("transport", "desktop"):
            raise DshUnavailable("执行器传输已变化，请重新选择会话。")
        if settings.transport == "sdk":
            from .dsh_sdk import SdkClient

            return await SdkClient.start_turn(context, binding, settings, approve, ask)
        if settings.migration_required:
            raise DshUnavailable(
                "旧 SDK 配置不能连接桌面，请更新 desktop_url 与 home。"
            )
        if not binding.get("native") and (
            binding.get("inflight") or binding.get("quarantined")
        ):
            DshSessions.rotate_remote(binding)
        client = await DesktopClient.open(settings.desktop_url, settings.home)
        turn = _Turn(context, binding, client, approve, ask)
        cls._turns[context.run.run_id] = turn
        try:
            if client.version != "0.2.2-rc.3":
                raise DshUnavailable("当前桌面版本不在已核验协议范围内，已阻止执行。")
            guard_revision = hashlib.sha256(INTEGRATION.read_bytes()).hexdigest()
            if binding.get("native"):
                await cls._start_native(turn, settings, guard_revision)
                return turn
            if (
                binding["created"]
                and not binding.get("inherited")
                and (
                    binding.get("desktop_identity") != list(client.identity)
                    or binding.get("guard_revision") != guard_revision
                )
            ):
                DshSessions.rotate_remote(binding)
            presets = await client.rpc("agentPresets/list", args={})
            entries = (
                presets if isinstance(presets, list) else presets.get("presets", [])
            )
            if not any(p.get("id") == binding["preset"] for p in entries):
                raise DshUnavailable("缺少真寻专用 DSH 安全预设，已阻止执行。")
            document = await client.rpc(
                "agentPresets/read", args={"agentPreset": binding["preset"]}
            )
            execution_mode = "chat" if binding["preset"] == PRESETS["chat"] else "agent"
            preset_revision = cls.verify_preset(document, execution_mode)
            if (
                binding["created"]
                and not binding.get("inherited")
                and binding.get("preset_revision") != preset_revision
            ):
                DshSessions.rotate_remote(binding)
            from nonebot import get_driver

            port = int(get_driver().config.port)
            workspace = (
                Path(binding["cwd"])
                if binding.get("inherited")
                else DSH_PATH / "workspaces" / binding["owner"]
            )
            group = None
            if binding["mode"] == "chat" and not binding.get("inherited"):
                from .dsh_control import DshControl

                group = await DshControl.chat_workspace(client)
                workspace = Path(group["path"])
                if binding["created"] and binding.get("cwd") != str(workspace):
                    DshSessions.rotate_remote(binding)
            workspace.mkdir(parents=True, exist_ok=True)
            leases = DSH_PATH / "leases"
            leases.mkdir(parents=True, exist_ok=True)
            write_json_locked(
                leases / (binding["remote_id"] + ".json"),
                {
                    "sessionId": binding["remote_id"],
                    "runId": context.run.run_id,
                    "token": turn.token,
                    "mode": execution_mode,
                    "requestId": turn.request_id,
                    "modelSelection": binding.get("model")
                    if binding.get("inherited")
                    else None,
                    "cwd": str(workspace.resolve()),
                    "expires": time.time() + settings.timeout,
                    "authorize": f"http://127.0.0.1:{port}/zhenxun/api/ai/dsh/authorize",
                },
            )
            if not binding["created"]:
                await client.rpc(
                    "session/create",
                    {
                        "sessionId": binding["remote_id"],
                        **(
                            {"workspaceId": group["workspaceId"]}
                            if group
                            else {"cwd": str(workspace.resolve())}
                        ),
                        "agentPreset": binding["preset"],
                    },
                )
                if not turn.proof:
                    raise DshUnavailable("专用守卫没有完成执行前握手，未提交问题。")
                DshSessions.update(
                    binding,
                    created=True,
                    desktop_identity=list(client.identity),
                    guard_revision=guard_revision,
                    preset_revision=preset_revision,
                    cwd=str(workspace),
                )
            state = await client.rpc(
                "session/state", {"sessionId": binding["remote_id"]}
            )
            if (
                (not state.get("active") and not binding.get("inherited"))
                or state.get("running")
                or state.get("pendingInput")
                or state.get("queue")
            ):
                raise DshUnavailable("该 DSH 会话仍有未结束的执行，未提交新输入。")
            DshSessions.update(
                binding,
                inflight={
                    "boot": BOOT_ID,
                    "run_id": context.run.run_id,
                    "request_id": turn.request_id,
                },
            )
            return turn
        except BaseException:
            try:
                DshSessions.update(binding, quarantined="initialization_unconfirmed")
            except Exception as exc:
                context.run.operations.setdefault("dsh", {})["mapping_error"] = type(
                    exc
                ).__name__
                logger.warning(
                    "DSH 初始化结果未能写入映射，保留原记录等待核验。", LOG_NAME
                )
            finally:
                await cls.release(turn)
            raise

    @classmethod
    async def _start_native(cls, turn, settings, revision):
        """接入原生 Agent，保留其模型、思考等级、计划模式及执行权限。"""
        from nonebot import get_driver

        from .dsh_control import DshControl

        binding, client, context = turn.binding, turn.client, turn.context
        marker = read_json_locked(DSH_PATH / "forwarding-ready.json", None, strict=True)
        if (
            not marker
            or marker.get("pid") != client.identity[0]
            or marker.get("revision") != revision
        ):
            raise DshUnavailable("桌面转发重试组件尚未就绪，未发送问题。")
        if not binding["created"]:
            workspace = DSH_PATH / "agent-workspaces" / binding["owner"]
            workspace.mkdir(parents=True, exist_ok=True)
            created = await client.rpc(
                "session/create",
                {
                    "sessionId": binding["remote_id"],
                    "cwd": str(workspace.resolve()),
                    "agentPreset": "standard",
                },
            )
            if created.get("sessionId") != binding["remote_id"]:
                raise DshUnavailable("桌面返回了不同会话，未提交问题。")
            DshSessions.update(binding, created=True, cwd=str(workspace.resolve()))
        observed = await client.rpc(
            "session/projections", {"sessionId": binding["remote_id"]}
        )
        if not observed or observed.get("kind") != "sequenced":
            raise DshUnavailable("原会话不存在或需要桌面处理格式升级。")
        info = DshControl.describe(
            {
                "sessionId": binding["remote_id"],
                "projections": observed,
                "cwd": binding["cwd"],
            }
        )
        if info["subagent"]:
            raise DshUnavailable("子代理会话不能作为独立 QQ 会话接入。")
        state = await client.rpc("session/state", {"sessionId": binding["remote_id"]})
        if state.get("running") or state.get("queue") or state.get("pendingInput"):
            raise DshUnavailable("桌面会话仍有任务或等待输入，未抢占原任务。")
        await client.rpc(
            "workspace/unarchiveSession", {"sessionId": binding["remote_id"]}
        )
        (DSH_PATH / "leases").mkdir(parents=True, exist_ok=True)
        write_json_locked(
            DSH_PATH / "leases" / (binding["remote_id"] + ".json"),
            {
                "sessionId": binding["remote_id"],
                "runId": context.run.run_id,
                "requestId": turn.request_id,
                "native": True,
                "token": turn.token,
                "expires": time.time() + settings.timeout,
                "authorize": f"http://127.0.0.1:{int(get_driver().config.port)}/zhenxun/api/ai/dsh/authorize",
            },
        )
        DshSessions.update(
            binding,
            model=info["model"],
            plan=info["plan"],
            preset=info["preset"],
            title=info["title"],
            quarantined=None,
            inflight={
                "boot": BOOT_ID,
                "run_id": context.run.run_id,
                "request_id": turn.request_id,
            },
        )
        turn.proof = {
            "mode": "native",
            "model": info["model"],
            "preset": info["preset"],
        }
        context.run.operations["dsh"] = {
            "session_id": binding["remote_id"],
            "mode": "native",
            "guard": "desktop_permissions",
            "model": info["model"],
            "preset": info["preset"],
            "plan": info["plan"],
            "stop": "not_requested",
            "background_stop_guarantee": False,
        }
        cls.record_selection(turn, info["model"])

    @classmethod
    async def prompt(cls, turn: _Turn, content: list[dict]):
        if turn.binding.get("transport") == "sdk":
            return await turn.client.prompt(turn, content)
        from websockets.exceptions import WebSocketException

        if not (turn and turn.binding.get("native")) and any(
            block.get("type") != "text" for block in content
        ):
            raise DshUnavailable(IMAGE_UNAVAILABLE)

        async def collect():
            await turn.client.verify_process()
            async with turn.client.mux() as ws:
                follow, events = uuid.uuid4().hex, uuid.uuid4().hex
                await ws.send(
                    json.dumps(
                        {
                            "type": "open",
                            "streamId": follow,
                            "endpoint": "session/follow",
                            "payload": {
                                "args": {
                                    "request": {
                                        "address": {
                                            "kind": "session",
                                            "sessionId": turn.binding["remote_id"],
                                        },
                                        "activate": False,
                                        "maxMessages": 1,
                                    }
                                }
                            },
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "open",
                            "streamId": events,
                            "endpoint": "$events",
                            "payload": {"args": {}},
                        }
                    )
                )
                ready = False
                client_id = None
                submitted = False
                while True:
                    frame = await cls._receive(ws, turn)
                    if frame.get("type") in {"error", "end"}:
                        raise RunInterrupted("DSH 事件订阅")
                    value = frame.get("value") or {}
                    if frame.get("streamId") == events:
                        if value.get("type") == "ready":
                            client_id = value["clientId"]
                        elif (
                            value.get("type") == "waterfall"
                            and value.get("agentId") == turn.binding["remote_id"]
                            and client_id
                        ):
                            task = asyncio.create_task(
                                cls._answer_event(turn, client_id, value)
                            )
                            turn.pending.add(task)
                    elif frame.get("streamId") == follow:
                        if value.get("type") == "snapshot":
                            if submitted:
                                raise RunInterrupted("DSH 意外历史快照")
                            ready = True
                        elif value.get("type") == "event":
                            outcome = cls.consume(turn, value.get("event") or {})
                            if outcome is not None:
                                if not turn.proof:
                                    raise DshUnavailable("桌面未提供有效执行守卫证据。")
                                if outcome == "error":
                                    raise RunInterrupted("DSH 模型请求")
                                if outcome != "completed":
                                    raise DshUnavailable(
                                        f"DSH 本轮未完成（{outcome}）。"
                                    )
                                return (
                                    "\n\n".join(
                                        t for t in turn.responses.values() if t
                                    ),
                                    turn.usage,
                                    outcome,
                                )
                    for task in list(turn.pending):
                        if task.done():
                            turn.pending.remove(task)
                            task.result()
                    if ready and client_id and not submitted:
                        submitted = True
                        receipt = await turn.client.rpc(
                            "session/prompt",
                            {
                                "sessionId": turn.binding["remote_id"],
                                "requestId": turn.request_id,
                                "mode": "queue",
                                "content": content,
                            },
                        )
                        if not receipt or receipt.get("accepted") is not True:
                            raise RunInterrupted("DSH 输入接收")

        try:
            return await collect()
        except (WebSocketException, OSError, ValueError) as exc:
            raise RunInterrupted("DSH 事件连接") from exc

    @staticmethod
    async def _receive(ws, turn):
        incoming = asyncio.create_task(ws.recv())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {incoming, *turn.pending}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done - {incoming}:
                    turn.pending.discard(task)
                    task.result()
                if incoming in done:
                    return json.loads(incoming.result())
        finally:
            if not incoming.done():
                incoming.cancel()
            await asyncio.gather(incoming, return_exceptions=True)

    @staticmethod
    def record_selection(turn: _Turn, model: dict | None) -> None:
        """记录执行端确认的模型选择，不修改远端默认值或原生参数。"""
        if not model or not model.get("model"):
            return
        selected = {
            key: model[key]
            for key in ("provider", "model", "reasoningEffort")
            if key in model
        }
        turn.context.run.current_model = "dsh:" + str(selected["model"])
        configuration = turn.context.state.setdefault("_configuration", {})
        configuration.update(model=turn.context.run.current_model, native=selected)
        turn.context.run.operations.setdefault("dsh", {})["model"] = selected
        if turn.binding.get("model") != selected:
            DshSessions.for_binding(turn.binding).update(turn.binding, model=selected)

    @staticmethod
    def consume(turn: _Turn, event: dict) -> str | None:
        seq = event.get("seq")
        if not isinstance(seq, int) or seq in turn.seen:
            return None
        turn.seen.add(seq)
        data, kind = event.get("data") or {}, event.get("type")
        if (
            kind == "user/message"
            and (data.get("source") or {}).get("rpcId") == turn.request_id
        ):
            turn.admitted = True
        if kind == "turn/start":
            turn.remote_turn = data.get("turn")
        if not turn.admitted or data.get("turn") not in {None, turn.remote_turn}:
            return None
        if kind == "request/header":
            header = data.get("header") or {}
            selection = dict(header.get("config") or {})
            if (header.get("adapterDefaults") or {}).get("reasoningEffort") is True:
                selection.pop("reasoningEffort", None)
            DshRuntime.record_selection(turn, selection)
        elif kind == "model/selection":
            DshRuntime.record_selection(turn, data)
        tools = turn.context.run.operations.setdefault("dsh_tools", {})
        callback = turn.context.state.get("_dsh_event")
        if kind in {"llm/retry", "llm/retry-started"}:
            progress = {
                "attempt": int(data.get("retry") or 0) + 1,
                "max_attempts": 5,
                "delay_seconds": float(data.get("delayMs") or 0) / 1000,
                "state": "waiting" if kind == "llm/retry" else "running",
                "code": (data.get("failure") or {}).get("code"),
            }
            turn.context.run.operations.setdefault("dsh", {})["retry"] = progress
            if callback:
                callback(
                    {"type": "retry", "run_id": turn.context.run.run_id, **progress}
                )
        if kind == "tool/call":
            call_id = str(data.get("callId") or "")
            tools[call_id] = {"name": data.get("name"), "status": "running"}
            if callback:
                callback(
                    {
                        "type": "tool_start",
                        "run_id": turn.context.run.run_id,
                        "tool_call_id": call_id,
                        "name": data.get("name"),
                        "arguments": json.dumps(
                            data.get("arguments"), ensure_ascii=False
                        )[:2000],
                    }
                )
        elif kind == "tool/result":
            message = data.get("message") or {}
            call_id = str(message.get("toolCallId") or "")
            if call_id in tools:
                tools[call_id]["status"] = (
                    "failed" if message.get("isError") else "completed"
                )
                if callback:
                    callback(
                        {
                            "type": "tool_end",
                            "run_id": turn.context.run.run_id,
                            "tool_call_id": call_id,
                            "name": tools[call_id]["name"],
                            "is_error": bool(message.get("isError")),
                            "result": "\n".join(
                                p.get("text", "")
                                for p in message.get("content", [])
                                if p.get("type") == "text"
                            )[:2000],
                        }
                    )
        if kind == "assistant/message":
            message = data.get("message") or {}
            message_id = message.get("id")
            if not message_id or message_id in turn.responses:
                return None
            turn.responses[message_id] = "".join(
                str(p.get("text") or "")
                for p in message.get("content") or []
                if p.get("type") == "text"
            )
            counts = data.get("usage") or {}
            cached = int(counts.get("cacheReadTokens") or 0)
            prompt = int(counts.get("inputTokens") or 0) + cached
            output = int(counts.get("outputTokens") or 0)
            turn.usage.prompt_tokens += prompt
            turn.usage.prompt_cache_hit_tokens += cached
            if "cacheReadTokens" in counts:
                turn.context.run.budget["reported_cache_read"] = (
                    turn.usage.prompt_cache_hit_tokens
                )
            turn.usage.completion_tokens += output
            turn.usage.total_tokens += int(counts.get("totalTokens") or prompt + output)
            turn.context.run.budget["tokens"] = turn.usage.total_tokens
        if kind == "turn/end":
            return (data.get("reason") or {}).get("kind", "unknown")
        return None

    @classmethod
    async def _answer_event(cls, turn, client_id, value):
        event_id = value.get("eventId")
        if not isinstance(event_id, str) or event_id in turn.event_ids:
            return
        turn.event_ids.add(event_id)
        request = value.get("request") or {}
        PermissionUtils.ensure_current(turn.context)
        if value.get("event") == "approval/request":
            allowed = await turn.approve(
                str(request.get("toolName", "")), str(request.get("reason", ""))
            )
            answer = "allowed-once" if allowed else "rejected"
        elif value.get("event") == "user-questions/request":
            answers = []
            for q in request.get("questions") or []:
                options = [str(o.get("label", "")) for o in q.get("options") or []]
                question = str(q.get("question", ""))
                if q.get("detail"):
                    question += "\n\n" + str(q["detail"])
                if options:
                    question += "\n" + "\n".join(
                        f"{i}. {label}" for i, label in enumerate(options, 1)
                    )
                reply = (await turn.ask(question)).strip()
                if reply.isdigit() and 0 < int(reply) <= len(options):
                    reply = options[int(reply) - 1]
                answers.append(
                    {
                        "id": q.get("id"),
                        "selected": [reply] if reply in options else [],
                        **({"custom": reply} if reply not in options else {}),
                    }
                )
            answer = {"answers": answers}
        else:
            return
        await turn.client.rpc(
            "$events/result",
            args={
                "clientId": client_id,
                "eventId": value["eventId"],
                "outcome": {"kind": "result", "value": answer},
            },
        )

    @classmethod
    async def interrupt(cls, turn):
        turn.revoked = True
        for entry in turn.context.run.operations.get("dsh_tools", {}).values():
            if entry["status"] == "running":
                entry["status"] = "unknown"
        from zhenxun.services.ai.run.session import session_manager

        remaining = session_manager.cleanup_remaining(turn.context)
        native = turn.binding.get("native")
        explicit = (
            turn.context.run.cancellation_token
            and turn.context.run.cancellation_token.reason == "cancelled"
        )
        try:
            if native and not explicit:
                state = await asyncio.wait_for(
                    turn.client.rpc(
                        "session/state", {"sessionId": turn.binding["remote_id"]}
                    ),
                    max(0.001, remaining),
                )
                turn.stop_confirmation = (
                    "unknown"
                    if not state.get("loaded", turn.binding.get("transport") != "sdk")
                    else "foreground_idle"
                    if not state.get("running")
                    and not state.get("queue")
                    and not state.get("pendingInput")
                    else "desktop_continues"
                )
            else:
                turn.stop_confirmation = await asyncio.wait_for(
                    turn.client.stop(turn.binding["remote_id"], turn.request_id),
                    max(0.001, remaining),
                )
                if native and turn.stop_confirmation == "confirmed":
                    turn.stop_confirmation = "foreground_confirmed"
        except (Exception, RunInterrupted):
            turn.stop_confirmation = "unknown"
        turn.context.run.operations.setdefault("dsh", {})["stop"] = (
            turn.stop_confirmation
        )
        try:
            DshSessions.for_binding(turn.binding).update(
                turn.binding,
                quarantined="interrupted",
                inflight=None
                if turn.stop_confirmation
                in {
                    "confirmed",
                    "foreground_confirmed",
                    "foreground_idle",
                    "target_removed",
                }
                else {
                    "boot": BOOT_ID,
                    "run_id": turn.context.run.run_id,
                    "request_id": turn.request_id,
                },
            )
        except Exception as exc:
            turn.context.run.operations["dsh"]["mapping_error"] = type(exc).__name__
            logger.warning("DSH 停止结果未能写入映射，保留原记录等待核验。", LOG_NAME)

    @classmethod
    def extend(cls, turn, seconds: float) -> None:
        """延长桌面运行授权的有效期；SDK 运行由宿主心跳维持

        参数:
            turn: 当前轮次
            seconds: 延长秒数
        """
        if turn.binding.get("transport") == "sdk" or turn.revoked or seconds <= 0:
            return

        def mutate(lease: dict) -> bool:
            if lease.get("runId") != turn.context.run.run_id:
                return False
            lease["expires"] = float(lease.get("expires", 0)) + seconds
            return True

        try:
            mutate_json_locked(
                DSH_PATH / "leases" / (turn.binding["remote_id"] + ".json"),
                {},
                mutate,
                write_if=bool,
            )
        except (OSError, AtomicJsonRecordError):
            logger.warning("DSH 运行授权延期失败，后续工具调用可能被拒绝。", LOG_NAME)

    @classmethod
    async def release(cls, turn):
        turn.revoked = True
        for task in turn.pending:
            task.cancel()
        if turn.pending:
            await asyncio.gather(*turn.pending, return_exceptions=True)
        cls._turns.pop(turn.context.run.run_id, None)
        if turn.binding.get("transport") == "sdk":
            await turn.client.close()
            return
        try:
            (DSH_PATH / "leases" / (turn.binding["remote_id"] + ".json")).unlink(
                missing_ok=True
            )
        except OSError:
            turn.context.run.operations.setdefault("dsh", {})["cleanup_pending"] = True
            logger.warning("DSH 运行已撤权，授权文件清理待重试。", LOG_NAME)
        finally:
            await turn.client.close()

    @classmethod
    async def close(cls):
        async def stop(turn):
            try:
                await cls.interrupt(turn)
            finally:
                await cls.release(turn)

        await asyncio.gather(
            *(stop(t) for t in list(cls._turns.values())), return_exceptions=True
        )


async def content_blocks(
    parts: list[Any], *, native: bool = False
) -> list[dict[str, Any]]:
    """把文字转换为桌面内容块；不支持可靠中断的图片输入明确拒绝

    参数:
        parts: 消息片段

    返回:
        list[dict[str, Any]]: 内容块
    """
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, ImagePart):
            if not native:
                raise DshUnavailable(IMAGE_UNAVAILABLE)
            from PIL import Image

            from zhenxun.services.ai.utils.utils import fetch_media

            raw = part.raw
            if raw is None and part.path is not None:
                raw = await asyncio.to_thread(part.path.read_bytes)
            if raw is None and part.url:
                raw = await fetch_media(part.url)
            if not raw:
                raise DshUnavailable("图片内容不可读取，未发送桌面请求。")
            with Image.open(BytesIO(raw)) as image:
                mime = {
                    "PNG": "image/png",
                    "JPEG": "image/jpeg",
                    "GIF": "image/gif",
                    "WEBP": "image/webp",
                }.get(image.format or "")
                if not mime:
                    output = BytesIO()
                    image.save(output, "PNG")
                    raw, mime = output.getvalue(), "image/png"
            blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(raw).decode(),
                    "mediaType": mime,
                }
            )
            continue
        kind = getattr(part, "type", "")
        blocks.append({"type": "text", "text": f"[{kind or '附件'}]"})
    return blocks
