"""桌面能力查询及只接受当前运行凭据的执行授权。"""

import asyncio
import ipaddress

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from zhenxun.services.ai.run.dsh import DshRuntime, DshSessions

from ....base_model import Result
from ....utils import authentication

router = APIRouter()


class DshSessionChoice(BaseModel):
    owner: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    remote_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    expected_revision: int = Field(ge=0)


def _management_context(request):
    from zhenxun.builtin_plugins.ai_chat._agent import bind_source
    from zhenxun.services.ai.run.context import RunContext, ScheduledDeps

    from ....security import validate_access_token
    from .console import management_subject

    subject = management_subject(request)
    context = RunContext(
        deps=ScheduledDeps(user_id=f"webui:{subject}", platform="webui")
    )
    token = request.headers.get("Authorization", "")
    bind_source(context, "webui", management_check=lambda: validate_access_token(token))
    return context


def _agent_targets(request) -> list[dict]:
    """可选择 Agent 会话的归属：当前管理账号、已连接 Bot 的超级用户，以及已有
    Agent 会话记录的 QQ 超管

    参数:
        request: 当前请求

    返回:
        list[dict]: 归属，含 owner、label 与 caller
    """
    import nonebot

    from zhenxun.services.ai.run.context import CallerIdentity
    from zhenxun.utils.platform import PlatformUtils

    callers = [("当前 WebUI 账号", _management_context(request).run.caller)]
    for bot in nonebot.get_bots().values():
        platform = PlatformUtils.get_platform_scope(bot)
        for user_id in PlatformUtils.get_superusers(bot):
            callers.append(
                (
                    f"QQ 超管 {user_id} · Bot {bot.self_id}",
                    CallerIdentity("qq", platform, str(bot.self_id), user_id, None),
                )
            )
    for item in DshSessions._read()["owners"].values():
        who = item.get("identity") or {}
        if item["mode"] == "agent" and who.get("source") not in (None, "webui"):
            callers.append(
                (
                    f"QQ 超管 {who.get('user_id')} · Bot {who.get('bot_id')}",
                    CallerIdentity(
                        who["source"],
                        who.get("platform", ""),
                        str(who.get("bot_id", "")),
                        who.get("user_id"),
                        None,
                    ),
                )
            )
    targets: dict[str, dict] = {}
    for label, caller in callers:
        owner = DshSessions.owner_key(caller, "agent")
        targets.setdefault(owner, {"owner": owner, "label": label, "caller": caller})
    return list(targets.values())


def _target(request, owner: str | None) -> dict:
    targets = _agent_targets(request)
    for target in targets:
        if owner in (None, target["owner"]):
            return target
    raise HTTPException(404, detail={"message": "该超管不可选择，请刷新列表。"})


@router.get("/dsh/sessions", dependencies=[authentication()], response_model=Result)
async def desktop_sessions(request: Request, owner: str | None = None) -> Result:
    """列出可选归属，以及指定归属可接入的桌面会话"""
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
    from zhenxun.services.ai.core.exceptions import RunInterrupted
    from zhenxun.services.ai.run.dsh_control import DshControl
    from zhenxun.services.ai.run.dsh_desktop import DshUnavailable

    target = _target(request, owner)
    try:
        items = await DshControl.list(dsh_settings(), target["owner"])
    except (DshUnavailable, RunInterrupted) as exc:
        raise HTTPException(
            409,
            detail={
                "code": "dsh_unavailable",
                "message": str(exc)
                if isinstance(exc, DshUnavailable)
                else "桌面连接失败，请确认 DSH 桌面已启动。",
            },
        ) from exc
    current = DshSessions._read()["owners"].get(target["owner"])
    return Result.ok(
        {
            "targets": [
                {"owner": t["owner"], "label": t["label"]}
                for t in _agent_targets(request)
            ],
            "owner": target["owner"],
            "items": items,
            "current_id": current["remote_id"] if current else None,
            "revision": current["revision"] if current else 0,
        }
    )


@router.post("/dsh/session", dependencies=[authentication()], response_model=Result)
async def choose_desktop_session(payload: DshSessionChoice, request: Request) -> Result:
    """为指定归属显式选择原会话或准备新会话，后续发送仍按发起人身份准入。"""
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
    from zhenxun.services.ai.run.dsh_control import DshControl
    from zhenxun.services.ai.run.dsh_desktop import DshUnavailable

    target = _target(request, payload.owner)
    try:
        selected = await DshControl.choose(
            target["caller"],
            dsh_settings(),
            payload.remote_id,
            payload.expected_revision,
        )
    except DshUnavailable as exc:
        raise HTTPException(
            409, detail={"code": "dsh_selection_conflict", "message": str(exc)}
        ) from exc
    return Result.ok(
        selected,
        info="已选择原会话，保留桌面模型、思考等级和模式。"
        if payload.remote_id
        else "已准备新标准会话，首次发送时建立。",
    )


class DshAuthorization(BaseModel):
    sessionId: str = Field(max_length=128)
    runId: str = Field(max_length=64)
    revision: str = Field(min_length=64, max_length=64)
    mode: str = Field(max_length=8)
    tools: list[str] = Field(max_length=8)
    tool: str | None = Field(default=None, max_length=80)
    model: dict | None = None
    scope: str | None = Field(default=None, max_length=16)


@router.get("/dsh/status", dependencies=[authentication()], response_model=Result)
async def desktop_status() -> Result:
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings

    status = await DshRuntime.status(dsh_settings())
    try:
        status["sessions"] = list(DshSessions._read()["owners"].values())
        status["mapping_trusted"] = True
    except Exception:
        status["sessions"] = []
        status["mapping_trusted"] = False
        status["reason"] = "会话映射损坏或不可读取，已阻止新运行。"
    return Result.ok(status)


@router.post("/dsh/authorize")
async def authorize_desktop(payload: DshAuthorization, request: Request) -> dict:
    """桌面工具使用单轮短期凭据；不接受前端 JWT 代替执行归属。"""
    try:
        if (
            request.client is None
            or not ipaddress.ip_address(request.client.host).is_loopback
        ):
            raise ValueError
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or len(header) > 100:
            raise ValueError
        return await DshRuntime.authorize(header[7:], payload.model_dump())
    except (Exception, asyncio.CancelledError) as exc:
        raise HTTPException(403, detail={"code": "dsh_execution_denied"}) from exc
