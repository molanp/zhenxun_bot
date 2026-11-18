"""桌面能力查询及只接受当前运行凭据的执行授权。"""

import asyncio
import ipaddress
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from nonebot import get_driver
from pydantic import BaseModel, Field

from zhenxun.services.ai.run.dsh import DshRuntime, DshSessions, DshSettings
from zhenxun.services.ai.run.dsh_install import SdkInstallError, sdk_installer
from zhenxun.services.log import logger

from ....base_model import Result
from ....utils import authentication

router = APIRouter()
sdk_installer.bind_lifecycle(get_driver())


class DshInstallAction(BaseModel):
    model_config = {"extra": "forbid"}
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    expected_operation: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


@router.get("/dsh/sdk", dependencies=[authentication()], response_model=Result)
async def sdk_install_status() -> Result:
    """读取独立安装状态及可选模型，不返回凭据或启动宿主。"""
    from zhenxun.services.ai.run.dsh_sdk import eligible_providers

    providers = [
        {"name": item.name, "models": [m.name for m in item.models if m.enabled]}
        for item in eligible_providers()
    ]
    return Result.ok({**sdk_installer.status(), "providers": providers})


@router.post("/dsh/sdk/install", dependencies=[authentication()], response_model=Result)
async def install_sdk(payload: DshInstallAction) -> Result:
    """启动或查询同一次后台 SDK 安装，固定包名、版本和目标目录。"""
    try:
        result = sdk_installer.start(payload.operation_id, payload.expected_operation)
    except (SdkInstallError, OSError, RuntimeError, TimeoutError) as error:
        raise HTTPException(
            409,
            detail={
                "message": str(error)
                if isinstance(error, SdkInstallError)
                else "安装状态无法确认，请刷新后检查。"
            },
        ) from error
    return Result.ok(result)


@router.post("/dsh/sdk/cancel", dependencies=[authentication()], response_model=Result)
async def cancel_sdk_install(payload: DshInstallAction) -> Result:
    """停止本次安装并等待清理，不停止正常 SDK 或桌面会话。"""
    try:
        result = await sdk_installer.cancel(payload.operation_id)
    except SdkInstallError as error:
        raise HTTPException(409, detail={"message": str(error)}) from error
    return Result.ok(result)


class DshSessionChoice(BaseModel):
    owner: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    remote_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    expected_revision: int = Field(ge=0)


class DshStopRequest(DshSessionChoice):
    run_id: str = Field(min_length=1, max_length=64)


@router.post(
    "/dsh/session/stop", dependencies=[authentication()], response_model=Result
)
async def stop_dsh_session(payload: DshStopRequest) -> Result:
    """按持久映射和运行收据停止指定前台请求，不关闭共享宿主。"""
    from zhenxun.builtin_plugins.ai_chat._commands import stop_message
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
    from zhenxun.services.ai.run.dsh_control import DshControl
    from zhenxun.services.ai.run.session import session_manager

    settings = dsh_settings()
    records = DshSessions.for_settings(settings)
    item = records._read()["owners"].get(payload.owner)
    if (
        not item
        or item["revision"] != payload.expected_revision
        or item["remote_id"] != payload.remote_id
        or (item.get("inflight") or {}).get("run_id") != payload.run_id
    ):
        raise HTTPException(409, detail={"code": "dsh_run_changed"})
    entry = session_manager.runs.get(payload.run_id)
    if entry:
        if entry[0].state.get("_dsh_binding", {}).get("local_id") != item["local_id"]:
            raise HTTPException(409, detail={"code": "dsh_run_changed"})
        done = await session_manager.stop(payload.run_id)
        state = entry[0].run.operations.get("dsh", {}).get("stop", "unknown")
        return Result.ok(
            {"state": state, "local_done": done}, info=stop_message([state], done)
        )
    result = await DshControl.stop_owned(settings, dict(item), payload.run_id)
    return Result.ok(result, info=stop_message([result["state"]], True))


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
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings

    for item in DshSessions.for_settings(dsh_settings())._read()["owners"].values():
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


def _target(request, owner: str | None) -> tuple[dict, list[dict]]:
    """在可选归属中找到指定归属，未指定时为当前管理账号

    参数:
        request: 当前请求
        owner: 归属键

    返回:
        tuple[dict, list[dict]]: 选中的归属与全部可选归属
    """
    from zhenxun.utils.atomic_json import AtomicJsonRecordError

    try:
        targets = _agent_targets(request)
    except AtomicJsonRecordError as exc:
        raise HTTPException(
            409,
            detail={
                "code": "dsh_mapping_invalid",
                "message": "会话映射损坏或不可读取，已阻止切换。",
            },
        ) from exc
    for target in targets:
        if owner in (None, target["owner"]):
            return target, targets
    raise HTTPException(404, detail={"message": "该超管不可选择，请刷新列表。"})


@router.get("/dsh/sessions", dependencies=[authentication()], response_model=Result)
async def desktop_sessions(request: Request, owner: str | None = None) -> Result:
    """列出可选归属，以及指定归属可接入的桌面会话"""
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
    from zhenxun.services.ai.core.exceptions import RunInterrupted
    from zhenxun.services.ai.run.dsh_control import DshControl
    from zhenxun.services.ai.run.dsh_desktop import DshUnavailable

    target, targets = _target(request, owner)
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
    current = (
        DshSessions.for_settings(dsh_settings())._read()["owners"].get(target["owner"])
    )
    return Result.ok(
        {
            "targets": [{"owner": t["owner"], "label": t["label"]} for t in targets],
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

    target, targets = _target(request, payload.owner)
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
    if target is not targets[0]:
        logger.info(
            f"WebUI 为 {target['label']} 选择了 DSH 会话 "
            f"{payload.remote_id or '（新建）'}",
            "WebUi",
        )
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
        status["sessions"] = list(
            DshSessions.for_settings(dsh_settings())._read()["owners"].values()
        )
        from zhenxun.services.ai.run.session import session_manager

        for item in status["sessions"]:
            run_id = (item.get("inflight") or {}).get("run_id")
            active = session_manager.runs.get(run_id)
            item["local_running"] = bool(active and not active[1].done())
            item["stop_state_label"] = (
                "本地转发中"
                if item["local_running"]
                else "远端前台结果待核验"
                if item.get("inflight")
                else "前台已停止或已移除（后台另行管理）"
                if item.get("stop")
                in {
                    "confirmed",
                    "foreground_confirmed",
                    "foreground_idle",
                    "target_removed",
                }
                else "下次对话将重建会话"
                if item.get("quarantined")
                else "停止结果待核验"
                if item.get("stop")
                else "无登记运行（非远端状态）"
            )
        status["mapping_trusted"] = True
    except Exception:
        status["sessions"] = []
        status["mapping_trusted"] = False
        status["reason"] = "会话映射损坏或不可读取，已阻止新运行。"
    return Result.ok(status)


class DshHostAction(BaseModel):
    mode: Literal["chat", "agent"] = "chat"
    expected_boot: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    expected_settings: DshSettings | None = None


@router.post(
    "/dsh/host/{action}", dependencies=[authentication()], response_model=Result
)
async def sdk_host_action(
    action: Literal["start", "stop"], payload: DshHostAction
) -> Result:
    """显式管理独立 SDK 宿主；不影响桌面进程或其他 DSH home。"""
    from zhenxun.builtin_plugins.ai_chat._runner import dsh_settings
    from zhenxun.services.ai.core.exceptions import RunInterrupted
    from zhenxun.services.ai.run.dsh_desktop import DshUnavailable
    from zhenxun.services.ai.run.dsh_sdk import SdkClient

    settings = dsh_settings()
    if payload.expected_settings is not None and payload.expected_settings != settings:
        raise HTTPException(409, detail={"message": "连接配置已变化，请刷新后再连接。"})
    if settings.transport != "sdk":
        raise HTTPException(409, detail={"message": "当前未选择 SDK 模式。"})
    try:
        client = (
            await SdkClient.ensure(settings, payload.mode)
            if action == "start"
            else await SdkClient.open(settings)
        )
        if action == "stop":
            if payload.expected_boot != client.endpoint["boot"]:
                raise HTTPException(
                    409, detail={"message": "宿主代际已变化，请刷新后确认。"}
                )
            result = await client.call("shutdown")
            import psutil

            for _ in range(100):
                if not psutil.pid_exists(client.endpoint["pid"]):
                    result = {"stopped": True}
                    break
                await asyncio.sleep(0.1)
        else:
            result = await client.call("status")
    except (DshUnavailable, RunInterrupted) as error:
        raise HTTPException(409, detail={"message": str(error)}) from error
    info = (
        "宿主已启动；不会重放原用户输入。"
        if action == "start"
        else "SDK 宿主已退出。"
        if result.get("stopped")
        else "已请求停止，宿主退出结果待核验，请刷新状态。"
    )
    return Result.ok(result, info=info)


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
