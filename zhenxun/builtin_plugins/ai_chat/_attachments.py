from contextlib import contextmanager
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile

from nonebot_plugin_alconna import File, Image, UniMessage

from zhenxun.configs.path_config import TEMP_PATH
from zhenxun.services.log import logger
from zhenxun.utils.platform import PlatformUtils

from ._reply import DeliveryResult, _deliver, delivery_target
from ._safety import Safety

MAX_BYTES = 8 * 1024 * 1024


@contextmanager
def export_slot(context, binding):
    """仅为本轮 OneBot 定向 Agent 分配可写文件，始终从原文件描述符读取。"""
    caller = context.run.caller
    if not (
        caller
        and caller.source == "qq"
        and caller.platform == "qq_client"
        and binding.get("mode") == "agent"
        and binding.get("native")
    ):
        yield None
        return
    cwd = binding.get("cwd")
    if (
        not cwd
        and binding.get("sdk_home")
        and re.fullmatch(r"zx-[0-9a-f]{32}", binding.get("remote_id", ""))
    ):
        cwd = Path(binding["sdk_home"]) / "agent" / "workspace" / binding["remote_id"]
    try:
        if cwd and not binding.get("cwd"):
            cwd.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=".zx-export-", dir=cwd)
    except OSError:
        logger.warning("本轮无法创建附件导出文件", "ai_chat")
        yield None
        return
    try:
        with os.fdopen(fd, "rb") as stream:
            yield stream, path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("本轮附件临时文件未能清理", "ai_chat")


def export_instruction(path):
    """生成该轮附件的显式导出约定。"""
    return (
        "仅当当前用户明确要求发文件或图片时，将一份成品的原始字节写入已有临时文件 "
        f"{path}（原地写入，不要重命名替换，最多 8 MiB）。成功后最终回复最后一行仅写 "
        '{"qq_attachment":"展示文件名.扩展名"}；由机器人实际投递。'
        "不要输出临时路径、不要声称已经发送。无附件时照常回答。"
    )


def collect_export(context, slot, text):
    """校验交付声明并截取有限字节；文件名只用于展示，不用于读取宿主路径。"""
    stream, path = slot
    line = text.strip().rsplit("\n", 1)[-1]
    declared = line.startswith("{") and '"qq_attachment"' in line
    size = os.fstat(stream.fileno()).st_size
    if not size and not declared:
        return text.replace(path, "[临时文件]")
    try:
        obj = json.loads(line) if declared else {}
        name = obj["qq_attachment"]
        if not (
            set(obj) == {"qq_attachment"}
            and isinstance(name, str)
            and re.fullmatch(r"[\w\-][\w .()\-]{0,95}", name)
            and name[-1] not in ". "
            and not PureWindowsPath(name).is_reserved()
            and not Safety.filter_output(name)[1]
        ):
            raise ValueError
        if not 0 < size <= MAX_BYTES:
            raise ValueError
        data = stream.read(MAX_BYTES + 1)
        if len(data) != size:
            raise ValueError
    except (KeyError, TypeError, ValueError, OSError):
        context.run.operations["attachment"] = {"status": "invalid"}
        return "附件未发送：请导出一个不超过 8 MiB 的非空文件，并提供有效文件名。"
    context.state["_qq_attachment"] = (name, data)
    return f"[附件：{name}]"


async def send_export(context, payload, result: DeliveryResult):
    """在当前调用者的目标内投递一次，图片传字节，文件使用独立上传快照。"""
    ops = context.run.operations
    if "attachment" in ops:
        return
    name, data = payload
    record = ops["attachment"] = {"name": name, "status": "failed"}
    bot, caller = context.get_bot(), context.run.caller
    check = context.state.get("_dsh_authorization_check")
    if not (
        caller
        and caller.source == "qq"
        and bot
        and caller.bot_id == bot.self_id
        and PlatformUtils.get_platform_scope(bot) == caller.platform == "qq_client"
        and check
        and await check() == "agent"
    ):
        result.status, result.error = "failed", "当前会话无附件发送权限"
        return

    async def send(message, first):
        return await PlatformUtils.send_message(
            bot, caller.user_id, caller.group_id, message
        )

    from contextlib import ExitStack

    with ExitStack() as cleanup:
        try:
            if data.startswith(
                (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")
            ) or (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
                segment = Image(raw=data, name=name)
            else:
                folder = cleanup.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix="zx-qq-upload-",
                        dir=TEMP_PATH,
                        ignore_cleanup_errors=True,
                    )
                )
                path = Path(folder) / name
                path.write_bytes(data)
                segment = File(path=path, name=name)
        except OSError:
            result.status, result.error = "failed", "无法准备临时上传文件"
            return
        try:
            await _deliver(
                f"[附件：{name}]",
                bot,
                send,
                result=result,
                context=context,
                target=delivery_target(bot, caller.group_id, caller.user_id),
                chunks=[(UniMessage(segment), f"[附件：{name}]")],
            )
        finally:
            record["status"] = (
                "unknown"
                if result.uncertain
                else "sent"
                if result.status == "sent"
                else "failed"
            )
