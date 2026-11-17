from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
import re
from typing import Any
from urllib.parse import quote, urlsplit
import uuid

from filelock import FileLock

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.utils.atomic_json import (
    _lock_path,
    _write_unlocked,
    mutate_json_locked,
    read_json_locked,
)

CARD_PATH = DATA_PATH / "qq_official" / "presentation.json"
CARD_KINDS = ("help", "welcome_friend", "welcome_group")
_welcome_attempts: dict[tuple[str, str, str], float] = {}
_editor_contacts: dict[tuple[str, str, str], dict] = {}


def remember_editor_contact(bot, event) -> None:
    """暂存当前账号收到的 QQ 联系人及短期私聊回复上下文。"""
    from .context import event_official_context

    context = event_official_context(event)
    if (
        not context
        or context.app_id != str(bot.self_id)
        or context.scene not in {"c2c", "group"}
    ):
        return
    target = context.actor_openid if context.scene == "c2c" else context.group_openid
    if not target:
        return
    now = datetime.now(timezone.utc)
    for key, value in list(_editor_contacts.items()):
        if now.timestamp() - value["seen_at"] > 86400:
            _editor_contacts.pop(key, None)
        elif value["deadline"] <= now:
            value.pop("event", None)
    key = (context.app_id, context.scene, target)
    previous = _editor_contacts.pop(key, {})
    label = (
        getattr(getattr(event, "author", None), "username", None)
        if context.scene == "c2c"
        else getattr(event, "group_name", None)
    )
    record = {
        "openid": target,
        "label": str(
            label
            or previous.get("name")
            or ("用户" if context.scene == "c2c" else "群")
        )[:80]
        + " · "
        + target[-6:],
        "name": str(label or previous.get("name") or "")[:80],
        "seen_at": now.timestamp(),
        "deadline": context.reply_deadline,
    }
    if (
        context.scene == "c2c"
        and context.source_kind == "msg_id"
        and now < context.reply_deadline
    ):
        record["event"] = event
    _editor_contacts[key] = record
    while len(_editor_contacts) > 300:
        _editor_contacts.pop(next(iter(_editor_contacts)))


def editor_contacts(app_id: str, scene: str) -> list[dict]:
    """列出本次运行见到的联系人，不导出事件内容或反推身份摘要。"""
    now = datetime.now(timezone.utc)
    items = []
    for (owner, scope, _), record in reversed(_editor_contacts.items()):
        if record["deadline"] <= now:
            record.pop("event", None)
        if (
            owner == app_id
            and scope == scene
            and now.timestamp() - record["seen_at"] <= 86400
        ):
            items.append(
                {
                    "openid": record["openid"],
                    "label": record["label"],
                    "can_test": bool(record.get("event")),
                }
            )
    return items[:100]


def editor_test_event(app_id: str, openid: str):
    """取得当前账号已确认归属且回复窗口有效的私聊事件。"""
    record = _editor_contacts.get((app_id, "c2c", openid))
    if (
        not record
        or not record.get("event")
        or record["deadline"] <= datetime.now(timezone.utc)
    ):
        raise ValueError("请接收人先私聊当前机器人一条消息，再刷新接收人。")
    return record["event"]


def digest(value: Any) -> str:
    """计算配置内容摘要。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def text_value(value: Any, name: str, limit: int, *, required: bool = True) -> str:
    """校验界面输入文本，按官方中英文字符规则计数。"""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"{name}不能为空且必须是文本。")
    if sum(1 if ord(c) < 128 else 2 for c in value) > limit:
        raise ValueError(f"{name}过长，上限为 {limit} 个字符，中文计 2 个。")
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise ValueError(f"{name}含有无效控制字符。")
    return value


def https_url(value: Any, *, optional: bool = False) -> str:
    """校验公开 HTTPS 地址，不下载远端资源。"""
    if optional and value == "":
        return ""
    raw = text_value(value, "链接", 2048)
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or any(c.isspace() for c in raw)
        or "\\" in raw
        or host in {"localhost", "localhost.localdomain"}
        or host.endswith((".localhost", ".local", ".localdomain"))
    ):
        raise ValueError("链接必须是无凭据的公网 HTTPS 地址。")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.encode("idna").decode("ascii").split(".")
        if (
            len(labels) < 2
            or re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", labels[-1])
            or any(
                not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in labels
            )
        ):
            raise ValueError("图片和链接不可使用本机或局域网地址。") from None
    else:
        if not address.is_global:
            raise ValueError("图片和链接不可使用本机或局域网地址。")
    return raw


def _items(value: Any, name: str, maximum: int) -> list[dict]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name}必须是列表，最多 {maximum} 项。")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name}包含无效项。")
    return value


def validate_menu(menu: dict) -> dict:
    """验证菜单已知字段并保留远端扩展字段。"""
    if not isinstance(menu, dict):
        raise ValueError("菜单配置必须是对象。")
    result = deepcopy(menu)
    switch_ids = set()

    def check(item: dict, child: bool = False) -> None:
        text_value(item.get("name"), "按钮名称", 14 if child else 10)
        kind = item.get("type")
        if kind == "send_message":
            text_value(item.get("send_message"), "指令", 2000)
        elif kind == "link":
            https_url(item.get("link"))
        elif kind == "menu" and not child:
            for sub in _items(item.get("sub_menu_items", []), "子菜单", 5):
                check(sub, True)
        elif kind == "switch" and not child:
            switch = item.get("switch")
            if not isinstance(switch, dict):
                raise ValueError("缺少开关配置。")
            key = text_value(switch.get("switch_id"), "开关标识", 100)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in switch_ids:
                raise ValueError(
                    "开关标识只能使用字母、数字、下划线和横线，且不能重复。"
                )
            switch_ids.add(key)
            if not isinstance(switch.get("default"), bool):
                raise ValueError("开关默认状态必须是布尔值。")
        else:
            raise ValueError("此菜单类型不支持编辑，请保留原配置并更新编辑器。")

    for item in _items(result.get("items", []), "菜单", 10):
        check(item)
    return result


def validate_panel(panel: dict) -> dict:
    """验证指令面板内容，不丢弃未知属性。"""
    if not isinstance(panel, dict):
        raise ValueError("面板配置必须是对象。")
    result = deepcopy(panel)
    for item in _items(result.get("items", []), "面板元素", 20):
        text_value(item.get("name"), "元素名称／指令", 14)
        text_value(item.get("desc", ""), "元素说明", 30, required=False)
        if item.get("type") == "link":
            https_url(item.get("link"))
        elif item.get("type") != "command":
            raise ValueError("面板仅支持指令和链接。")
        if not isinstance(item.get("only_admin", False), bool):
            raise ValueError("管理员限制必须是布尔值。")
    text_value(result.get("remark", ""), "面板备注", 255, required=False)
    return result


def empty_card() -> dict:
    """生成默认关闭的图文回复。"""
    return {
        "enabled": False,
        "title": "",
        "image_url": "",
        "body": "",
        "quote": "",
        "rows": [],
    }


def validate_cards(cards: dict) -> dict:
    """校验本地图文配置与按钮，不接受任意执行型回调。"""
    if not isinstance(cards, dict) or set(cards) != set(CARD_KINDS):
        raise ValueError("图文配置必须包含帮助、加好友欢迎和入群欢迎三项。")
    result = deepcopy(cards)
    for kind, card in result.items():
        if not isinstance(card, dict) or set(card) - set(empty_card()):
            raise ValueError("图文配置含有不支持的字段。")
        if not isinstance(card.get("enabled"), bool):
            raise ValueError("启用状态必须是布尔值。")
        for field, limit in (("title", 200), ("body", 8000), ("quote", 2000)):
            text_value(card.get(field, ""), field, limit, required=False)
        image_url = card.get("image_url", "")
        if isinstance(image_url, str):
            image_url = image_url.strip()
        card["image_url"] = https_url(image_url, optional=True)
        if card["enabled"] and not (card.get("title") or card.get("body")):
            raise ValueError("启用的卡片至少需要标题或正文。")
        ids = set()
        for row in _items(card.get("rows", []), "按钮行", 5):
            if set(row) - {"buttons"}:
                raise ValueError("按钮行含有不支持的字段。")
            for button in _items(row.get("buttons", []), "每行按钮", 5):
                allowed = {"id", "label", "data", "type", "style", "enter"}
                if set(button) - allowed:
                    raise ValueError("按钮含有不支持的字段。")
                bid = text_value(button.get("id"), "按钮标识", 64)
                if bid in ids:
                    raise ValueError("按钮标识不能重复。")
                ids.add(bid)
                text_value(button.get("label"), "按钮文字", 10)
                if button.get("type") == "link":
                    https_url(button.get("data"))
                elif button.get("type") == "command":
                    text_value(button.get("data"), "指令", 2000)
                else:
                    raise ValueError("图文按钮仅支持指令与 HTTPS 链接。")
                style = button.get("style", 1)
                if type(style) is not int or style not in (0, 1, 3, 4):
                    raise ValueError("按钮样式无效。")
                if not isinstance(button.get("enter", False), bool):
                    raise ValueError("直接发送选项无效。")
                if kind == "welcome_group" and button.get("enter"):
                    raise ValueError("群聊按钮不支持直接发送。")
    return result


def read_cards(app_id: str) -> dict:
    """读取账号图文配置，损坏记录不能当作默认配置。"""
    record = read_json_locked(CARD_PATH, {}, strict=True)
    if not isinstance(record, dict):
        raise ValueError("图文配置记录结构不可确认。")
    cards = record.get(app_id, {kind: empty_card() for kind in CARD_KINDS})
    validated = validate_cards(cards)
    return {"revision": digest(cards), "cards": validated}


def save_cards(app_id: str, revision: str, cards: dict) -> dict:
    """比较修订后原子保存账号图文配置。"""
    validated = validate_cards(cards)

    def update(record: dict) -> None:
        if not isinstance(record, dict):
            raise ValueError("图文配置记录结构不可确认。")
        current = record.get(app_id, {kind: empty_card() for kind in CARD_KINDS})
        validate_cards(current)
        if digest(current) == digest(validated):
            return
        if digest(current) != revision:
            raise ValueError("图文配置已被其他页面修改，请重新读取并比较草稿。")
        record[app_id] = validated

    mutate_json_locked(CARD_PATH, {}, update, strict=True)
    return {"revision": digest(validated), "cards": validated}


def recover_cards(
    app_id: str, revision: str | None = None, cards: dict | None = None
) -> dict:
    """读取损坏配置的修复基线，或在备份原文件后保存明确确认的修复。"""
    validated = validate_cards(cards) if cards is not None else None
    with FileLock(_lock_path(CARD_PATH), timeout=5):
        raw = CARD_PATH.read_bytes()
        current = hashlib.sha256(raw).hexdigest()
        try:
            record = json.loads(raw)
        except (ValueError, UnicodeError):
            record = None
        whole_file = not isinstance(record, dict)
        if not whole_file:
            try:
                validate_cards(
                    record.get(app_id, {kind: empty_card() for kind in CARD_KINDS})
                )
            except ValueError:
                pass
            else:
                raise ValueError("图文配置已可读取，请重新读取后保存。")
        recovery = {
            "scope": "file" if whole_file else "account",
            "message": (
                "整个图文文件无法解析；修复会备份原文件并重建当前账号配置，其他账号需要从备份恢复。"
                if whole_file
                else "当前账号图文配置损坏；修复会备份原文件，保留其他账号配置。"
            ),
        }
        if validated is None:
            return {
                "revision": current,
                "cards": {kind: empty_card() for kind in CARD_KINDS},
                "recovery": recovery,
            }
        if revision != current:
            raise ValueError("图文文件已被其他页面或进程修改，请重新读取。")
        backup = CARD_PATH.with_name(
            f"{CARD_PATH.stem}.recovery-{uuid.uuid4().hex}.json"
        )
        with backup.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if whole_file:
            record = {}
        record[app_id] = validated
        _write_unlocked(CARD_PATH, record)
    return {"revision": digest(validated), "cards": validated, "backup": backup.name}


def build_card_message(card: dict, *, private: bool):
    """把结构化卡片转换为 QQ 原生 Markdown 与内联按钮。"""
    from nonebot.adapters.qq.message import Message, MessageSegment
    from nonebot.adapters.qq.models import MessageKeyboard

    def plain(value: str) -> str:
        return re.sub(r"([\\*_\[\]()#<>])", r"\\\1", value)

    parts = []
    if card.get("image_url"):
        image = quote(card["image_url"], safe=":/?=&%#@+~!$,-._")
        parts.append(f"![图片 #600px #300px]({image})")
    if card.get("title"):
        parts.append("## " + plain(card["title"]))
    if card.get("body"):
        parts.append(plain(card["body"]))
    if card.get("quote"):
        parts.append(
            "\n".join("> " + plain(line) for line in card["quote"].splitlines())
        )
    message = Message(MessageSegment.markdown("\n\n".join(parts)))
    rows = []
    for row in card.get("rows", []):
        buttons = []
        for item in row.get("buttons", []):
            buttons.append(
                {
                    "id": item["id"],
                    "render_data": {
                        "label": item["label"],
                        "style": item.get("style", 1),
                    },
                    "action": {
                        "type": 0 if item["type"] == "link" else 2,
                        "data": item["data"],
                        "permission": {"type": 2},
                        "enter": bool(private and item.get("enter", False)),
                    },
                }
            )
        if buttons:
            rows.append({"buttons": buttons})
    if rows:
        message += MessageSegment.keyboard(MessageKeyboard(content={"rows": rows}))
    return message


async def send_configured_card(bot, event, kind: str) -> bool:
    """发送已启用卡片；返回是否已处理，未知投递不允许重复补发。"""
    from nonebot.adapters.qq import Bot
    from nonebot.adapters.qq.exception import ActionFailed

    from zhenxun.services.log import logger

    from .context import OfficialReplyUnavailable, event_official_context

    if not isinstance(bot, Bot):
        return False
    context = event_official_context(event)
    if not context or context.scene not in {"c2c", "group"}:
        return False
    try:
        config = await asyncio.to_thread(read_cards, str(bot.self_id))
    except Exception as error:
        logger.warning(f"QQ 图文配置读取失败: {type(error).__name__}", "QQOfficial")
        return False
    card = config["cards"].get(kind, {})
    if not card.get("enabled"):
        return False
    now = datetime.now(timezone.utc)
    if now >= context.reply_deadline:
        logger.warning("QQ 图文回复窗口已过期，未转为主动发送", "QQOfficial")
        return False
    try:
        message = build_card_message(card, private=context.scene == "c2c")
    except Exception as error:
        logger.warning(
            f"QQ 图文构造失败，尚未发送: {type(error).__name__}", "QQOfficial"
        )
        return False
    if kind != "help":
        key = (str(bot.self_id), context.source_id, kind)
        if key in _welcome_attempts:
            return True
        for old, deadline in list(_welcome_attempts.items()):
            if deadline < now.timestamp():
                _welcome_attempts.pop(old, None)
        if len(_welcome_attempts) >= 512:
            logger.warning("QQ 欢迎事件缓存忙，本次未发送", "QQOfficial")
            return True
        _welcome_attempts[key] = context.reply_deadline.timestamp()
    try:
        await bot.send(event, message, passive_only=True)
        return True
    except OfficialReplyUnavailable as error:
        logger.warning(f"QQ 图文未发送: {type(error).__name__}", "QQOfficial")
        return False
    except ActionFailed as error:
        logger.warning(
            f"QQ 图文发送失败 code={error.code} status={error.status_code}",
            "QQOfficial",
        )
        if str(error.code) in {"40054002", "304103", "40034005", "40034026"}:
            return True
        return not (error.status_code and 400 <= error.status_code < 500)
    except Exception as error:
        logger.warning(f"QQ 图文投递结果未知: {type(error).__name__}", "QQOfficial")
        return True
