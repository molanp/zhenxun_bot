"""长期记忆区段的准备、模型整理与事务提交。"""

import asyncio
import hashlib
import json
import time
import uuid

from tortoise.expressions import F
from tortoise.timezone import now
from tortoise.transactions import in_transaction

from zhenxun.configs.config import BotConfig
from zhenxun.models.ai_agent_run import AIAgentRun
from zhenxun.models.ai_chat_message import AIChatMessage
from zhenxun.models.ai_memory import AIMemory
from zhenxun.models.ai_memory_cursor import AIMemoryCursor
from zhenxun.services.ai.core.templates import wrap_untrusted
from zhenxun.services.ai.run.session import session_manager
from zhenxun.services.log import logger
from zhenxun.utils.time_utils import TimeUtils

from ._data_source import parse_session


async def summarize(
    cls, session_id: str, start: int = 0, *, retry: bool = False
) -> AIMemory | None:
    """处理一个有界区段，在短事务内提交摘要与游标。"""
    from zhenxun.services.ai.core.engine.token_counter import parse_usage_info
    from zhenxun.services.ai.llm.api import chat
    from zhenxun.services.ai.llm.manager import get_default_model

    from ._memory import (
        MAX_CHAT_CHARS,
        MAX_ROWS,
        MIN_IMPORTANCE,
        RETRY_AFTER,
        SPLIT_GAP,
        SUMMARY_PROMPT,
        _cfg,
        _message_text,
        _parse_json,
        _strings,
        memory_mode,
        session_scope,
    )
    from ._safety import Safety

    scope = session_scope(session_id)
    lock = cls._locks.setdefault(session_id, asyncio.Lock())
    if scope is None or lock.locked() or memory_mode() == "off":
        return None
    epoch = session_manager.revision(session_id)
    cls._summary_tasks[session_id] = asyncio.current_task()
    try:
        async with lock, cls._slots:
            cls._running.add(session_id)
            cls._summary_tasks[session_id] = asyncio.current_task()
            try:
                cursor, _ = await AIMemoryCursor.get_or_create(session_id=session_id)
                state = dict(cursor.state or {})
                if not retry and (
                    state.get("status") == "pending"
                    or state.get("next_retry_at", 0) > time.time()
                ):
                    return None
                rows = (
                    await AIChatMessage.filter(
                        session_id=session_id,
                        is_summary=False,
                        id__gt=cursor.message_id,
                    )
                    .order_by("id")
                    .limit(MAX_ROWS)
                )
                selected, lines, total = [], [], 0
                end_id, partial_id, offset = cursor.message_id, 0, 0
                interrupted = set()
                for row in rows:
                    meta = row.metadata or {}
                    unsettled = meta.get("chat_run") and not meta.get("chat_settled")
                    if unsettled and meta["chat_run"] not in interrupted:
                        record = await AIAgentRun.get_or_none(
                            run_id=meta["chat_run"],
                            session_id=session_id,
                            status="interrupted",
                        )
                        if (
                            not record
                            or (record.details or {}).get("interruption")
                            != "previous_boot"
                        ):
                            break
                        interrupted.add(meta["chat_run"])
                    if selected and (
                        (row.created_at - selected[-1].created_at).total_seconds()
                        > SPLIT_GAP
                        or row.created_at.astimezone(TimeUtils.zone()).date()
                        != selected[-1].created_at.astimezone(TimeUtils.zone()).date()
                    ):
                        break
                    text = (
                        _message_text(row.role, row.content)
                        if row.role in {"user", "assistant"}
                        else ""
                    )
                    if unsettled or (
                        meta.get("chat_run") and not meta.get("chat_visible")
                    ):
                        text = ""
                    if (
                        row.role == "assistant"
                        and meta.get("chat_run")
                        and not meta.get("chat_final")
                    ):
                        text = ""
                    used = (
                        int(state.get("offset", 0))
                        if state.get("partial_id") == row.id
                        else 0
                    )
                    text = text[used:]
                    label = (
                        f"[{row.created_at.astimezone(TimeUtils.zone()):%m-%d %H:%M}] "
                        + ("你：" if row.role == "assistant" else "")
                    )
                    room = MAX_CHAT_CHARS - total - len(label) - 1
                    if text and room <= 0:
                        break
                    selected.append(row)
                    if text:
                        take = text[:room]
                        lines.append(label + take)
                        total += len(label) + len(take) + 1
                        if len(take) < len(text):
                            partial_id, offset = row.id, used + len(take)
                            break
                    end_id = row.id
                if not selected:
                    return None
                signature = hashlib.sha256(
                    json.dumps(
                        [(x.id, x.content, x.metadata) for x in selected],
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                attempts = 0 if retry else int(state.get("attempts", 0))
                state.update(
                    status="running",
                    attempts=attempts + 1,
                    error="",
                    next_retry_at=0,
                )
                claimed = await AIMemoryCursor.filter(
                    id=cursor.id, revision=cursor.revision
                ).update(revision=F("revision") + 1, state=state, updated_at=now())
                if not claimed:
                    return None
                revision = cursor.revision + 1
                model = str(_cfg("MEMORY_MODEL", "")).strip() or None
                begun = time.monotonic()
                status, output, error, tokens, memory = "silent", "", "", 0, None
                try:
                    parsed = {"summary": "", "importance": 0}
                    if lines:
                        name = (
                            next(
                                (
                                    x.metadata.get("persona")
                                    for x in reversed(selected)
                                    if x.metadata and x.metadata.get("persona")
                                ),
                                None,
                            )
                            or BotConfig.self_nickname
                            or "Bot"
                        )
                        prompt = SUMMARY_PROMPT.format(
                            name=name,
                            scene=f"群 {scope[0]}" if scope[0] else "私聊",
                            start=selected[0].created_at.isoformat(),
                            end=selected[-1].created_at.isoformat(),
                            chat=wrap_untrusted("\n".join(lines), "chat_log"),
                        )
                        response = await chat(prompt, model=model, timeout=120)
                        output = response.text
                        tokens = parse_usage_info(response.usage_info).total_tokens
                        parsed = _parse_json(output)
                    summary = parsed["summary"].strip()
                    should_save = bool(
                        summary and parsed["importance"] >= MIN_IMPORTANCE
                    )
                    summary, blocked = Safety.filter_output(summary)
                    if blocked:
                        raise ValueError("摘要包含不能保存的内容")
                    vector = (await cls._embed([summary]))[0] if should_save else None
                    async with in_transaction() as connection:
                        current = (
                            await AIMemoryCursor.filter(id=cursor.id)
                            .using_db(connection)
                            .select_for_update()
                            .first()
                        )
                        fresh = (
                            await AIChatMessage.filter(
                                id__in=[x.id for x in selected],
                                session_id=session_id,
                            )
                            .using_db(connection)
                            .order_by("id")
                        )
                        digest = hashlib.sha256(
                            json.dumps(
                                [(x.id, x.content, x.metadata) for x in fresh],
                                ensure_ascii=False,
                                sort_keys=True,
                            ).encode()
                        ).hexdigest()
                        if (
                            current is None
                            or current.revision != revision
                            or epoch != session_manager.revision(session_id)
                            or memory_mode() == "off"
                            or digest != signature
                        ):
                            raise asyncio.CancelledError("整理期间会话已改变")
                        if should_save:
                            participants = list(
                                dict.fromkeys(
                                    x.metadata.get("sender")
                                    for x in selected
                                    if x.metadata and x.metadata.get("sender")
                                )
                            )
                            memory, _ = await cls.add(
                                scope,
                                summary,
                                _strings(parsed.get("topics"), 5),
                                _strings(parsed.get("key_facts"), 8),
                                participants,
                                parsed["importance"],
                                created_at=selected[-1].created_at,
                                vector=vector,
                                embedded=True,
                                using_db=connection,
                            )
                            status = "ok"
                        current.message_id = end_id
                        current.state = {
                            "status": "ready",
                            "attempts": 0,
                            "partial_id": partial_id,
                            "offset": offset,
                        }
                        current.revision += 1
                        await current.save(
                            using_db=connection,
                            update_fields=[
                                "message_id",
                                "state",
                                "revision",
                                "updated_at",
                            ],
                        )
                    cls.invalidate(scope)
                except asyncio.CancelledError:
                    status, error = "cancelled", "整理已中止，未提交区段已保留"
                    if epoch == session_manager.revision(session_id):
                        state.update(
                            status="pending" if attempts + 1 >= 3 else "retry_wait",
                            error=error,
                            next_retry_at=time.time() + RETRY_AFTER,
                        )
                        await AIMemoryCursor.filter(
                            id=cursor.id, revision=revision
                        ).update(state=state, updated_at=now())
                    raise
                except Exception as exc:
                    status, error = "error", f"整理未完成（{type(exc).__name__}）"
                    state.update(
                        status="pending" if attempts + 1 >= 3 else "retry_wait",
                        error=error,
                        next_retry_at=time.time() + RETRY_AFTER,
                    )
                    await AIMemoryCursor.filter(id=cursor.id, revision=revision).update(
                        state=state, updated_at=now()
                    )
                finally:
                    try:
                        await AIAgentRun.create(
                            run_id=uuid.uuid4().hex,
                            source="memory",
                            session_id=session_id,
                            user_id=parse_session(session_id)[0],
                            group_id=scope[0],
                            model=model or get_default_model("chat"),
                            input=f"整理区段 {selected[0].id}-{selected[-1].id}",
                            output=Safety.filter_output(output)[0][:4000],
                            tokens=tokens,
                            status=status,
                            error=error,
                            duration=int((time.monotonic() - begun) * 1000),
                            details={
                                "attempt": attempts + 1,
                                "start": selected[0].id,
                                "end": selected[-1].id,
                            },
                        )
                    except Exception as exc:
                        logger.warning(
                            f"整理运行记录保存失败: {type(exc).__name__}",
                            "ai_chat",
                        )
                return memory
            finally:
                cls._running.discard(session_id)
                cls._summary_tasks.pop(session_id, None)
    finally:
        if cls._summary_tasks.get(session_id) is asyncio.current_task():
            cls._summary_tasks.pop(session_id, None)
