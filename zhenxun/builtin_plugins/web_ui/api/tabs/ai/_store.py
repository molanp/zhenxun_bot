"""AI 配置文件的读写、校验与页面视图"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import cache
import hashlib
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from zhenxun.services.ai.config import PRESETS, PROTOCOLS, get_llm_config
from zhenxun.services.ai.config.models import (
    LLMConfig,
    ModelConfig,
    ProviderConfig,
    is_placeholder_key,
)
from zhenxun.services.ai.config.presets import detect_preset
from zhenxun.services.ai.core.models import ModelProfile
from zhenxun.services.ai.llm.capability import (
    check_status,
    get_observation,
    record_observation,
)
from zhenxun.services.ai.llm.discovery import get_discovery, get_snapshot
from zhenxun.services.ai.llm.profile import catalog_status, resolve_profile
from zhenxun.services.plugin_host import RuntimeOperation
from zhenxun.services.runtime_config import reload_runtime_config
from zhenxun.utils._restart_utils import issue_restart_ticket
from zhenxun.utils.pydantic_compat import model_dump, model_json_schema, parse_as

from ....apply_result import (
    APPLY_NO_CHANGE,
    APPLY_RESTART_PENDING,
    apply_result_data,
    update_pending_restart,
)
from ....config_validation import _yaml_parser
from ....restart_service import restart_status_data
from ...configure.persistence import _write_transaction

CONFIG_FILE = Path("data/config.yaml")
_PERSIST_LOCK = asyncio.Lock()


def read_config() -> str:
    return CONFIG_FILE.read_text(encoding="utf-8") if CONFIG_FILE.exists() else ""


def config_revision(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_config(content: str) -> dict[str, Any]:
    data = _yaml_parser().load(StringIO(content)) or {}
    if not isinstance(data, dict):
        raise configuration_error(
            "yaml_top_level_mapping_required", "config.yaml 顶层必须是映射。"
        )
    ai = data.setdefault("AI", {})
    if not isinstance(ai, dict):
        raise configuration_error("ai_mapping_required", "AI 配置组必须是映射。", "AI")
    return data


def dump_config(data: dict[str, Any]) -> str:
    stream = StringIO()
    _yaml_parser().dump(data, stream)
    return stream.getvalue()


def configuration_error(
    code: str, message: str, path: str | None = None
) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "message": "AI 配置校验失败。",
            "issues": [
                {
                    "code": code,
                    "file": "config.yaml",
                    "path": path,
                    "line": None,
                    "column": None,
                    "severity": "error",
                    "message": message,
                }
            ],
        },
    )


def existing_key(ai: dict[str, Any], key: str) -> str | None:
    expected = key.casefold()
    return next((str(item) for item in ai if str(item).casefold() == expected), None)


def ai_get(ai: dict[str, Any], key: str, default: Any = None) -> Any:
    existing = existing_key(ai, key)
    return ai[existing] if existing is not None else default


def ai_set(ai: dict[str, Any], key: str, value: Any) -> None:
    ai[existing_key(ai, key) or key] = value


def _llm_payload(ai: dict[str, Any]) -> dict[str, Any]:
    current = get_llm_config()
    return {
        "default_models": ai_get(
            ai, "default_models", model_dump(current.default_models)
        ),
        "client_settings": ai_get(
            ai, "client_settings", model_dump(current.client_settings)
        ),
        "debug_log": ai_get(ai, "debug_log", model_dump(current.debug_log)),
        "providers": ai_get(
            ai, "PROVIDERS", [model_dump(item) for item in current.providers]
        ),
        "context_settings": ai_get(
            ai, "context_settings", model_dump(current.context_settings)
        ),
        "model_groups": ai_get(ai, "MODEL_GROUPS", current.model_groups),
        "agent_settings": ai_get(
            ai, "agent_settings", model_dump(current.agent_settings)
        ),
        "sandbox": ai_get(ai, "sandbox", model_dump(current.sandbox)),
        "web_search": ai_get(ai, "web_search", model_dump(current.web_search)),
        "deepseek_tools": ai_get(
            ai, "deepseek_tools", model_dump(current.deepseek_tools)
        ),
    }


def draft_profile(provider: ProviderConfig, model: ModelConfig) -> ModelProfile:
    """解析配置草稿中模型的能力档案

    参数:
        provider: 服务商配置
        model: 模型配置

    返回:
        ModelProfile: 能力档案
    """
    return resolve_profile(
        provider,
        model,
        get_snapshot(provider, model.name),
        get_observation(provider, model),
    )


def reference_issues(config: LLMConfig) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    model_details = {
        f"{provider.name}/{model.name}": (provider, model)
        for provider in config.providers
        for model in provider.models
    }
    available_models = set(model_details)
    group_names = set(config.model_groups)

    def add(code: str, path: str, message: str) -> None:
        issue = {"code": code, "path": path, "message": message}
        if issue not in issues:
            issues.append(issue)

    def visit(group: str, stack: tuple[str, ...]) -> None:
        if group in stack:
            add(
                "model_group_cycle",
                "AI.MODEL_GROUPS",
                f"模型路由组 {' -> '.join((*stack, group))} 存在循环引用。",
            )
            return
        targets = config.model_groups.get(group, [])
        for index, target in enumerate(targets):
            if target in group_names:
                visit(target, (*stack, group))
            elif target not in available_models:
                add(
                    "model_reference_missing",
                    f"AI.MODEL_GROUPS.{group}.{index}",
                    f"模型路由组 {group} 引用了不存在的模型 {target}。",
                )

    for group_name in group_names:
        targets = config.model_groups.get(group_name, [])
        if len(targets) != len(set(targets)):
            add(
                "model_group_target_duplicate",
                f"AI.MODEL_GROUPS.{group_name}",
                f"模型路由组 {group_name} 包含重复目标。",
            )
        visit(group_name, ())

    def model_supports(target: str, task: str) -> bool:
        found = model_details.get(target)
        if found is None:
            return False
        profile = draft_profile(*found)
        if task == "vision":
            return profile.supports_task("chat") and profile.accepts_input("image")
        return profile.supports_task(task)

    def target_supports(target: str, task: str, seen: set[str]) -> bool:
        if target in model_details:
            return model_supports(target, task)
        if target not in group_names or target in seen:
            return False
        return any(
            target_supports(child, task, {*seen, target})
            for child in config.model_groups.get(target, [])
        )

    for task, target in model_dump(config.default_models).items():
        if target and target not in available_models and target not in group_names:
            add(
                "default_model_missing",
                f"AI.default_models.{task}",
                f"{task} 默认模型 {target} 不存在。",
            )
        elif target and not target_supports(target, task, set()):
            if target in model_details:
                label = f"模型 {target}"
            else:
                label = f"路由组 {target}"
            if target in available_models or target in group_names:
                add(
                    "default_model_capability_mismatch",
                    f"AI.default_models.{task}",
                    f"{label}中没有支持 {task} 任务的可用模型。",
                )
    return issues


def raise_reference_issue(issue: dict[str, str]) -> None:
    raise configuration_error(issue["code"], issue["message"], issue["path"])


def validate_ai(ai: dict[str, Any], *, strict_references: bool = True) -> LLMConfig:
    try:
        config = parse_as(LLMConfig, _llm_payload(ai))
    except Exception as error:
        errors = error.errors() if callable(getattr(error, "errors", None)) else []
        path = (
            ".".join(str(item) for item in errors[0].get("loc", [])) if errors else "AI"
        )
        raise configuration_error(
            "ai_field_invalid", "字段类型或取值无效。", path
        ) from error

    provider_names: set[str] = set()
    for provider in config.providers:
        normalized = provider.name.strip().casefold()
        if not normalized or "/" in provider.name:
            raise configuration_error(
                "provider_name_invalid",
                "服务商名称不能为空或包含斜杠。",
                "AI.PROVIDERS",
            )
        if normalized in provider_names:
            raise configuration_error(
                "provider_name_duplicate",
                f"服务商名称 {provider.name} 重复。",
                "AI.PROVIDERS",
            )
        provider_names.add(normalized)
        _check_url(provider.base_url, f"服务商 {provider.name} 的接口地址")
        model_names: set[str] = set()
        for model in provider.models:
            if not model.name.strip():
                raise configuration_error(
                    "model_name_empty",
                    f"服务商 {provider.name} 存在空的模型名称。",
                    "AI.PROVIDERS",
                )
            if model.name in model_names:
                raise configuration_error(
                    "model_name_duplicate",
                    f"服务商 {provider.name} 存在重复模型 {model.name}。",
                    "AI.PROVIDERS",
                )
            model_names.add(model.name)
            if model.base_url:
                label = f"模型 {provider.name}/{model.name} 的接口地址"
                _check_url(model.base_url, label)
    issues = reference_issues(config)
    if strict_references and issues:
        raise_reference_issue(issues[0])
    return config


def _check_url(url: str, label: str) -> None:
    """校验接口地址为 http(s) URL

    参数:
        url: 接口地址
        label: 错误提示中的字段描述
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise configuration_error(
            "provider_url_invalid", f"{label}无效或未填写。", "AI.PROVIDERS"
        )


def provider_index(providers: Any, name: str) -> int | None:
    """按名称查找服务商下标，名称不区分大小写

    参数:
        providers: 原始服务商列表
        name: 服务商名称

    返回:
        int | None: 下标，未找到时为 None
    """
    return next(
        (
            index
            for index, item in enumerate(providers or [])
            if isinstance(item, dict)
            and str(item.get("name", "")).casefold() == name.strip().casefold()
        ),
        None,
    )


def find_provider(ai: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    providers = ai_get(ai, "PROVIDERS", [])
    index = provider_index(providers, name)
    if index is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "provider_not_found", "message": "未找到该 AI 服务商。"},
        )
    return index, providers[index]


def clean_keys(value: Any) -> list[str]:
    """整理 API Key 列表，去掉空值与占位符

    参数:
        value: 单个密钥或密钥列表

    返回:
        list[str]: 有效密钥
    """
    candidates = [value] if isinstance(value, str) else value or []
    return [str(item).strip() for item in candidates if not is_placeholder_key(item)]


async def persist(
    expected_revision: str,
    mutate: Any,
    *,
    reload_consumers: bool = True,
) -> tuple[str, LLMConfig, RuntimeOperation | None]:
    """校验修订号后修改 AI 配置并热加载，失败时恢复原文件

    参数:
        expected_revision: 前端持有的配置修订号
        mutate: 修改 AI 配置组的函数
        reload_consumers: 是否重载配置消费者

    返回:
        tuple[str, LLMConfig, RuntimeOperation | None]: 新修订号、配置与重载结果
    """
    async with _PERSIST_LOCK:
        current = read_config()
        if config_revision(current) != expected_revision:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "configuration_revision_conflict",
                    "message": "配置文件已被外部修改，请重新加载。",
                },
            )
        data = load_config(current)
        before = validate_ai(data["AI"], strict_references=False)
        baseline = {
            (item["code"], item["path"], item["message"])
            for item in reference_issues(before)
        }
        candidate = deepcopy(data)
        mutate(candidate["AI"])
        config = validate_ai(candidate["AI"], strict_references=False)
        new_issues = [
            item
            for item in reference_issues(config)
            if (item["code"], item["path"], item["message"]) not in baseline
        ]
        if new_issues:
            raise_reference_issue(new_issues[0])
        content = dump_config(candidate)
        if content == current:
            return config_revision(content), config, None
        original = CONFIG_FILE.read_bytes() if CONFIG_FILE.exists() else None
        try:
            _write_transaction([(CONFIG_FILE, content.encode("utf-8"))])
            operation = await reload_runtime_config(
                submit_restart=False, reload_consumers=reload_consumers
            )
        except Exception as error:
            if original is None:
                CONFIG_FILE.unlink(missing_ok=True)
            else:
                _write_transaction([(CONFIG_FILE, original)])
            try:
                await reload_runtime_config(submit_restart=False)
            except Exception:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "ai_reload_failed",
                    "message": "AI 配置保存或热加载失败，已恢复原配置。",
                },
            ) from error
        mark_saved_models_skipped(before, config)
        return config_revision(content), config, operation


def _model_scope(provider: ProviderConfig, model: ModelConfig) -> tuple[str, str]:
    return (
        model.protocol or provider.protocol,
        (model.base_url or provider.base_url).rstrip("/"),
    )


def mark_saved_models_skipped(before: LLMConfig, after: LLMConfig) -> None:
    """在 WebUI 中新增或修改了协议、地址的模型记为跳过核验，启动时不再自动核验

    参数:
        before: 保存前的 AI 配置
        after: 保存后的 AI 配置
    """
    scopes = {
        (provider.name, model.name): _model_scope(provider, model)
        for provider in before.providers
        for model in provider.models
    }
    for provider in after.providers:
        for model in provider.models:
            key = (provider.name, model.name)
            if scopes.get(key) != _model_scope(provider, model):
                record_observation(provider, model, skipped=True)


def apply_response(
    config: LLMConfig,
    revision: str,
    operation: RuntimeOperation | None,
    *,
    apply_mode: str | None = None,
    changed_keys: list[str] | None = None,
    reason_codes: list[str] | None = None,
    pending_source: str = "webui.ai",
) -> dict[str, Any]:
    """生成保存结果，需要重启时登记待重启原因

    参数:
        config: 保存后的 AI 配置
        revision: 新修订号
        operation: 配置重载结果
        apply_mode: 指定的生效方式，为空时取重载结果
        changed_keys: 指定的变更键，为空时取重载结果
        reason_codes: 指定的重启原因，为空时取重载结果
        pending_source: 待重启原因的来源

    返回:
        dict[str, Any]: 生效结果与最新配置视图
    """
    mode = apply_mode or (
        operation.mode.value if operation is not None else APPLY_NO_CHANGE
    )
    required = mode == APPLY_RESTART_PENDING
    if reason_codes is None:
        reason_codes = [operation.reason] if operation and operation.reason else []
    if changed_keys is None:
        changed_keys = operation.config_keys if operation is not None else []
    if required and update_pending_restart(
        pending_source,
        reason_codes or ["ai_runtime_restart_required"],
        config_keys=changed_keys,
        consumers=operation.changed if operation else [],
    ):
        issue_restart_ticket("webui.settings", ttl_seconds=10 * 60)
    status = restart_status_data()
    return apply_result_data(
        restart_source=pending_source,
        apply_mode=mode,
        changed_keys=changed_keys,
        restart_required=required,
        reason_codes=reason_codes,
        access_urls=status["access_urls"],
        access_targets=status["access_targets"],
        **configuration_view(config, revision),
    )


def discovered_profile(
    provider: ProviderConfig, name: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """生成发现列表中单个模型的能力摘要

    参数:
        provider: 服务商配置
        name: 模型名称
        snapshot: 模型发现返回的能力字段

    返回:
        dict[str, Any]: 任务、输入模态、推理、上下文、能力库是否收录与可用协议
    """
    profile = resolve_profile(provider, ModelConfig(name=name), snapshot)
    return {
        "tasks": sorted(profile.tasks),
        "input_modalities": sorted(profile.input_modalities),
        "reasoning": bool(profile.reasoning_levels or profile.reasoning_off),
        "context_window": profile.context_window,
        "catalog": "catalog" in profile.sources.values(),
        "protocols": snapshot.get("protocols"),
    }


def _discovery_view(provider: ProviderConfig) -> dict[str, Any] | None:
    """读取服务商最近一次模型发现的结果，供前端对比已配置的模型

    参数:
        provider: 服务商配置

    返回:
        dict[str, Any] | None: 发现时间与上游模型的能力摘要，未发现过时为 None
    """
    discovery = get_discovery(provider)
    if not discovery:
        return None
    return {
        "updated_at": discovery.get("updated_at"),
        "profiles": {
            name: discovered_profile(provider, name, snapshot)
            for name, snapshot in sorted(discovery["models"].items())
        },
    }


def _provider_view(
    provider: ProviderConfig, raw_provider: dict[str, Any] | None = None
) -> dict[str, Any]:
    """生成服务商视图，包含密钥、模型能力档案与自动发现状态

    参数:
        provider: 服务商配置
        raw_provider: 配置文件中的原始服务商数据，用于保留未声明的模型字段

    返回:
        dict[str, Any]: 服务商视图
    """
    keys = clean_keys(provider.api_keys)
    raw_models = {
        str(item.get("name")): item
        for item in (raw_provider or {}).get("models") or []
        if isinstance(item, dict) and item.get("name")
    }
    if not provider.base_url:
        discovery_status = "missing_base"
    elif not keys:
        discovery_status = "missing_credentials"
    else:
        discovery_status = "ready"
    return {
        **model_dump(provider),
        "api_keys": keys,
        "models": [
            {
                **jsonable_encoder(raw_models.get(model.name, {})),
                **model_dump(model),
                "resolved_profile": jsonable_encoder(
                    model_dump(draft_profile(provider, model))
                ),
                "check": check_status(provider, model),
            }
            for model in provider.models
        ],
        "discovery_status": discovery_status,
        "discovery": _discovery_view(provider),
        "official": detect_preset(provider.base_url),
    }


@cache
def _schema() -> dict[str, Any]:
    return model_json_schema(LLMConfig)


def configuration_view(config: LLMConfig, revision: str) -> dict[str, Any]:
    """生成 AI 配置页面数据

    参数:
        config: AI 配置
        revision: 配置修订号

    返回:
        dict[str, Any]: 服务商、各分区配置与引用问题
    """
    try:
        raw_ai = load_config(read_config())["AI"]
    except Exception:
        raw_ai = {}
    raw_providers = ai_get(raw_ai, "PROVIDERS", [])

    def raw_provider(name: str) -> dict[str, Any] | None:
        index = provider_index(raw_providers, name)
        return None if index is None else raw_providers[index]

    def section(key: str, normalized: Any) -> Any:
        value = ai_get(raw_ai, key)
        return jsonable_encoder(value) if value is not None else normalized

    return {
        "revision": revision,
        "schema": _schema(),
        "presets": {
            name: {
                "label": preset["label"],
                "protocol": preset["protocol"],
                "base_urls": preset["base_urls"],
                "key_url": preset.get("key_url"),
                "web_search": list(preset.get("web_search", ())),
            }
            for name, preset in PRESETS.items()
        },
        "protocols": list(PROTOCOLS),
        "providers": [
            _provider_view(provider, raw_provider(provider.name))
            for provider in config.providers
        ],
        "sections": {
            "default_models": section(
                "default_models", model_dump(config.default_models)
            ),
            "model_groups": section("MODEL_GROUPS", config.model_groups),
            "context": section("context_settings", model_dump(config.context_settings)),
            "agent": section("agent_settings", model_dump(config.agent_settings)),
            "web_search": section("web_search", model_dump(config.web_search)),
            "deepseek_tools": section(
                "deepseek_tools", model_dump(config.deepseek_tools)
            ),
            "advanced": {
                "client_settings": section(
                    "client_settings", model_dump(config.client_settings)
                ),
                "debug_log": section("debug_log", model_dump(config.debug_log)),
            },
        },
        "validation_issues": reference_issues(config),
        "catalog": catalog_status(),
    }
