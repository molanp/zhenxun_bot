"""AI 配置页：服务商、分区设置、模型路由与失效引用修复"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from zhenxun.services.ai.config import PRESETS
from zhenxun.services.ai.config.presets import preset_base_url
from zhenxun.utils.pydantic_compat import model_dump

from ....base_model import Result
from ....utils import authentication
from ._store import (
    ai_get,
    ai_set,
    apply_response,
    clean_keys,
    config_revision,
    configuration_error,
    configuration_view,
    existing_key,
    find_provider,
    load_config,
    persist,
    provider_index,
    read_config,
    reference_issues,
    validate_ai,
)

router = APIRouter()

_SECTIONS = {
    "default_models": "default_models",
    "context": "context_settings",
    "agent": "agent_settings",
    "web_search": "web_search",
    "deepseek_tools": "deepseek_tools",
    "advanced": None,
}
_ADVANCED_KEYS = {"client_settings", "debug_log"}


class ProviderSettingsUpdate(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    preset: str | None = Field(default=None, max_length=64)
    protocol: str | None = Field(default=None, max_length=64)
    base_url: str | None = Field(default=None, max_length=500)
    timeout: int = Field(default=120, ge=1, le=1800)
    headers: dict[str, str] | None = None
    params: dict[str, Any] | None = None
    api_keys: list[str] | None = None
    models: list[dict[str, Any]] | None = None


class ProviderModelsUpdate(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)
    models: list[dict[str, Any]]


class SectionUpdate(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)
    value: Any


class RoutingValidationRequest(BaseModel):
    value: Any
    default_models: dict[str, Any] | None = None
    expected_revision: str | None = None


class RoutingUpdate(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)
    model_groups: Any
    default_models: dict[str, Any]


class ReferenceRepairRequest(BaseModel):
    expected_revision: str = Field(min_length=64, max_length=64)


def _rename_provider_references(ai: dict[str, Any], old: str, new: str) -> None:
    if old == new:
        return
    old_prefix = f"{old}/"
    new_prefix = f"{new}/"

    defaults = ai_get(ai, "default_models", {})
    if isinstance(defaults, dict):
        for task, target in defaults.items():
            if isinstance(target, str) and target.startswith(old_prefix):
                defaults[task] = new_prefix + target[len(old_prefix) :]

    groups = ai_get(ai, "MODEL_GROUPS", {})
    if isinstance(groups, dict):
        for group, targets in groups.items():
            if isinstance(targets, list):
                groups[group] = [
                    new_prefix + target[len(old_prefix) :]
                    if isinstance(target, str) and target.startswith(old_prefix)
                    else target
                    for target in targets
                ]


def _merge_models(
    provider: dict[str, Any], models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """用提交的模型列表覆盖服务商模型，保留同名模型中未提交的字段

    参数:
        provider: 原始服务商配置
        models: 提交的模型列表

    返回:
        list[dict[str, Any]]: 合并后的模型列表
    """
    existing = {
        str(item.get("name")): item
        for item in provider.get("models") or []
        if isinstance(item, dict) and item.get("name")
    }
    return [
        {**deepcopy(existing.get(str(model.get("name", "")), {})), **model}
        for model in models
    ]


def _normalize_model_groups(value: Any) -> dict[str, list[str]]:
    if isinstance(value, dict):
        rows = [{"name": name, "targets": targets} for name, targets in value.items()]
    elif isinstance(value, list):
        rows = value
    else:
        raise configuration_error(
            "model_groups_invalid",
            "模型路由组必须是列表或映射。",
            "AI.MODEL_GROUPS",
        )
    groups: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise configuration_error(
                "model_group_invalid",
                "模型路由组格式无效。",
                f"AI.MODEL_GROUPS.{index}",
            )
        name = str(row.get("name") or "").strip()
        if not name:
            raise configuration_error(
                "model_group_name_required",
                "请填写路由组名称。",
                f"AI.MODEL_GROUPS.{index}.name",
            )
        if name in groups:
            raise configuration_error(
                "model_group_name_duplicate",
                f"路由组名称 {name} 重复。",
                f"AI.MODEL_GROUPS.{index}.name",
            )
        targets = row.get("targets", [])
        if not isinstance(targets, list) or not all(
            isinstance(target, str) and target.strip() for target in targets
        ):
            raise configuration_error(
                "model_group_targets_invalid",
                "模型路由目标必须是非空字符串列表。",
                f"AI.MODEL_GROUPS.{index}.targets",
            )
        normalized_targets = [target.strip() for target in targets]
        if len(normalized_targets) != len(set(normalized_targets)):
            raise configuration_error(
                "model_group_target_duplicate",
                f"路由组 {name} 不能重复引用同一目标。",
                f"AI.MODEL_GROUPS.{index}.targets",
            )
        groups[name] = normalized_targets
    return groups


def _normalize_defaults(value: Any) -> dict[str, str | None]:
    tasks = {"chat", "vision", "embedding", "tts", "stt", "image", "rerank"}
    if (
        not isinstance(value, dict)
        or set(value) - tasks
        or any(
            item is not None and not isinstance(item, str) for item in value.values()
        )
    ):
        raise configuration_error(
            "default_models_invalid",
            "默认模型必须是任务与模型名称的映射。",
            "AI.default_models",
        )
    return {task: (value.get(task) or "").strip() or None for task in sorted(tasks)}


@router.get(
    "/configuration",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def get_configuration() -> Result:
    content = read_config()
    data = load_config(content)
    config = validate_ai(data["AI"], strict_references=False)
    return Result.ok(configuration_view(config, config_revision(content)))


def _provider_fields(payload: ProviderSettingsUpdate, keys: Any) -> dict[str, Any]:
    """生成要写入的服务商字段，与预设默认值相同的协议和地址记为 None

    参数:
        payload: 提交的服务商设置
        keys: 要保存的 API Key

    返回:
        dict[str, Any]: 服务商字段，值为 None 的字段会从配置中移除
    """
    preset = payload.preset or None
    default_protocol = PRESETS.get(preset or "", {}).get("protocol", "openai_chat")
    protocol = payload.protocol or default_protocol
    base_url = (payload.base_url or "").strip().rstrip("/")
    fields: dict[str, Any] = {
        "name": payload.name.strip(),
        "preset": preset,
        "protocol": None if protocol == default_protocol else protocol,
        "base_url": None if base_url == preset_base_url(preset, protocol) else base_url,
        "api_keys": clean_keys(keys),
        "timeout": payload.timeout,
    }
    for key in ("headers", "params"):
        value = getattr(payload, key)
        if value is not None:
            fields[key] = value or None
    return fields


def _set_fields(target: dict[str, Any], fields: dict[str, Any]) -> None:
    """写入字段，值为 None 的字段从目标中移除"""
    for key, value in fields.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


def _check_provider_name(
    ai: dict[str, Any], name: str, own_index: int | None = None
) -> None:
    if provider_index(ai_get(ai, "PROVIDERS", []), name) not in (None, own_index):
        raise configuration_error(
            "provider_name_duplicate", "服务商名称已存在。", "AI.PROVIDERS"
        )


@router.post(
    "/providers",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def create_provider(payload: ProviderSettingsUpdate) -> Result:
    def mutate(ai: dict[str, Any]) -> None:
        _check_provider_name(ai, payload.name)
        providers = ai_get(ai, "PROVIDERS")
        if providers is None:
            providers = []
            ai_set(ai, "PROVIDERS", providers)
        provider: dict[str, Any] = {}
        _set_fields(provider, _provider_fields(payload, payload.api_keys))
        provider["models"] = deepcopy(payload.models or [])
        providers.append(provider)

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation),
        info="AI 服务商已保存并热加载。",
    )


@router.put(
    "/providers/{provider_name}",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def update_provider(
    provider_name: str, payload: ProviderSettingsUpdate
) -> Result:
    def mutate(ai: dict[str, Any]) -> None:
        index, provider = find_provider(ai, provider_name)
        _check_provider_name(ai, payload.name, index)
        fields = _provider_fields(
            payload,
            provider.get("api_keys") if payload.api_keys is None else payload.api_keys,
        )
        _rename_provider_references(ai, str(provider.get("name", "")), fields["name"])
        _set_fields(provider, fields)
        if payload.models is not None:
            provider["models"] = _merge_models(provider, payload.models)

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation),
        info="AI 服务商已保存并热加载。",
    )


@router.delete(
    "/providers/{provider_name}",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def delete_provider(provider_name: str, expected_revision: str) -> Result:
    def mutate(ai: dict[str, Any]) -> None:
        index, _ = find_provider(ai, provider_name)
        ai_get(ai, "PROVIDERS", []).pop(index)

    revision, config, operation = await persist(expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation),
        info="AI 服务商已删除并热加载。",
    )


@router.put(
    "/providers/{provider_name}/models",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def update_models(provider_name: str, payload: ProviderModelsUpdate) -> Result:
    def mutate(ai: dict[str, Any]) -> None:
        _, provider = find_provider(ai, provider_name)
        provider["models"] = _merge_models(provider, payload.models)

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation),
        info="模型列表已保存并热加载。",
    )


@router.put(
    "/configuration/sections/{section}",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def update_section(section: str, payload: SectionUpdate) -> Result:
    if section not in _SECTIONS:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ai_section_not_found",
                "message": "不支持的 AI 配置分区。",
            },
        )

    def update_value(ai: dict[str, Any], key: str, value: Any) -> None:
        found_key = existing_key(ai, key)
        existing = ai.get(found_key) if found_key is not None else None
        if isinstance(existing, dict) and isinstance(value, dict):
            for item_key, item_value in value.items():
                current = existing.get(item_key)
                if isinstance(current, dict) and isinstance(item_value, dict):
                    current.update(item_value)
                else:
                    existing[item_key] = item_value
        else:
            ai_set(ai, key, value)

    def mutate(ai: dict[str, Any]) -> None:
        key = _SECTIONS[section]
        if section == "advanced":
            if (
                not isinstance(payload.value, dict)
                or set(payload.value) - _ADVANCED_KEYS
            ):
                raise configuration_error(
                    "advanced_section_invalid", "高级设置包含不支持的字段。", "AI"
                )
            for advanced_key in _ADVANCED_KEYS & set(payload.value):
                update_value(ai, advanced_key, payload.value[advanced_key])
        elif section == "default_models":
            ai_set(ai, "default_models", _normalize_defaults(payload.value))
        elif key:
            update_value(ai, key, payload.value)

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation), info="AI 配置已保存。"
    )


@router.post(
    "/configuration/validate-routing",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
)
async def validate_routing(payload: RoutingValidationRequest) -> Result:
    content = read_config()
    if payload.expected_revision and payload.expected_revision != config_revision(
        content
    ):
        raise HTTPException(409, detail={"code": "configuration_revision_conflict"})
    data = load_config(content)
    candidate = deepcopy(data)
    ai_set(candidate["AI"], "MODEL_GROUPS", _normalize_model_groups(payload.value))
    if payload.default_models is not None:
        ai_set(
            candidate["AI"],
            "default_models",
            _normalize_defaults(payload.default_models),
        )
    config = validate_ai(candidate["AI"], strict_references=False)
    baseline = reference_issues(validate_ai(data["AI"], strict_references=False))
    blockers = [issue for issue in reference_issues(config) if issue not in baseline]
    return Result.ok({"valid": not blockers, "issues": blockers})


@router.put(
    "/configuration/routing", dependencies=[authentication()], response_model=Result
)
async def update_routing(payload: RoutingUpdate) -> Result:
    def mutate(ai):
        ai_set(ai, "MODEL_GROUPS", _normalize_model_groups(payload.model_groups))
        ai_set(ai, "default_models", _normalize_defaults(payload.default_models))

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation), info="模型路由与默认模型已保存。"
    )


def _repair_references(
    ai: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = validate_ai(ai, strict_references=False)
    groups = deepcopy(config.model_groups)
    defaults = _normalize_defaults(model_dump(config.default_models))
    models = {f"{p.name}/{m.name}" for p in config.providers for m in p.models}
    changes = []
    while True:
        changed = False
        for group, targets in list(groups.items()):
            kept = [
                target for target in targets if target in models or target in groups
            ]
            if kept and kept == targets:
                continue
            if kept:
                groups[group] = kept
            else:
                groups.pop(group)
            changes.append(
                {
                    "path": f"AI.MODEL_GROUPS.{group}",
                    "before": targets,
                    "after": kept or None,
                }
            )
            changed = True
        if not changed:
            break
    result = deepcopy(ai)
    ai_set(result, "MODEL_GROUPS", groups)
    ai_set(result, "default_models", defaults)
    for task, target in defaults.items():
        if target and target not in models and target not in groups:
            changes.append(
                {"path": f"AI.default_models.{task}", "before": target, "after": None}
            )
            defaults[task] = None
    for issue in reference_issues(validate_ai(result, strict_references=False)):
        if issue["code"] in {
            "default_model_missing",
            "default_model_capability_mismatch",
        }:
            task = issue["path"].rsplit(".", 1)[-1]
            changes.append(
                {"path": issue["path"], "before": defaults[task], "after": None}
            )
            defaults[task] = None
    return result, changes


@router.post(
    "/configuration/repair-references/preview",
    dependencies=[authentication()],
    response_model=Result,
)
async def preview_reference_repair(payload: ReferenceRepairRequest) -> Result:
    content = read_config()
    if config_revision(content) != payload.expected_revision:
        raise HTTPException(409, detail={"code": "configuration_revision_conflict"})
    candidate, changes = _repair_references(load_config(content)["AI"])
    config = validate_ai(candidate, strict_references=False)
    return Result.ok(
        {
            "revision": payload.expected_revision,
            "changes": changes,
            "remaining_issues": reference_issues(config),
            "model_groups": config.model_groups,
            "default_models": model_dump(config.default_models),
        }
    )


@router.post(
    "/configuration/repair-references/apply",
    dependencies=[authentication()],
    response_model=Result,
)
async def apply_reference_repair(payload: ReferenceRepairRequest) -> Result:
    def mutate(ai):
        candidate, _ = _repair_references(ai)
        ai_set(ai, "MODEL_GROUPS", ai_get(candidate, "MODEL_GROUPS"))
        ai_set(ai, "default_models", ai_get(candidate, "default_models"))

    revision, config, operation = await persist(payload.expected_revision, mutate)
    return Result.ok(
        apply_response(config, revision, operation),
        info="失效引用已修复，服务商和密钥保持不变。",
    )
