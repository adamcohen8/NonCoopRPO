from __future__ import annotations

import copy
import importlib
from typing import Any

import yaml


def pointer_form_schema(pointer: dict[str, Any], schemas: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]] | None:
    class_name = str(pointer.get("class_name", "") or "")
    return schemas.get(class_name)


def pointer_display_name(pointer: dict[str, Any]) -> str:
    return f"{pointer.get('module', '')}.{pointer.get('class_name', '') or pointer.get('function', '')}".strip(".")


def default_params_for_pointer(pointer: dict[str, Any]) -> dict[str, Any]:
    module_name = str(pointer.get("module", "") or "")
    class_name = str(pointer.get("class_name", "") or "")
    if not module_name or not class_name:
        return {}
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        return copy.deepcopy(getattr(instance, "__dict__", {}) or {})
    except Exception:
        return {}


def normalize_form_value(field_spec: dict[str, Any], params: dict[str, Any], defaults: dict[str, Any]) -> object:
    key = field_spec["key"]
    if key in params:
        return params.get(key)
    return defaults.get(key)


def format_vector_text(value: object, length: int | None = None) -> str:
    if value is None:
        values = [0.0] * int(length or 0)
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    if length is not None and len(values) < length:
        values = values + [0.0] * (length - len(values))
    return ", ".join(str(v) for v in values)


def parse_vector_text(text: str, length: int | None = None) -> list[float]:
    raw = text.strip()
    if not raw:
        values: list[float] = [0.0] * int(length or 0)
    else:
        parsed = yaml.safe_load(f"[{raw}]") if "[" not in raw else yaml.safe_load(raw)
        if not isinstance(parsed, list):
            raise ValueError("Vector values must be a comma-separated list.")
        values = [float(v) for v in parsed]
    if length is not None and len(values) != length:
        raise ValueError(f"Expected {length} values.")
    return values


def format_yaml_text(value: object) -> str:
    payload: object = [] if value is None else value
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip()
    return text if text else "[]"


def parse_yaml_text(text: str) -> object:
    raw = text.strip()
    if not raw:
        return []
    return yaml.safe_load(raw)
