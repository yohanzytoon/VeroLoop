"""Canonical schema generation and response validation."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel


def canonicalize(value: Any) -> Any:
    """Recursively make JSON-compatible data deterministic."""
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    return cast(dict[str, Any], canonicalize(model.model_json_schema()))


def stable_json(value: Any) -> str:
    return json.dumps(canonicalize(value), separators=(",", ":"), ensure_ascii=False)


def validate_output(model: type[BaseModel], data: Any) -> dict[str, Any]:
    return model.model_validate(data).model_dump(mode="json")
