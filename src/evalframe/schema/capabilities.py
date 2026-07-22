"""Schema keyword compatibility checking."""

from __future__ import annotations

from typing import Any

from evalframe.models import CompatibilityReport, ProviderCapabilities

_ANNOTATIONS = {"title", "description", "$id", "$schema", "default", "examples"}


def _keywords(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"properties", "$defs"} and isinstance(child, dict):
                for nested_schema in child.values():
                    found.update(_keywords(nested_schema))
                continue
            if key not in _ANNOTATIONS:
                found.add(key)
            found.update(_keywords(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keywords(child))
    return found


def check_compatibility(
    schema: dict[str, Any], capabilities: ProviderCapabilities
) -> CompatibilityReport:
    if capabilities.native_structured_output:
        strategy = "native_strict"
    elif capabilities.strict_tool_schema:
        strategy = "strict_tool"
    else:
        return CompatibilityReport(
            supported=False,
            strategy="none",
            hard_errors=["Provider has no strict structured-output strategy"],
        )
    allowed = capabilities.supported_json_schema_keywords
    unsupported = sorted(_keywords(schema) - allowed) if allowed is not None else []
    return CompatibilityReport(
        supported=not unsupported,
        unsupported_keywords=unsupported,
        strategy=strategy,
        hard_errors=[f"Unsupported JSON Schema keyword: {x}" for x in unsupported],
    )
