"""Deterministic selected-field correctness."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

from evalframe.models import AttemptResult, EvaluationCase, MetricResult


class NormalizationConfig(BaseModel):
    strip_strings: bool = True
    lowercase_strings: bool = False
    numeric_tolerance: float = Field(default=0, ge=0)
    list_order_matters: bool = True


def _normalize(value: Any, config: NormalizationConfig) -> Any:
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, str):
        value = value.strip() if config.strip_strings else value
        return value.lower() if config.lowercase_strings else value
    if isinstance(value, list):
        values = [_normalize(item, config) for item in value]
        return values if config.list_order_matters else sorted(values, key=repr)
    if isinstance(value, dict):
        return {key: _normalize(child, config) for key, child in value.items()}
    return value


class FieldCorrectnessMetric:
    name = "correctness"

    def __init__(
        self, fields: list[str] | None = None, config: NormalizationConfig | None = None
    ) -> None:
        self.fields = fields
        self.config = config or NormalizationConfig()

    def score(self, case: EvaluationCase, attempt: AttemptResult) -> MetricResult:
        if case.expected is None or attempt.validated_output is None:
            return MetricResult(name=self.name, value=None)
        fields = self.fields or list(case.expected)
        matches: dict[str, bool] = {}
        for field in fields:
            actual = _normalize(attempt.validated_output.get(field), self.config)
            expected = _normalize(case.expected.get(field), self.config)
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                matches[field] = abs(actual - expected) <= self.config.numeric_tolerance
            else:
                matches[field] = actual == expected
        score = sum(matches.values()) / len(matches) if matches else None
        return MetricResult(name=self.name, value=score, details={"fields": matches})
