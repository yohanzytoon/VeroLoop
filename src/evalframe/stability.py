"""Stability over validated structured outputs."""

from collections import Counter
from typing import Any

from evalframe.schema.canonical import stable_json


def exact_agreement(outputs: list[dict[str, Any]]) -> float | None:
    if not outputs:
        return None
    counts = Counter(stable_json(output) for output in outputs)
    return max(counts.values()) / len(outputs)


def categorical_agreement(outputs: list[dict[str, Any]], field: str) -> float | None:
    values = [stable_json(output[field]) for output in outputs if field in output]
    if not values:
        return None
    return max(Counter(values).values()) / len(values)
