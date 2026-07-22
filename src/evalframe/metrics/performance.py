from __future__ import annotations

import math
import statistics


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def latency_statistics(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return statistics.fmean(values), statistics.median(values), percentile(values, 0.95)
