from typing import Protocol

from evalframe.models import AttemptResult, EvaluationCase, MetricResult


class Metric(Protocol):
    name: str

    def score(self, case: EvaluationCase, attempt: AttemptResult) -> MetricResult: ...
