import pytest

from evalframe.metrics.correctness import FieldCorrectnessMetric, NormalizationConfig
from evalframe.metrics.performance import latency_statistics, percentile
from evalframe.models import AttemptResult, EvaluationCase
from evalframe.stability import categorical_agreement, exact_agreement


def attempt(output: dict[str, object] | None) -> AttemptResult:
    return AttemptResult(
        case_id="1",
        candidate_name="x",
        repetition=0,
        response=None,
        schema_valid=output is not None,
        validated_output=output,
    )


def test_explicit_normalized_correctness() -> None:
    case = EvaluationCase(
        id="1", inputs={}, expected={"name": " Alice ", "score": 1.01, "tags": ["b", "a"]}
    )
    metric = FieldCorrectnessMetric(
        config=NormalizationConfig(
            lowercase_strings=True, numeric_tolerance=0.02, list_order_matters=False
        )
    )
    result = metric.score(case, attempt({"name": "alice", "score": 1, "tags": ["a", "b"]}))
    assert result.value == 1


def test_unknown_correctness() -> None:
    assert (
        FieldCorrectnessMetric().score(EvaluationCase(id="1", inputs={}), attempt(None)).value
        is None
    )


def test_stability_uses_structured_outputs() -> None:
    outputs = [{"category": "a", "x": 1}, {"x": 1, "category": "a"}, {"category": "b", "x": 1}]
    assert exact_agreement(outputs) == pytest.approx(2 / 3)
    assert categorical_agreement(outputs, "category") == pytest.approx(2 / 3)


def test_latency_nearest_rank_p95() -> None:
    assert percentile(list(range(1, 21)), 0.95) == 19
    mean, median, p95 = latency_statistics([1, 2, 100])
    assert mean == pytest.approx(103 / 3)
    assert median == 2
    assert p95 == 100
