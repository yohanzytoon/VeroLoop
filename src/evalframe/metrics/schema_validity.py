from evalframe.models import AttemptResult, EvaluationCase, MetricResult


class SchemaValidityMetric:
    name = "schema_validity"

    def score(self, case: EvaluationCase, attempt: AttemptResult) -> MetricResult:
        return MetricResult(name=self.name, value=attempt.schema_valid)
