"""Optional MLflow integration with privacy-safe defaults."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from evalframe.models import EvaluationReport


class MLflowReporter:
    def __init__(self, *, experiment_name: str = "evalframe", log_content: bool = False) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Install MLflow support with: pip install 'evalframe[mlflow]'"
            ) from exc
        self.mlflow: Any = mlflow
        self.experiment_name = experiment_name
        self.log_content = log_content

    def report(self, report: EvaluationReport) -> None:
        self.mlflow.set_experiment(self.experiment_name)
        with self.mlflow.start_run():
            self.mlflow.log_params(
                {
                    "task": report.task_name,
                    "schema_hash": report.schema_hash,
                    "dataset_hash": report.dataset_hash,
                }
            )
            for summary in report.summaries:
                prefix = summary.candidate_name.replace(" ", "_")
                values = summary.model_dump()
                for name, value in values.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.mlflow.log_metric(f"{prefix}.{name}", value)
            with tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / "report.json"
                payload = (
                    report
                    if self.log_content
                    else report.model_copy(
                        update={
                            "attempts": [
                                a.model_copy(update={"response": None, "validated_output": None})
                                for a in report.attempts
                            ]
                        }
                    )
                )
                artifact.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
                self.mlflow.log_artifact(str(artifact))
