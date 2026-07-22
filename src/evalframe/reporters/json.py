from pathlib import Path

from evalframe.models import EvaluationReport


class JsonReporter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def report(self, report: EvaluationReport) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return self.path
