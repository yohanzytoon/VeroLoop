from pathlib import Path
from typing import Protocol

from evalframe.models import EvaluationReport


class Reporter(Protocol):
    def report(self, report: EvaluationReport) -> Path | None: ...
