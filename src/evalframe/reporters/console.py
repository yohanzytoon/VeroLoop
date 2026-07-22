"""Rich terminal comparison."""

from rich.console import Console
from rich.table import Table

from evalframe.models import EvaluationReport


def _number(value: float | None, suffix: str = "") -> str:
    return "unknown" if value is None else f"{value:.3f}{suffix}"


class ConsoleReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def report(self, report: EvaluationReport) -> None:
        table = Table(title=f"EvalFrame: {report.task_name}")
        for column in (
            "Candidate",
            "Cases",
            "Attempts",
            "Errors",
            "Schema",
            "Correct",
            "Stable",
            "Mean ms",
            "P95 ms",
            "Mean cost",
            "Total cost",
        ):
            table.add_column(column)
        for row in report.summaries:
            table.add_row(
                row.candidate_name,
                str(row.cases),
                str(row.attempts),
                str(row.errors),
                _number(row.schema_validity_rate),
                _number(row.correctness_score),
                _number(row.stability_score),
                _number(row.mean_latency_ms),
                _number(row.p95_latency_ms),
                _number(row.mean_cost_usd, " USD"),
                _number(row.total_cost_usd, " USD"),
            )
        self.console.print(table)
