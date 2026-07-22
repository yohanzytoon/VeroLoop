"""EvalFrame command-line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from evalframe import __version__
from evalframe.config import load_config, resolve_task
from evalframe.exceptions import EvalFrameException
from evalframe.metrics import FieldCorrectnessMetric, Metric, SchemaValidityMetric
from evalframe.models import ModelRequest
from evalframe.providers.fake import FakeProvider
from evalframe.reporters import ConsoleReporter, JsonReporter
from evalframe.runner import EvaluationRunner, RunnerConfig, load_dataset
from evalframe.schema import schema_for

app = typer.Typer(help="Provider-neutral structured-output evaluation.", no_args_is_help=True)

_CONFIG = """task:
  name: support-classification
  dataset: data.jsonl
  output_model: schema.py:SupportAnswer
  repetitions: 3
candidates:
  - {name: baseline, provider: fake, model: deterministic, prompt: Classify support requests.}
metrics: {correctness: {fields: [category, should_escalate]}}
constraints: {max_concurrency: 10}
reporters: {console: true, json: report.json}
"""
_SCHEMA = """from typing import Literal
from pydantic import BaseModel, ConfigDict

class SupportAnswer(BaseModel):
    model_config = ConfigDict(extra=\"forbid\")
    category: Literal[\"billing\", \"technical\", \"account\", \"other\"]
    answer: str
    should_escalate: bool
"""
_DATA = (
    '{"id":"case-1","inputs":{"message":"I was charged twice"},'
    '"expected":{"category":"billing","should_escalate":true}}\n'
)


def _create(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


@app.command()
def init(directory: Annotated[Path, typer.Argument()] = Path("evalframe-example")) -> None:
    """Create an offline example without overwriting files."""
    created = [
        path
        for path, content in (
            (directory / "eval.yaml", _CONFIG),
            (directory / "schema.py", _SCHEMA),
            (directory / "data.jsonl", _DATA),
        )
        if _create(path, content)
    ]
    typer.echo(f"Created {len(created)} file(s) in {directory}")


@app.command()
def validate(path: Path) -> None:
    """Validate configuration, dataset, and canonical schema."""
    try:
        config = load_config(path)
        task = resolve_task(config, path.parent)
        cases = load_dataset(task.dataset)
        schema_for(task.output_model)
        typer.echo(f"Valid: {len(cases)} cases, {len(config.candidates)} candidates")
    except (EvalFrameException, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _fake_response(request: ModelRequest) -> dict[str, object]:
    import json

    if not isinstance(request.user_input, str):
        raise EvalFrameException("Fake CLI provider expects text input")
    inputs = json.loads(request.user_input)
    message = str(inputs.get("message", "")).lower()
    if any(word in message for word in ("charge", "bill", "refund")):
        category = "billing"
    elif any(word in message for word in ("login", "password", "account")):
        category = "account"
    elif any(word in message for word in ("error", "broken", "technical")):
        category = "technical"
    else:
        category = "other"
    return {
        "category": category,
        "answer": "We will help with your request.",
        "should_escalate": category == "billing",
    }


@app.command()
def run(path: Path) -> None:
    """Run an evaluation."""
    try:
        config = load_config(path)
        task = resolve_task(config, path.parent)
        providers = {"fake": FakeProvider(_fake_response)}
        unknown = sorted({c.provider for c in config.candidates} - providers.keys())
        if unknown:
            raise EvalFrameException(
                f"CLI provider setup is required for: {', '.join(unknown)}; "
                "see examples/multi_provider_demo.py"
            )
        correctness = config.metrics.get("correctness", {})
        metrics: list[Metric] = [
            SchemaValidityMetric(),
            FieldCorrectnessMetric(fields=correctness.get("fields")),
        ]
        runner_config = RunnerConfig.model_validate(config.constraints)
        report = asyncio.run(
            EvaluationRunner(providers, metrics=metrics, config=runner_config).run(
                task, config.candidates
            )
        )
        if config.reporters.get("console", True):
            ConsoleReporter().report(report)
        output = config.reporters.get("json", "report.json")
        if output:
            saved = JsonReporter(path.parent / str(output)).report(report)
            typer.echo(f"Saved {saved}")
        mlflow_config = config.reporters.get("mlflow")
        if mlflow_config:
            from evalframe.reporters.mlflow import MLflowReporter

            options = mlflow_config if isinstance(mlflow_config, dict) else {}
            MLflowReporter(**options).report(report)
    except (EvalFrameException, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def capabilities() -> None:
    """Show built-in adapter capabilities."""
    caps = FakeProvider().capabilities()
    typer.echo(f"fake: {caps.model_dump_json()}")
    typer.echo(
        "openai: optional [openai]; anthropic: optional [anthropic]; gemini: optional [gemini]"
    )


@app.command(name="version")
def version_command() -> None:
    typer.echo(__version__)


if __name__ == "__main__":
    app()
