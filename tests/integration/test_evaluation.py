import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict
from typer.testing import CliRunner

from evalframe import Candidate, EvaluationRunner, TaskDefinition
from evalframe.cli import app
from evalframe.costs import PriceRegistry, TokenPrice
from evalframe.models import EvaluationReport
from evalframe.providers.fake import FakeProvider
from evalframe.reporters.json import JsonReporter


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["yes", "no"]


@pytest.mark.asyncio
async def test_complete_evaluation_and_json_round_trip(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id":"one","inputs":{"value":true},"expected":{"label":"yes"}}\n')
    provider = FakeProvider(lambda request: {"label": "yes"})
    task = TaskDefinition(name="answer", dataset=cases, output_model=Answer, repetitions=3)
    report = await EvaluationRunner(
        {"fake": provider}, price_registry=PriceRegistry({("fake", "m"): TokenPrice(1, 2)})
    ).run(task, [Candidate(name="candidate", provider="fake", model="m", prompt="answer")])
    assert len(report.attempts) == 3
    assert report.summaries[0].schema_validity_rate == 1
    assert report.summaries[0].correctness_score == 1
    assert report.summaries[0].stability_score == 1
    assert report.summaries[0].total_cost_usd is not None
    path = JsonReporter(tmp_path / "report.json").report(report)
    restored = EvaluationReport.model_validate_json(path.read_text())
    assert restored.dataset_hash == report.dataset_hash
    assert "prompt" not in json.loads(path.read_text())["candidates"][0]


def test_cli_init_validate_run(tmp_path: Path) -> None:
    runner = CliRunner()
    directory = tmp_path / "example"
    assert runner.invoke(app, ["init", str(directory)]).exit_code == 0
    assert runner.invoke(app, ["init", str(directory)]).exit_code == 0
    assert runner.invoke(app, ["validate", str(directory / "eval.yaml")]).exit_code == 0
    result = runner.invoke(app, ["run", str(directory / "eval.yaml")])
    assert result.exit_code == 0, result.output
    assert (directory / "report.json").exists()
    assert runner.invoke(app, ["version"]).output.strip() == "0.1.0"
