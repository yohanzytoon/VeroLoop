from pathlib import Path

import pytest

from evalframe.config import load_config
from evalframe.costs import PriceRegistry, TokenPrice
from evalframe.exceptions import ConfigurationError, EvaluationError, ProviderError
from evalframe.models import EvaluationCase
from evalframe.runner import load_dataset


def test_jsonl_and_duplicate_validation(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text('{"id":"a","inputs":{}}\n{"id":"a","inputs":{}}\n')
    with pytest.raises(ConfigurationError, match="Duplicate"):
        load_dataset(source)
    assert load_dataset([EvaluationCase(id="one", inputs={})])[0].id == "one"


def test_config_environment_interpolation_without_secret_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "task: {name: x, dataset: d, output_model: s.py:S}\n"
        "candidates: [{name: c, provider: fake, model: '${MODEL_SECRET}', prompt: p}]\n"
    )
    monkeypatch.setenv("MODEL_SECRET", "private-model")
    assert load_config(path).candidates[0].model == "private-model"
    monkeypatch.delenv("MODEL_SECRET")
    with pytest.raises(ConfigurationError) as error:
        load_config(path)
    assert "private-model" not in str(error.value)


def test_cost_unknown_and_known() -> None:
    registry = PriceRegistry({("p", "m"): TokenPrice(1, 2)})
    assert registry.estimate("p", "unknown", 10, 10) is None
    assert registry.estimate("p", "m", None, 10) is None
    assert registry.estimate("p", "m", 1_000_000, 500_000) == 2


def test_secrets_are_sanitized() -> None:
    error = EvaluationError.from_exception(ProviderError("Authorization: Bearer super-secret"))
    assert "super-secret" not in error.message
    assert "REDACTED" in error.message
