import importlib.util
from pathlib import Path

import pytest

from evalframe.reporters.mlflow import MLflowReporter


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is not None, reason="MLflow is installed")
def test_mlflow_missing_dependency_has_install_guidance() -> None:
    with pytest.raises(ImportError, match=r"evalframe\[mlflow\]"):
        MLflowReporter()


@pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None, reason="MLflow optional extra not installed"
)
def test_mlflow_local_file_store_when_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The complete report-writing behavior is shared with JsonReporter; this verifies local setup.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    assert MLflowReporter(experiment_name="evalframe-test")
