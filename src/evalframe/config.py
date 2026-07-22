"""YAML configuration with environment interpolation."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from evalframe.exceptions import ConfigurationError
from evalframe.models import Candidate, TaskDefinition

_ENV = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


class TaskConfig(BaseModel):
    name: str
    dataset: Path
    output_model: str
    repetitions: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    task: TaskConfig
    candidates: list[Candidate]
    metrics: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    reporters: dict[str, Any] = Field(default_factory=dict)


def _interpolate(text: str) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name)
        if value is None:
            missing.add(name)
            return ""
        return value

    result = _ENV.sub(replace, text)
    if missing:
        raise ConfigurationError(f"Missing environment variables: {', '.join(sorted(missing))}")
    return result


def load_config(path: Path) -> EvalConfig:
    try:
        data = yaml.safe_load(_interpolate(path.read_text(encoding="utf-8")))
        return EvalConfig.model_validate(data)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


def resolve_task(config: EvalConfig, base: Path) -> TaskDefinition:
    reference = config.task.output_model
    if ":" not in reference:
        raise ConfigurationError("output_model must use 'path.py:ClassName'")
    file_name, class_name = reference.split(":", 1)
    module_path = (base / file_name).resolve()
    spec = importlib.util.spec_from_file_location("evalframe_user_schema", module_path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Cannot load schema module: {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_model = getattr(module, class_name, None)
    if not isinstance(output_model, type) or not issubclass(output_model, BaseModel):
        raise ConfigurationError(f"{class_name} is not a Pydantic model")
    return TaskDefinition(
        name=config.task.name,
        dataset=(base / config.task.dataset).resolve(),
        output_model=output_model,
        repetitions=config.task.repetitions,
        metadata=config.task.metadata,
    )
