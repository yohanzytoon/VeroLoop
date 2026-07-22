"""EvalFrame public API."""

__version__ = "0.1.0"

from evalframe.models import Candidate, EvaluationCase, EvaluationReport, TaskDefinition
from evalframe.runner import EvaluationRunner, RunnerConfig

__all__ = [
    "Candidate",
    "EvaluationCase",
    "EvaluationReport",
    "EvaluationRunner",
    "RunnerConfig",
    "TaskDefinition",
]
