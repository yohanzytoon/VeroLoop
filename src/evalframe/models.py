"""Public domain models. Provider SDK objects never cross this boundary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalframe.exceptions import EvaluationError


class EvaluationCase(BaseModel):
    id: str
    inputs: dict[str, Any]
    expected: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    dataset: list[EvaluationCase] | Path
    output_model: type[BaseModel]
    repetitions: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    name: str
    provider: str
    model: str
    prompt: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCapabilities(BaseModel):
    native_structured_output: bool
    strict_tool_schema: bool
    json_mode: bool
    streaming: bool
    usage_reporting: bool
    supported_json_schema_keywords: frozenset[str] | None = None


class CompatibilityReport(BaseModel):
    supported: bool
    unsupported_keywords: list[str] = Field(default_factory=list)
    transformed_keywords: list[str] = Field(default_factory=list)
    strategy: Literal["native_strict", "strict_tool", "json_mode", "prompt_only", "none"]
    warnings: list[str] = Field(default_factory=list)
    hard_errors: list[str] = Field(default_factory=list)


class ModelRequest(BaseModel):
    case_id: str
    system_prompt: str | None = None
    user_input: str | list[dict[str, Any]]
    output_json_schema: dict[str, Any]
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    raw_output: Any
    parsed_output: dict[str, Any] | None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    provider_request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    name: str
    value: float | bool | None
    details: dict[str, Any] = Field(default_factory=dict)


class AttemptResult(BaseModel):
    case_id: str
    candidate_name: str
    repetition: int
    response: ModelResponse | None
    schema_valid: bool
    validated_output: dict[str, Any] | None
    metric_results: list[MetricResult] = Field(default_factory=list)
    error: EvaluationError | None = None


class CandidateSummary(BaseModel):
    candidate_name: str
    cases: int
    attempts: int
    errors: int
    schema_validity_rate: float
    correctness_score: float | None
    stability_score: float | None
    mean_latency_ms: float | None
    median_latency_ms: float | None
    p95_latency_ms: float | None
    mean_cost_usd: float | None
    total_cost_usd: float | None


class EvaluationReport(BaseModel):
    schema_version: str = "1.0"
    package_version: str
    task_name: str
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]]
    started_at: datetime
    finished_at: datetime
    environment: dict[str, str]
    dataset_hash: str
    schema_hash: str
    attempts: list[AttemptResult]
    summaries: list[CandidateSummary]
    compatibility_reports: dict[str, CompatibilityReport]

    @field_validator("candidates")
    @classmethod
    def no_prompts(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in value:
            if "prompt" in item:
                raise ValueError("candidate prompts must be sanitized")
        return value
