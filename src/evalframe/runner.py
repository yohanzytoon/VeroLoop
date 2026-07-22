"""Asynchronous evaluation orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from evalframe import __version__
from evalframe.costs import PriceRegistry
from evalframe.exceptions import ConfigurationError, EvaluationError, ProviderError
from evalframe.metrics import FieldCorrectnessMetric, Metric, SchemaValidityMetric
from evalframe.metrics.performance import latency_statistics
from evalframe.models import (
    AttemptResult,
    Candidate,
    CandidateSummary,
    EvaluationCase,
    EvaluationReport,
    ModelRequest,
    TaskDefinition,
)
from evalframe.providers.base import ProviderAdapter
from evalframe.schema import compile_schema, schema_for, validate_output
from evalframe.schema.canonical import stable_json
from evalframe.stability import exact_agreement

ProgressCallback = Callable[[int, int, AttemptResult], None]


class RunnerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_concurrency: int = Field(default=10, ge=1)
    per_provider_concurrency: dict[str, int] = Field(default_factory=dict)
    max_transport_attempts: int = Field(default=3, ge=1)
    progress: ProgressCallback | None = None


def load_dataset(source: list[EvaluationCase] | Path) -> list[EvaluationCase]:
    if isinstance(source, list):
        cases = source
    else:
        cases = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        cases.append(EvaluationCase.model_validate_json(line))
                    except ValidationError as exc:
                        raise ConfigurationError(
                            f"Invalid JSONL line {line_number}: {exc}"
                        ) from exc
    ids = [case.id for case in cases]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ConfigurationError(f"Duplicate case IDs: {', '.join(duplicates)}")
    if not cases:
        raise ConfigurationError("Dataset must contain at least one case")
    return cases


class EvaluationRunner:
    def __init__(
        self,
        providers: Mapping[str, ProviderAdapter],
        *,
        metrics: Sequence[Metric] | None = None,
        price_registry: PriceRegistry | None = None,
        config: RunnerConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.providers = providers
        self.metrics: list[Metric] = list(
            metrics or [SchemaValidityMetric(), FieldCorrectnessMetric()]
        )
        self.prices = price_registry or PriceRegistry()
        self.config = config or RunnerConfig()
        self.clock = clock

    async def run(self, task: TaskDefinition, candidates: Sequence[Candidate]) -> EvaluationReport:
        cases = load_dataset(task.dataset)
        names = [candidate.name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ConfigurationError("Candidate names must be unique")
        schema = schema_for(task.output_model)
        compatibility = {}
        for candidate in candidates:
            provider = self.providers.get(candidate.provider)
            if provider is None:
                raise ConfigurationError(
                    f"No adapter registered for provider: {candidate.provider}"
                )
            _, compatibility[candidate.name] = compile_schema(schema, provider.capabilities())

        started = datetime.now(UTC)
        global_limit = asyncio.Semaphore(self.config.max_concurrency)
        provider_limits = {
            name: asyncio.Semaphore(limit)
            for name, limit in self.config.per_provider_concurrency.items()
        }
        jobs = [
            (candidate_index, case_index, repetition, candidate, case)
            for candidate_index, candidate in enumerate(candidates)
            for case_index, case in enumerate(cases)
            for repetition in range(task.repetitions)
        ]
        completed = 0

        async def execute(
            job: tuple[int, int, int, Candidate, EvaluationCase],
        ) -> tuple[tuple[int, int, int], AttemptResult]:
            nonlocal completed
            candidate_index, case_index, repetition, candidate, case = job
            provider = self.providers[candidate.provider]
            report = compatibility[candidate.name]
            if not report.supported:
                result = AttemptResult(
                    case_id=case.id,
                    candidate_name=candidate.name,
                    repetition=repetition,
                    response=None,
                    schema_valid=False,
                    validated_output=None,
                    error=EvaluationError(
                        kind="SchemaCompatibilityError", message="; ".join(report.hard_errors)
                    ),
                )
            else:
                limit = provider_limits.get(candidate.provider)
                async with global_limit:
                    if limit is None:
                        result = await self._attempt(
                            task, candidate, case, provider, schema, repetition
                        )
                    else:
                        async with limit:
                            result = await self._attempt(
                                task, candidate, case, provider, schema, repetition
                            )
            completed += 1
            if self.config.progress:
                self.config.progress(completed, len(jobs), result)
            return (candidate_index, case_index, repetition), result

        ordered = sorted(
            await asyncio.gather(*(execute(job) for job in jobs)), key=lambda item: item[0]
        )
        attempts = [item[1] for item in ordered]
        summaries = self._summaries(cases, candidates, attempts)
        finished = datetime.now(UTC)
        dataset_payload = [case.model_dump(mode="json") for case in cases]
        return EvaluationReport(
            package_version=__version__,
            task_name=task.name,
            task_metadata=task.metadata,
            candidates=[
                {
                    "name": c.name,
                    "provider": c.provider,
                    "model": c.model,
                    "parameters": c.parameters,
                    "metadata": c.metadata,
                }
                for c in candidates
            ],
            started_at=started,
            finished_at=finished,
            environment={"python": platform.python_version(), "platform": platform.platform()},
            dataset_hash=hashlib.sha256(stable_json(dataset_payload).encode()).hexdigest(),
            schema_hash=hashlib.sha256(stable_json(schema).encode()).hexdigest(),
            attempts=attempts,
            summaries=summaries,
            compatibility_reports=compatibility,
        )

    async def _attempt(
        self,
        task: TaskDefinition,
        candidate: Candidate,
        case: EvaluationCase,
        provider: ProviderAdapter,
        schema: dict[str, Any],
        repetition: int,
    ) -> AttemptResult:
        request = ModelRequest(
            case_id=case.id,
            system_prompt=candidate.prompt,
            user_input=json.dumps(case.inputs, sort_keys=True),
            output_json_schema=schema,
            parameters={"model": candidate.model, **candidate.parameters},
        )

        def retryable(exc: BaseException) -> bool:
            return isinstance(exc, ProviderError) and exc.retryable

        @retry(
            retry=retry_if_exception(retryable),
            stop=stop_after_attempt(self.config.max_transport_attempts),
            wait=wait_exponential(multiplier=0.01, max=0.1),
            reraise=True,
        )
        async def generate() -> Any:
            return await provider.generate(request.model_copy(deep=True))

        response = None
        started = self.clock()
        try:
            response = await generate()
            response.latency_ms = (self.clock() - started) * 1000
            response.estimated_cost_usd = self.prices.estimate(
                candidate.provider, candidate.model, response.input_tokens, response.output_tokens
            )
            validated = validate_output(task.output_model, response.parsed_output)
            attempt = AttemptResult(
                case_id=case.id,
                candidate_name=candidate.name,
                repetition=repetition,
                response=response,
                schema_valid=True,
                validated_output=validated,
            )
        except Exception as exc:
            attempt = AttemptResult(
                case_id=case.id,
                candidate_name=candidate.name,
                repetition=repetition,
                response=response,
                schema_valid=False,
                validated_output=None,
                error=EvaluationError.from_exception(exc),
            )
        attempt.metric_results = [metric.score(case, attempt) for metric in self.metrics]
        return attempt

    def _summaries(
        self,
        cases: list[EvaluationCase],
        candidates: Sequence[Candidate],
        attempts: list[AttemptResult],
    ) -> list[CandidateSummary]:
        result = []
        for candidate in candidates:
            selected = [item for item in attempts if item.candidate_name == candidate.name]
            latencies = [item.response.latency_ms for item in selected if item.response]
            mean_latency, median_latency, p95_latency = latency_statistics(latencies)
            correctness = [
                float(metric.value)
                for item in selected
                for metric in item.metric_results
                if metric.name == "correctness" and metric.value is not None
            ]
            costs = [item.response.estimated_cost_usd for item in selected if item.response]
            known_costs = [value for value in costs if value is not None]
            per_case = defaultdict(list)
            for item in selected:
                if item.validated_output is not None:
                    per_case[item.case_id].append(item.validated_output)
            stabilities = [
                value
                for values in per_case.values()
                if (value := exact_agreement(values)) is not None
            ]
            result.append(
                CandidateSummary(
                    candidate_name=candidate.name,
                    cases=len(cases),
                    attempts=len(selected),
                    errors=sum(item.error is not None for item in selected),
                    schema_validity_rate=sum(item.schema_valid for item in selected)
                    / len(selected),
                    correctness_score=statistics.fmean(correctness) if correctness else None,
                    stability_score=statistics.fmean(stabilities) if stabilities else None,
                    mean_latency_ms=mean_latency,
                    median_latency_ms=median_latency,
                    p95_latency_ms=p95_latency,
                    mean_cost_usd=statistics.fmean(known_costs)
                    if known_costs and len(known_costs) == len(costs)
                    else None,
                    total_cost_usd=sum(known_costs)
                    if known_costs and len(known_costs) == len(costs)
                    else None,
                )
            )
        return result
