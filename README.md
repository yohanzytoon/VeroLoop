# VeroLoop

VeroLoop is a Python SDK for evaluating one typed AI task consistently across provider candidates.

## Installation

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --no-editable
```

Provider clients remain optional: `uv sync --extra openai`, `--extra anthropic`,
`--extra gemini`, `--extra providers`, or `--extra mlflow`.

## Five-minute offline quick start

```bash
uv run --no-sync python examples/offline_demo.py
uv run --no-sync evalframe validate examples/eval.yaml
uv run --no-sync evalframe run examples/eval.yaml
```

The demo evaluates four customer-support cases against two deterministic fake candidates, repeats
each case three times, prints a comparison, and writes `examples/report.json` without credentials.

## Python API

```python
report = await EvaluationRunner({"fake": FakeProvider(responder)}).run(
    TaskDefinition(name="support", dataset=Path("cases.jsonl"),
                   output_model=SupportAnswer, repetitions=3),
    [Candidate(name="baseline", provider="fake", model="offline", prompt="Classify")],
)
```

The injected provider mapping, price registry, metrics, and runner configuration make tests fully
deterministic. `RunnerConfig` controls global and per-provider concurrency and transport retries.

## CLI

```bash
uv run --no-sync evalframe init my-evaluation
uv run --no-sync evalframe validate my-evaluation/eval.yaml
uv run --no-sync evalframe run my-evaluation/eval.yaml
uv run --no-sync evalframe capabilities
uv run --no-sync evalframe version
```

YAML supports `task`, `candidates`, `metrics`, `constraints`, and `reporters`. `${VARIABLE}` values
are interpolated from the environment; missing names are reported but values are never printed.
Version 0.1's generic CLI runs the offline provider. Real-provider setup is deliberately explicit
in Python so credentials and model capabilities are not guessed.

## Canonical schemas

A Pydantic model is the canonical contract. EvalFrame produces stable JSON Schema, checks adapter
capabilities, and validates every returned object against the original model. Unsupported keywords
are hard errors; the SDK never silently weakens the schema. Models should use
`ConfigDict(extra="forbid")` when extra properties must fail validation.

## Stability and metrics

Repetitions preserve every attempt. Stability is modal exact agreement over validated structured
outputs. Version 0.1 also calculates configurable normalized field correctness, schema validity,
mean/median/p95 latency, and known token cost. Missing prices or usage are `unknown`, never zero.

## MLflow

```python
from evalframe.reporters.mlflow import MLflowReporter
MLflowReporter(experiment_name="support").report(report)
```

Install with `uv sync --extra mlflow`. Summary metrics and a redacted report artifact are logged by
default. `log_content=True` includes outputs and should only be used after a privacy review.

## Privacy and limitations

Provider calls transmit prompts and inputs under the selected provider's retention terms. JSON
reports contain model outputs. MLflow omits prompt/output content by default; protect artifact
storage and review [the privacy guide](docs/privacy.md).

Current limitations: model capability claims are adapter-level rather than a versioned model
registry; provider APIs require explicit model selection and credentials; prices are supplied by
the caller; no fallback schema transformations, caching, replay, Azure, Bedrock, or LLM judge is
included. See [the roadmap](docs/roadmap.md).

## Development and contributing

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src
uv run --no-sync pytest
uv build
```

Read [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/architecture.md), and the
[provider capability notes](docs/provider-capabilities.md) before proposing substantial changes.
