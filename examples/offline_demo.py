"""Run with: uv run python examples/offline_demo.py"""

import asyncio
import json
from pathlib import Path

from schema import SupportAnswer

from evalframe import Candidate, EvaluationRunner, TaskDefinition
from evalframe.providers.fake import FakeProvider
from evalframe.reporters import ConsoleReporter, JsonReporter

HERE = Path(__file__).parent


def classify(request: object) -> dict[str, object]:
    message = json.loads(request.user_input)["message"].lower()
    mapping = {"charge": "billing", "error": "technical", "password": "account"}
    category = next((value for key, value in mapping.items() if key in message), "other")
    return {
        "category": category,
        "answer": "Thanks—our team can help.",
        "should_escalate": category == "billing",
    }


def weak(request: object) -> dict[str, object]:
    result = classify(request)
    result["category"] = "other"
    result["should_escalate"] = False
    return result


async def main() -> None:
    task = TaskDefinition(
        name="support-classification",
        dataset=HERE / "data/support_eval.jsonl",
        output_model=SupportAnswer,
        repetitions=3,
    )
    candidates = [
        Candidate(name="careful", provider="careful-fake", model="offline", prompt="Classify."),
        Candidate(name="weak", provider="weak-fake", model="offline", prompt="Classify."),
    ]
    report = await EvaluationRunner(
        {
            "careful-fake": FakeProvider(classify, provider_name="careful-fake", latency_ms=2),
            "weak-fake": FakeProvider(weak, provider_name="weak-fake"),
        }
    ).run(task, candidates)
    ConsoleReporter().report(report)
    print(f"Saved {JsonReporter(HERE / 'report.json').report(report)}")


if __name__ == "__main__":
    asyncio.run(main())
