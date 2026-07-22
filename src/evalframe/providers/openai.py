"""OpenAI strict structured-output adapter (optional dependency)."""

from __future__ import annotations

import json
import time
from typing import Any

from evalframe.exceptions import ProviderError
from evalframe.models import ModelRequest, ModelResponse, ProviderCapabilities


class OpenAIProvider:
    def __init__(self, *, api_key: str | None = None, client: Any = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Install OpenAI support with: pip install 'evalframe[openai]'"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_structured_output=True,
            strict_tool_schema=True,
            json_mode=True,
            streaming=True,
            usage_reporting=True,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        try:
            response = await self._client.responses.create(
                model=request.parameters.pop("model"),
                instructions=request.system_prompt,
                input=request.user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "output",
                        "strict": True,
                        "schema": request.output_json_schema,
                    }
                },
                **request.parameters,
            )
            parsed = json.loads(response.output_text)
            usage = response.usage
            return ModelResponse(
                raw_output=response.output_text,
                parsed_output=parsed,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                provider_request_id=getattr(response, "id", None),
                metadata={"finish_reason": "completed"},
            )
        except Exception as exc:
            raise ProviderError(
                str(exc), retryable=type(exc).__name__ in {"RateLimitError", "APITimeoutError"}
            ) from exc
