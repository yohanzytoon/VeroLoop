"""Anthropic strict tool-schema adapter (optional dependency)."""

from __future__ import annotations

import time
from typing import Any

from evalframe.exceptions import ProviderError
from evalframe.models import ModelRequest, ModelResponse, ProviderCapabilities


class AnthropicProvider:
    def __init__(self, *, api_key: str | None = None, client: Any = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Install Anthropic support with: pip install 'evalframe[anthropic]'"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_structured_output=False,
            strict_tool_schema=True,
            json_mode=False,
            streaming=True,
            usage_reporting=True,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        parameters = dict(request.parameters)
        model = str(parameters.pop("model"))
        max_tokens = int(parameters.pop("max_tokens", 1024))
        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=request.system_prompt or "",
                messages=[{"role": "user", "content": request.user_input}],
                tools=[
                    {
                        "name": "submit_output",
                        "description": "Return structured output",
                        "input_schema": request.output_json_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": "submit_output"},
                **parameters,
            )
            block = next(
                item for item in response.content if getattr(item, "type", "") == "tool_use"
            )
            parsed = dict(block.input)
            return ModelResponse(
                raw_output=parsed,
                parsed_output=parsed,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=getattr(response.usage, "input_tokens", None),
                output_tokens=getattr(response.usage, "output_tokens", None),
                provider_request_id=getattr(response, "id", None),
                metadata={"finish_reason": getattr(response, "stop_reason", None)},
            )
        except Exception as exc:
            raise ProviderError(
                str(exc), retryable=type(exc).__name__ in {"RateLimitError", "APITimeoutError"}
            ) from exc
