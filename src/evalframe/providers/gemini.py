"""Google Gemini structured-output adapter (optional dependency)."""

from __future__ import annotations

import json
import time
from typing import Any

from evalframe.exceptions import ProviderError
from evalframe.models import ModelRequest, ModelResponse, ProviderCapabilities


class GeminiProvider:
    def __init__(self, *, api_key: str | None = None, client: Any = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Install Gemini support with: pip install 'evalframe[gemini]'"
            ) from exc
        self._client = genai.Client(api_key=api_key).aio

    @property
    def name(self) -> str:
        return "gemini"

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
        parameters = dict(request.parameters)
        model = str(parameters.pop("model"))
        try:
            response = await self._client.models.generate_content(
                model=model,
                contents=request.user_input,
                config={
                    "system_instruction": request.system_prompt,
                    "response_mime_type": "application/json",
                    "response_json_schema": request.output_json_schema,
                    **parameters,
                },
            )
            parsed = json.loads(response.text)
            usage = getattr(response, "usage_metadata", None)
            return ModelResponse(
                raw_output=response.text,
                parsed_output=parsed,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                provider_request_id=getattr(response, "response_id", None),
                metadata={"finish_reason": "stop"},
            )
        except Exception as exc:
            raise ProviderError(
                str(exc), retryable=type(exc).__name__ in {"ClientError", "ServerError"}
            ) from exc
