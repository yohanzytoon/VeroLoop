"""Deterministic offline provider for examples and tests."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from typing import Any

from evalframe.models import ModelRequest, ModelResponse, ProviderCapabilities


class FakeProvider:
    def __init__(
        self,
        responder: Callable[[ModelRequest], dict[str, Any]] | None = None,
        *,
        provider_name: str = "fake",
        latency_ms: float = 0,
    ) -> None:
        self._responder = responder or (lambda request: {"echo": request.user_input})
        self._name = provider_name
        self._latency_ms = latency_ms

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_structured_output=True,
            strict_tool_schema=True,
            json_mode=True,
            streaming=False,
            usage_reporting=True,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)
        output = self._responder(request)
        digest = hashlib.sha256(f"{self.name}:{request.case_id}".encode()).hexdigest()[:16]
        return ModelResponse(
            raw_output=output,
            parsed_output=output,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=max(1, len(str(request.user_input)) // 4),
            output_tokens=max(1, len(str(output)) // 4),
            provider_request_id=f"fake-{digest}",
            metadata={"finish_reason": "stop"},
        )
