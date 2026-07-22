"""Provider adapter contract."""

from __future__ import annotations

from typing import Protocol

from evalframe.models import ModelRequest, ModelResponse, ProviderCapabilities


class ProviderAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def capabilities(self) -> ProviderCapabilities: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...
