"""Pluggable token price registry. Prices are USD per million tokens."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenPrice:
    input_per_million: float
    output_per_million: float


@dataclass
class PriceRegistry:
    prices: dict[tuple[str, str], TokenPrice] = field(default_factory=dict)

    def estimate(
        self, provider: str, model: str, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        price = self.prices.get((provider, model))
        if price is None or input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * price.input_per_million + output_tokens * price.output_per_million
        ) / 1_000_000
