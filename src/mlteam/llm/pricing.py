"""Model pricing table and cost estimation.

Prices are USD per million tokens (input, output), sourced from the Anthropic
model catalog (cached 2026-06). Cache reads bill at ~0.1x input, cache writes
at ~1.25x input. Unknown models produce a None cost estimate — logged loudly,
never silently treated as free.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

_log = structlog.get_logger(__name__)

# model id -> (input $/MTok, output $/MTok)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def estimate_cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Estimated cost of one call; None (and a loud log) for unknown models."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        _log.warning("unknown_model_pricing", model=model, hint="cost tracking disabled")
        return None
    input_rate, output_rate = prices
    cost = (
        usage.input_tokens * input_rate
        + usage.output_tokens * output_rate
        + usage.cache_creation_input_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * input_rate * CACHE_READ_MULTIPLIER
    ) / 1_000_000
    return round(cost, 8)
