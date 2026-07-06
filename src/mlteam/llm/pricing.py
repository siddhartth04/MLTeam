"""Model pricing table and cost estimation.

Prices are USD per million tokens (input, output), sourced from the Anthropic
model catalog (cached 2026-06). Cache reads bill at ~0.1x input, cache writes
at ~1.25x input.

The table only ever covers Anthropic models by design — this platform is
provider-agnostic (LiteLLM routes to OpenAI, Azure, Bedrock, Ollama, any
provider a user configures), and hardcoding one entry per third-party model
snapshot is a losing game: every dated release on every provider needs a new
row, forever. Instead, ANY model the table misses falls back to an optional
env-configured flat rate (MLTEAM_LLM_PRICE_INPUT_PER_M /
MLTEAM_LLM_PRICE_OUTPUT_PER_M) — set once per deployment for whichever
non-Anthropic model is actually in use. Only when neither the table nor the
override resolves does cost tracking go dark, and that is always logged
loudly, never silently treated as free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import structlog

_log = structlog.get_logger(__name__)

_ENV_INPUT_PRICE = "MLTEAM_LLM_PRICE_INPUT_PER_M"
_ENV_OUTPUT_PRICE = "MLTEAM_LLM_PRICE_OUTPUT_PER_M"

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


def _resolve_prices(model: str) -> tuple[float, float] | None:
    """Look up prices, tolerating LiteLLM provider prefixes and date suffixes.

    e.g. "anthropic/claude-sonnet-4-6" and "claude-sonnet-4-6-20251114" both
    resolve to the "claude-sonnet-4-6" row.
    """
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    bare = model.split("/", 1)[-1]  # strip provider prefix
    if bare in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[bare]
    # Longest known id that is a prefix of the bare model (handles date suffix).
    matches = sorted(
        (key for key in PRICES_PER_MTOK if bare.startswith(key)), key=len, reverse=True
    )
    return PRICES_PER_MTOK[matches[0]] if matches else None


def _env_override_prices() -> tuple[float, float] | None:
    """Flat per-MTok rate for any model the table doesn't know, set once per
    deployment via env vars. Malformed values are treated as absent (with a
    warning) rather than crashing the run over a cost-tracking side channel.
    """
    raw_input = os.environ.get(_ENV_INPUT_PRICE)
    raw_output = os.environ.get(_ENV_OUTPUT_PRICE)
    if raw_input is None or raw_output is None:
        return None
    try:
        return float(raw_input), float(raw_output)
    except ValueError:
        _log.warning(
            "invalid_price_override",
            **{_ENV_INPUT_PRICE: raw_input, _ENV_OUTPUT_PRICE: raw_output},
            hint="both must be floats; falling back to unknown-model handling",
        )
        return None


def estimate_cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Estimated cost of one call.

    Falls back to the MLTEAM_LLM_PRICE_*_PER_M env override for any model the
    table doesn't recognize (any non-Anthropic provider). Returns None (and a
    loud log) only when neither the table nor the override resolves.
    """
    prices = _resolve_prices(model)
    if prices is None:
        prices = _env_override_prices()
        if prices is None:
            _log.warning("unknown_model_pricing", model=model, hint="cost tracking disabled")
            return None
        _log.info("using_price_override", model=model, input_rate=prices[0], output_rate=prices[1])
    input_rate, output_rate = prices
    cost = (
        usage.input_tokens * input_rate
        + usage.output_tokens * output_rate
        + usage.cache_creation_input_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * input_rate * CACHE_READ_MULTIPLIER
    ) / 1_000_000
    return round(cost, 8)
