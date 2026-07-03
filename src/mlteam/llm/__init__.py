"""LLM layer: schema-validated structured calls with cost tracking.

The LLM proposes; deterministic code validates, executes, and enforces.
Transports are provider-agnostic: AnthropicTransport (SDK), LiteLLMTransport
(any provider via litellm; optional dependency), plus fixture/recording.
"""

from typing import TYPE_CHECKING, Any

from mlteam.llm.client import LLMClient, LLMValidationError, StructuredCallResult
from mlteam.llm.pricing import PRICES_PER_MTOK, TokenUsage, estimate_cost_usd
from mlteam.llm.schema import UnsupportedSchemaError, output_schema_for
from mlteam.llm.summaries import evaluation_summary, profile_summary
from mlteam.llm.transport import (
    AnthropicTransport,
    FixturesExhaustedError,
    FixtureTransport,
    LLMRawResponse,
    LLMRefusalError,
    LLMRequest,
    RecordingTransport,
    Transport,
)

if TYPE_CHECKING:
    from mlteam.llm.litellm_transport import LiteLLMTransport


def __getattr__(name: str) -> Any:
    # Lazy so `import mlteam.llm` never requires the optional litellm extra.
    if name == "LiteLLMTransport":
        from mlteam.llm.litellm_transport import LiteLLMTransport

        return LiteLLMTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PRICES_PER_MTOK",
    "AnthropicTransport",
    "FixtureTransport",
    "FixturesExhaustedError",
    "LLMClient",
    "LLMRawResponse",
    "LLMRefusalError",
    "LLMRequest",
    "LLMValidationError",
    "LiteLLMTransport",
    "RecordingTransport",
    "StructuredCallResult",
    "TokenUsage",
    "Transport",
    "UnsupportedSchemaError",
    "estimate_cost_usd",
    "evaluation_summary",
    "output_schema_for",
    "profile_summary",
]
