"""LLM layer: schema-validated structured calls with cost tracking.

The LLM proposes; deterministic code validates, executes, and enforces.
"""

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
