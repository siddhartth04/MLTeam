"""Structured-output LLM client: every call returns a validated Pydantic model.

Contract (from the platform spec):
- All LLM calls return structured JSON validated against Pydantic schemas.
- On validation failure, retry with error feedback — max attempts configurable
  (default 3).
- Every call is logged and traced: prompt, response, tokens, latency, cost.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from mlteam.contracts import TraceEventType
from mlteam.llm.pricing import TokenUsage, estimate_cost_usd
from mlteam.llm.schema import output_schema_for
from mlteam.llm.transport import LLMRequest, Transport
from mlteam.observability.tracing import TraceWriter

_log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FEEDBACK_TEMPLATE = (
    "\n\n---\nYour previous response failed validation with these errors:\n"
    "{errors}\n"
    "Return ONLY corrected JSON that satisfies the schema. Do not repeat the mistake."
)


class LLMValidationError(RuntimeError):
    """Raised when the model cannot produce valid output within max attempts."""

    def __init__(self, agent: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"agent '{agent}' failed schema validation after {attempts} attempts: {last_error}"
        )
        self.agent = agent
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class StructuredCallResult(Generic[T]):
    output: T
    attempts: int
    usage: TokenUsage
    cost_usd: float
    latency_seconds: float


class LLMClient:
    """Schema-validated calls over a pluggable transport, with cost tracking."""

    def __init__(
        self,
        transport: Transport,
        *,
        model: str,
        trace: TraceWriter | None = None,
        max_attempts: int = 3,
        max_tokens: int = 8192,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._transport = transport
        self._model = model
        self._trace = trace
        self._max_attempts = max_attempts
        self._max_tokens = max_tokens
        self._total_usage = TokenUsage()
        self._total_cost_usd = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    def structured(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_payload: dict | str,
        schema: type[T],
        post_validate: Callable[[T], None] | None = None,
    ) -> StructuredCallResult[T]:
        """Call the model and validate the JSON response against ``schema``.

        ``post_validate`` may raise ValueError for domain checks the schema
        cannot express (e.g. "column must exist in the dataset"); its message
        is fed back to the model on retry exactly like a schema error.
        """
        base_content = (
            user_payload
            if isinstance(user_payload, str)
            else json.dumps(user_payload, indent=2, sort_keys=True, default=str)
        )
        wire_schema = output_schema_for(schema)

        usage = TokenUsage()
        cost = 0.0
        latency = 0.0
        feedback: str | None = None
        error_text = "no attempts made"

        for attempt in range(1, self._max_attempts + 1):
            content = base_content if feedback is None else base_content + feedback
            raw = self._transport.complete(
                LLMRequest(
                    model=self._model,
                    system=system_prompt,
                    user_content=content,
                    output_schema=wire_schema,
                    max_tokens=self._max_tokens,
                )
            )
            usage = usage + raw.usage
            latency += raw.latency_seconds
            attempt_cost = estimate_cost_usd(raw.model, raw.usage) or 0.0
            cost += attempt_cost
            self._total_usage = self._total_usage + raw.usage
            self._total_cost_usd = round(self._total_cost_usd + attempt_cost, 8)
            self._emit_trace(agent, attempt, content, raw.text, raw, attempt_cost)

            try:
                output = schema.model_validate(json.loads(raw.text))
                if post_validate is not None:
                    post_validate(output)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error_text = str(exc)[:2000]
                feedback = _FEEDBACK_TEMPLATE.format(errors=error_text)
                _log.warning(
                    "llm_validation_retry",
                    agent=agent,
                    attempt=attempt,
                    error=error_text[:300],
                )
                continue

            return StructuredCallResult(
                output=output,
                attempts=attempt,
                usage=usage,
                cost_usd=round(cost, 8),
                latency_seconds=round(latency, 4),
            )

        raise LLMValidationError(agent, self._max_attempts, error_text)

    def _emit_trace(
        self,
        agent: str,
        attempt: int,
        prompt: str,
        response_text: str,
        raw: object,
        cost: float,
    ) -> None:
        if self._trace is None:
            return
        from mlteam.llm.transport import LLMRawResponse

        assert isinstance(raw, LLMRawResponse)
        self._trace.emit(
            TraceEventType.LLM_CALL,
            agent=agent,
            message=f"structured call attempt {attempt}",
            tokens_in=raw.usage.input_tokens,
            tokens_out=raw.usage.output_tokens,
            cost_usd=cost,
            model=raw.model,
            attempt=attempt,
            latency_seconds=raw.latency_seconds,
            prompt=prompt,
            response=response_text,
        )
