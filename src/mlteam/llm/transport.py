"""LLM transports: live Anthropic API, fixture replay, and recording.

The client depends only on the Transport protocol, so tests replay recorded
fixtures with zero network access and the live path is one thin adapter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from mlteam.llm.pricing import TokenUsage

_log = structlog.get_logger(__name__)


class LLMRefusalError(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


class FixturesExhaustedError(RuntimeError):
    """FixtureTransport ran out of recorded responses."""


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user_content: str
    output_schema: dict[str, Any]
    max_tokens: int


@dataclass(frozen=True)
class LLMRawResponse:
    text: str
    usage: TokenUsage
    model: str
    latency_seconds: float


@runtime_checkable
class Transport(Protocol):
    def complete(self, request: LLMRequest) -> LLMRawResponse: ...


class AnthropicTransport:
    """Live transport over the official anthropic SDK.

    Credentials resolve from the environment (ANTHROPIC_API_KEY or an
    `ant auth login` profile) — never hardcoded. The SDK retries 429/5xx
    itself; schema-validation retries live in LLMClient, not here.
    """

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, request: LLMRequest) -> LLMRawResponse:
        started = time.perf_counter()
        response = self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user_content}],
            output_config={"format": {"type": "json_schema", "schema": request.output_schema}},
        )
        latency = time.perf_counter() - started
        if response.stop_reason == "refusal":
            raise LLMRefusalError("model declined the request (stop_reason=refusal)")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        return LLMRawResponse(text=text, usage=usage, model=response.model, latency_seconds=latency)


@dataclass
class FixtureTransport:
    """Replays recorded responses in order; captures requests for assertions."""

    responses: list[dict[str, Any]]
    requests: list[LLMRequest] = field(default_factory=list)
    _cursor: int = 0

    @classmethod
    def from_texts(cls, *texts: str) -> FixtureTransport:
        return cls(responses=[{"text": t} for t in texts])

    @classmethod
    def from_file(cls, path: Path) -> FixtureTransport:
        """Load fixtures from a JSON file: a list of {text | json, usage?}."""
        entries = json.loads(path.read_text(encoding="utf-8"))
        responses = []
        for entry in entries:
            if "json" in entry:
                entry = {**entry, "text": json.dumps(entry["json"])}
            responses.append(entry)
        return cls(responses=responses)

    def complete(self, request: LLMRequest) -> LLMRawResponse:
        self.requests.append(request)
        if self._cursor >= len(self.responses):
            raise FixturesExhaustedError(
                f"fixture transport exhausted after {len(self.responses)} responses"
            )
        entry = self.responses[self._cursor]
        self._cursor += 1
        usage_data = entry.get("usage", {})
        return LLMRawResponse(
            text=entry["text"],
            usage=TokenUsage(
                input_tokens=int(usage_data.get("input_tokens", 1000)),
                output_tokens=int(usage_data.get("output_tokens", 200)),
            ),
            model=entry.get("model", request.model),
            latency_seconds=float(entry.get("latency_seconds", 0.01)),
        )


class RecordingTransport:
    """Wraps a live transport and appends request/response pairs to a JSONL
    file, so real interactions can be turned into replayable fixtures."""

    def __init__(self, inner: Transport, record_path: Path) -> None:
        self._inner = inner
        self._path = record_path
        record_path.parent.mkdir(parents=True, exist_ok=True)

    def complete(self, request: LLMRequest) -> LLMRawResponse:
        response = self._inner.complete(request)
        record = {
            "request": {
                "model": request.model,
                "system": request.system,
                "user_content": request.user_content,
            },
            "text": response.text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "latency_seconds": response.latency_seconds,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _log.info("llm_interaction_recorded", path=str(self._path))
        return response
