"""LiteLLMTransport tests with a fake litellm module — no real dep, no network."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from mlteam.llm import (
    LLMClient,
    LLMRefusalError,
    LLMRequest,
    estimate_cost_usd,
)
from mlteam.llm.pricing import TokenUsage


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int = 1000
    completion_tokens: int = 200
    prompt_tokens_details: object | None = None


@dataclass
class _Response:
    choices: list
    usage: _Usage
    model: str = "anthropic/claude-sonnet-4-6"


@dataclass
class _FakeLiteLLM(types.ModuleType):
    """Stand-in for the litellm module."""

    supports_schema: bool = True
    calls: list[dict] = field(default_factory=list)
    scripted: list[str] = field(default_factory=list)
    drop_params: bool = False

    def __init__(self, **kwargs):
        types.ModuleType.__init__(self, "litellm")
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "calls"):
            self.calls = []

    def supports_response_schema(self, model: str) -> bool:
        return self.supports_schema

    def supports_response_format(self, model: str) -> bool:
        return self.supports_schema

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        text = self.scripted[len(self.calls) - 1]
        return _Response(choices=[_Choice(message=_Msg(content=text))], usage=_Usage())


@pytest.fixture()
def fake_litellm(monkeypatch):
    def _install(scripted, supports_schema=True):
        module = _FakeLiteLLM(scripted=list(scripted), supports_schema=supports_schema)
        monkeypatch.setitem(sys.modules, "litellm", module)
        return module

    return _install


class Answer:  # minimal schema via pydantic
    pass


def _answer_model():
    from pydantic import BaseModel, Field

    class Answer(BaseModel):
        label: str
        confidence: float = Field(ge=0.0, le=1.0)

    return Answer


def test_native_response_format_path(fake_litellm):
    from mlteam.llm import LiteLLMTransport

    module = fake_litellm(['{"label": "churn", "confidence": 0.9}'], supports_schema=True)
    client = LLMClient(LiteLLMTransport(), model="anthropic/claude-sonnet-4-6")
    result = client.structured(
        agent="t", system_prompt="s", user_payload={"q": 1}, schema=_answer_model()
    )
    assert result.output.label == "churn"
    # Native schema mode: response_format sent, schema NOT appended to prompt.
    call = module.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert "JSON Schema" not in call["messages"][1]["content"]


def test_prompt_fallback_when_schema_unsupported(fake_litellm):
    from mlteam.llm import LiteLLMTransport

    module = fake_litellm(['{"label": "x", "confidence": 0.5}'], supports_schema=False)
    client = LLMClient(LiteLLMTransport(), model="ollama/llama3")
    client.structured(agent="t", system_prompt="s", user_payload={"q": 1}, schema=_answer_model())
    call = module.calls[0]
    assert "response_format" not in call  # no native support
    assert "JSON Schema" in call["messages"][1]["content"]  # embedded in prompt


def test_validation_retry_works_over_litellm(fake_litellm):
    from mlteam.llm import LiteLLMTransport

    fake_litellm(["not json", '{"label": "x", "confidence": 0.5}'], supports_schema=True)
    client = LLMClient(LiteLLMTransport(), model="openai/gpt-4o")
    result = client.structured(
        agent="t", system_prompt="s", user_payload={}, schema=_answer_model()
    )
    assert result.attempts == 2


def test_content_filter_raises_refusal(fake_litellm):
    from mlteam.llm import LiteLLMTransport

    module = fake_litellm(["blocked"], supports_schema=True)

    def filtered_completion(**kwargs):
        module.calls.append(kwargs)
        return _Response(
            choices=[_Choice(message=_Msg(content=""), finish_reason="content_filter")],
            usage=_Usage(),
        )

    module.completion = filtered_completion
    transport = LiteLLMTransport()
    with pytest.raises(LLMRefusalError):
        transport.complete(
            LLMRequest(
                model="anthropic/claude-sonnet-4-6",
                system="s",
                user_content="hi",
                output_schema={"type": "object"},
                max_tokens=100,
            )
        )


def test_pricing_resolves_provider_prefix():
    # "anthropic/claude-sonnet-4-6" must price like "claude-sonnet-4-6".
    prefixed = estimate_cost_usd(
        "anthropic/claude-sonnet-4-6", TokenUsage(input_tokens=1000, output_tokens=500)
    )
    bare = estimate_cost_usd("claude-sonnet-4-6", TokenUsage(input_tokens=1000, output_tokens=500))
    assert prefixed == bare == pytest.approx(0.0105)


def test_pricing_resolves_date_suffix():
    assert estimate_cost_usd(
        "claude-haiku-4-5-20251001", TokenUsage(input_tokens=1000)
    ) == pytest.approx(0.001)
