"""LLM client tests: validation retry loop, cost tracking, tracing, sanitizer."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from mlteam.llm import (
    FixturesExhaustedError,
    FixtureTransport,
    LLMClient,
    LLMValidationError,
    TokenUsage,
    estimate_cost_usd,
    output_schema_for,
)


class Answer(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


def _client(transport, **kwargs):
    return LLMClient(transport, model="claude-sonnet-4-6", **kwargs)


def test_valid_response_first_attempt():
    transport = FixtureTransport.from_texts('{"label": "churn", "confidence": 0.9}')
    result = _client(transport).structured(
        agent="t", system_prompt="s", user_payload={"q": 1}, schema=Answer
    )
    assert result.output.label == "churn"
    assert result.attempts == 1


def test_invalid_then_valid_retries_with_feedback():
    transport = FixtureTransport.from_texts(
        '{"label": "churn", "confidence": 2.5}',  # violates le=1.0 (client-side)
        '{"label": "churn", "confidence": 0.8}',
    )
    result = _client(transport).structured(
        agent="t", system_prompt="s", user_payload={"q": 1}, schema=Answer
    )
    assert result.attempts == 2
    # The retry prompt must carry the validation error back to the model.
    assert "failed validation" in transport.requests[1].user_content
    assert "confidence" in transport.requests[1].user_content


def test_exhausted_attempts_raises():
    transport = FixtureTransport.from_texts("not json", "not json", "still not json")
    with pytest.raises(LLMValidationError, match="after 3 attempts"):
        _client(transport).structured(agent="t", system_prompt="s", user_payload={}, schema=Answer)


def test_post_validate_feeds_retry():
    transport = FixtureTransport.from_texts(
        '{"label": "bad-column", "confidence": 0.9}',
        '{"label": "good", "confidence": 0.9}',
    )

    def post_validate(answer: Answer) -> None:
        if answer.label == "bad-column":
            raise ValueError("label 'bad-column' does not exist; pick 'good'")

    result = _client(transport).structured(
        agent="t",
        system_prompt="s",
        user_payload={},
        schema=Answer,
        post_validate=post_validate,
    )
    assert result.output.label == "good"
    assert "does not exist" in transport.requests[1].user_content


def test_cost_accumulates_across_attempts():
    transport = FixtureTransport(
        responses=[
            {"text": "broken", "usage": {"input_tokens": 1000, "output_tokens": 500}},
            {
                "text": '{"label": "x", "confidence": 0.5}',
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        ]
    )
    client = _client(transport)
    result = client.structured(agent="t", system_prompt="s", user_payload={}, schema=Answer)
    # sonnet-4-6: $3/M in, $15/M out -> (1000*3 + 500*15)/1e6 = 0.0105 per attempt
    assert result.cost_usd == pytest.approx(0.021)
    assert client.total_cost_usd == pytest.approx(0.021)
    assert client.total_usage.input_tokens == 2000


def test_trace_records_every_llm_call(tmp_path):
    from mlteam.observability import TraceWriter, load_trace_events

    writer = TraceWriter("run-llm", tmp_path)
    transport = FixtureTransport.from_texts("nope", '{"label": "x", "confidence": 0.1}')
    _client(transport, trace=writer).structured(
        agent="critic", system_prompt="s", user_payload={"a": 1}, schema=Answer
    )
    events = load_trace_events(writer.path)
    assert len(events) == 2  # one per attempt, failures included
    assert all(e.event_type.value == "llm_call" for e in events)
    assert events[0].agent == "critic"
    assert events[0].cost_usd > 0
    assert "prompt" in events[0].extra and "response" in events[0].extra


def test_fixture_exhaustion_is_loud():
    transport = FixtureTransport.from_texts('{"label": "x", "confidence": 0.1}')
    client = _client(transport)
    client.structured(agent="t", system_prompt="s", user_payload={}, schema=Answer)
    with pytest.raises(FixturesExhaustedError):
        client.structured(agent="t", system_prompt="s", user_payload={}, schema=Answer)


def test_pricing_unknown_model_returns_none():
    assert estimate_cost_usd("mystery-model", TokenUsage(input_tokens=100)) is None


def test_schema_sanitizer_strips_unsupported_constraints():
    schema = output_schema_for(Answer)
    text = json.dumps(schema)
    assert "maximum" not in text and "minimum" not in text  # ge/le stripped
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"label", "confidence"}
    assert set(schema["properties"]) == {"label", "confidence"}  # names survive


def test_schema_sanitizer_requires_fields_with_defaults():
    """OpenAI/Azure strict mode 400s if a defaulted field is missing from
    'required' (e.g. CostAsymmetryDraft.false_positive_cost = 1.0) — every
    property must be listed regardless of having a Python-side default."""

    class WithDefault(BaseModel):
        name: str
        cost: float = 1.0
        notes: str = ""

    schema = output_schema_for(WithDefault)
    assert set(schema["required"]) == {"name", "cost", "notes"}


def test_schema_sanitizer_handles_nested_models():
    class Inner(BaseModel):
        name: str = Field(min_length=1)

    class Outer(BaseModel):
        items: list[Inner]

    schema = output_schema_for(Outer)
    text = json.dumps(schema)
    assert "minLength" not in text
    inner_def = schema["$defs"]["Inner"]
    assert inner_def["additionalProperties"] is False


def test_schema_sanitizer_rejects_open_dict_fields():
    """The grammar demands additionalProperties:false on every object, which
    would silently force open dicts to be EMPTY — fail fast instead."""
    from typing import Any

    from mlteam.llm import UnsupportedSchemaError

    class Bad(BaseModel):
        params: dict[str, Any]

    with pytest.raises(UnsupportedSchemaError, match="open dict"):
        output_schema_for(Bad)


def test_all_production_draft_schemas_are_wire_safe():
    """Every agent draft schema must build without open-dict violations."""
    from mlteam.agents.cleaner import CleaningOpsDraft
    from mlteam.agents.critic import CritiqueDraft
    from mlteam.agents.experiment_planner import ExperimentPlanDraft
    from mlteam.agents.feature_engineer import FeaturePlanDraft
    from mlteam.agents.requirement import ProblemSpecDraft

    for draft in (
        ProblemSpecDraft,
        CleaningOpsDraft,
        FeaturePlanDraft,
        ExperimentPlanDraft,
        CritiqueDraft,
    ):
        schema = output_schema_for(draft)  # raises UnsupportedSchemaError if unsafe
        assert schema["additionalProperties"] is False
