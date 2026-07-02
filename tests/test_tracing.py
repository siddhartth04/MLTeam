"""Tracing tests: JSONL trace round-trip and RunTrace folding."""

from __future__ import annotations

import pytest

from mlteam.contracts import TraceEventType
from mlteam.observability import TraceWriter, build_run_trace, load_trace_events


def test_emit_and_load_round_trip(tmp_path):
    writer = TraceWriter("run-42", tmp_path / "traces")
    writer.emit(TraceEventType.AGENT_DECISION, agent="profiler", message="profiling started")
    writer.emit(
        TraceEventType.LLM_CALL,
        agent="cleaner",
        message="proposed 3 ops",
        tokens_in=1200,
        tokens_out=300,
        cost_usd=0.012,
        model="claude-sonnet-4-6",
    )

    events = load_trace_events(writer.path)
    assert len(events) == 2
    assert events[0].event_type is TraceEventType.AGENT_DECISION
    assert events[1].extra["model"] == "claude-sonnet-4-6"
    assert events[1].cost_usd == 0.012


def test_build_run_trace_totals(tmp_path):
    writer = TraceWriter("run-43", tmp_path / "traces")
    writer.emit(TraceEventType.LLM_CALL, agent="a", tokens_in=100, tokens_out=50, cost_usd=0.01)
    writer.emit(TraceEventType.LLM_CALL, agent="b", tokens_in=200, tokens_out=100, cost_usd=0.02)

    trace = build_run_trace("run-43", writer.path)
    assert trace.meta.run_id == "run-43"
    assert trace.total_tokens == 450
    assert trace.total_cost_usd == pytest.approx(0.03)
    assert len(trace.events) == 2


def test_load_missing_trace_returns_empty(tmp_path):
    assert load_trace_events(tmp_path / "nope.jsonl") == ()
