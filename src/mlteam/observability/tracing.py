"""Per-run trace: append-only JSONL of every agent decision, call, and cost.

The trace file is written incrementally (crash-safe); at run end it is folded
into an immutable ``RunTrace`` artifact and stored like any other artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlteam.contracts import RunTrace, TraceEvent, TraceEventType, new_artifact


class TraceWriter:
    """Appends TraceEvents for one run to a JSONL file."""

    def __init__(self, run_id: str, trace_dir: Path) -> None:
        self._run_id = run_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._path = trace_dir / f"{run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def emit(
        self,
        event_type: TraceEventType,
        agent: str,
        message: str = "",
        *,
        artifact_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        **extra: Any,
    ) -> TraceEvent:
        """Validate, append, and return one trace event."""
        event = TraceEvent(
            event_type=event_type,
            agent=agent,
            message=message,
            artifact_id=artifact_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            extra=extra,
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event


def load_trace_events(path: Path) -> tuple[TraceEvent, ...]:
    """Read and validate all events from a trace JSONL file."""
    if not path.exists():
        return ()
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(TraceEvent.model_validate(json.loads(line)))
    return tuple(events)


def build_run_trace(run_id: str, trace_path: Path, producer_agent: str = "tracer") -> RunTrace:
    """Fold a trace file into an immutable RunTrace artifact."""
    events = load_trace_events(trace_path)
    return new_artifact(
        RunTrace,
        producer_agent=producer_agent,
        run_id=run_id,
        events=events,
        total_tokens=sum(e.tokens_in + e.tokens_out for e in events),
        total_cost_usd=sum(e.cost_usd for e in events),
    )
