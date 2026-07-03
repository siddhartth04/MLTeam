"""In-memory run registry: live state, event log, SSE fan-out, gate handshake.

Thread-safe. A run executes on a background thread and publishes updates here;
SSE subscribers drain per-connection queues. Gates block the run thread on a
threading.Event until the dashboard resolves them via ``resolve_gate``.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from mlteam.api.events import (
    GATE_STAGES,
    LogEvent,
    RunStatus,
    Stage,
    StageState,
    StageStatus,
    StreamEvent,
    initial_stage_states,
)
from mlteam.contracts.base import utcnow

_MAX_EVENTS = 2000
_QUEUE_MAXSIZE = 1000


@dataclass
class GateRequest:
    stage: Stage
    summary: dict
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    notes: str = ""


@dataclass
class RunRecord:
    run_id: str
    goal_text: str
    dataset_name: str
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=utcnow)
    stages: list[StageState] = field(default_factory=initial_stage_states)
    events: deque[LogEvent] = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))
    pending_gate: GateRequest | None = None
    result_run_id: str | None = None  # the mlteam pipeline run id (artifact run_id)
    final_report_id: str | None = None
    error: str | None = None
    llm_cost_usd: float = 0.0


class RunRegistry:
    """Holds all runs and their live streams."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._subscribers: dict[str, list[queue.Queue[StreamEvent]]] = {}

    # -- lifecycle --------------------------------------------------------------

    def create(self, run_id: str, goal_text: str, dataset_name: str) -> RunRecord:
        with self._lock:
            record = RunRecord(run_id=run_id, goal_text=goal_text, dataset_name=dataset_name)
            self._runs[run_id] = record
            self._subscribers[run_id] = []
            return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    # -- publishing -------------------------------------------------------------

    def _publish(self, run_id: str, event: StreamEvent) -> None:
        for q in list(self._subscribers.get(run_id, [])):
            # A stuck subscriber must never block the run thread.
            with contextlib.suppress(queue.Full):
                q.put_nowait(event)

    def set_status(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = status
            self._publish(run_id, StreamEvent(type="status", data={"status": status.value}))

    def set_cost(self, run_id: str, cost_usd: float) -> None:
        with self._lock:
            self._runs[run_id].llm_cost_usd = round(cost_usd, 6)

    def update_stage(
        self,
        run_id: str,
        stage: Stage,
        status: StageStatus,
        detail: str = "",
    ) -> None:
        with self._lock:
            record = self._runs[run_id]
            for state in record.stages:
                if state.stage is stage:
                    state.status = status
                    if detail:
                        state.detail = detail
                    if status is StageStatus.ACTIVE and state.started_at is None:
                        state.started_at = utcnow()
                    if status in (StageStatus.DONE, StageStatus.SKIPPED, StageStatus.FAILED):
                        state.ended_at = utcnow()
                    payload = state.model_dump(mode="json")
                    break
            else:  # pragma: no cover - stage always exists
                return
        self._publish(run_id, StreamEvent(type="stage", data=payload))

    def log(self, run_id: str, event: LogEvent) -> None:
        with self._lock:
            self._runs[run_id].events.append(event)
        self._publish(run_id, StreamEvent(type="log", data=event.model_dump(mode="json")))

    def finish(
        self,
        run_id: str,
        *,
        status: RunStatus,
        final_report_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = status
            record.final_report_id = final_report_id
            record.error = error
        self._publish(run_id, StreamEvent(type="status", data={"status": status.value}))
        self._publish(run_id, StreamEvent(type="done", data={"status": status.value}))

    # -- gates ------------------------------------------------------------------

    def open_gate(self, run_id: str, stage: Stage, summary: dict) -> GateRequest:
        """Register a pending gate and mark the run awaiting approval."""
        request = GateRequest(stage=stage, summary=summary)
        with self._lock:
            record = self._runs[run_id]
            record.pending_gate = request
            record.status = RunStatus.AWAITING_APPROVAL
            for state in record.stages:
                if state.stage is stage:
                    state.status = StageStatus.AWAITING_APPROVAL
                    break
        self._publish(
            run_id,
            StreamEvent(
                type="status",
                data={"status": RunStatus.AWAITING_APPROVAL.value, "gate": stage.value},
            ),
        )
        self.log(
            run_id,
            LogEvent(level="gate", stage=stage, message=f"Awaiting human approval: {stage.value}"),
        )
        return request

    def resolve_gate(self, run_id: str, approved: bool, notes: str = "") -> bool:
        """Resolve a pending gate; unblocks the waiting run thread."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None or record.pending_gate is None:
                return False
            request = record.pending_gate
            request.approved = approved
            request.notes = notes
            stage = request.stage
            record.pending_gate = None
            if approved:
                record.status = RunStatus.RUNNING
                new_status = StageStatus.DONE
            else:
                record.status = RunStatus.REJECTED
                new_status = StageStatus.REJECTED
            for state in record.stages:
                if state.stage is stage:
                    state.status = new_status
                    state.ended_at = utcnow()
                    break
        request.event.set()
        self.log(
            run_id,
            LogEvent(
                level="gate" if approved else "warning",
                stage=stage,
                message=f"Gate {stage.value} {'approved' if approved else 'rejected'}",
            ),
        )
        return True

    def pending_gates(self) -> list[tuple[str, GateRequest]]:
        with self._lock:
            return [
                (rid, r.pending_gate) for rid, r in self._runs.items() if r.pending_gate is not None
            ]

    # -- subscriptions ----------------------------------------------------------

    def subscribe(self, run_id: str) -> queue.Queue[StreamEvent]:
        q: queue.Queue[StreamEvent] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(q)
            record = self._runs.get(run_id)
            snapshot = _snapshot_event(record) if record else None
        if snapshot is not None:
            q.put_nowait(snapshot)
        return q

    def unsubscribe(self, run_id: str, q: queue.Queue[StreamEvent]) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if q in subs:
                subs.remove(q)


def _snapshot_event(record: RunRecord) -> StreamEvent:
    return StreamEvent(
        type="snapshot",
        data={
            "run_id": record.run_id,
            "status": record.status.value,
            "stages": [s.model_dump(mode="json") for s in record.stages],
            "events": [e.model_dump(mode="json") for e in list(record.events)[-100:]],
            "llm_cost_usd": record.llm_cost_usd,
            "gate": record.pending_gate.stage.value if record.pending_gate else None,
        },
    )


def is_gate(stage: Stage) -> bool:
    return stage in GATE_STAGES


# Process-wide singleton registry (the API and run manager share it).
REGISTRY = RunRegistry()
