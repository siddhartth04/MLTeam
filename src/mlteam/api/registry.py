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

import structlog

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
from mlteam.api.run_store import RunStore
from mlteam.contracts.base import utcnow

_log = structlog.get_logger(__name__)

_MAX_EVENTS = 2000
_QUEUE_MAXSIZE = 1000


class RunActiveError(RuntimeError):
    """Raised when deletion is attempted on a run that is still in flight."""


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
    name: str = ""
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=utcnow)
    stages: list[StageState] = field(default_factory=initial_stage_states)
    events: deque[LogEvent] = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))
    pending_gate: GateRequest | None = None
    result_run_id: str | None = None  # the mlteam pipeline run id (artifact run_id)
    final_report_id: str | None = None
    error: str | None = None
    llm_cost_usd: float = 0.0
    cancelled: bool = False  # set to True to signal the run thread to stop gracefully


class RunRegistry:
    """Holds all runs and their live streams.

    Optionally backed by a ``RunStore`` (attach via ``attach_store``) that
    write-persists every field a fresh process needs to reconstruct
    provenance after a restart: identity, goal/dataset, status, the
    pipeline run id, the final report id, and stage history. Events, the
    pending-gate handshake, and the cancellation flag stay in-memory only
    -- see run_store.py's module docstring for why persisting those would
    create unrecoverable phantom state.
    """

    def __init__(self, run_store: RunStore | None = None) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._subscribers: dict[str, list[queue.Queue[StreamEvent]]] = {}
        self._store: RunStore | None = run_store

    def attach_store(self, run_store: RunStore) -> None:
        """Bind a ``RunStore`` after construction (called from the API
        lifespan) so tests can construct a bare in-memory ``RunRegistry()``
        without touching disk unless they explicitly opt in."""
        with self._lock:
            self._store = run_store

    def load_from_store(self) -> None:
        """Populate the registry from persisted rows. Call once at startup,
        before accepting any request. A run persisted mid-flight (its
        background thread died with the old process) is loaded as FAILED
        with an honest error -- there is no thread left to ever finish it,
        so leaving it QUEUED/RUNNING would strand it forever in a state the
        dashboard can neither resolve nor explain."""
        if self._store is None:
            return
        with self._lock:
            for row in self._store.load_all():
                status = row["status"]
                error = row["error"]
                if status in self._ACTIVE_STATUSES:
                    status = RunStatus.FAILED
                    error = "interrupted by server restart"
                record = RunRecord(
                    run_id=row["ui_run_id"],
                    goal_text=row["goal_text"],
                    dataset_name=row["dataset_name"],
                    name=row["name"],
                    status=status,
                    created_at=row["created_at"],
                    stages=row["stages"],
                    result_run_id=row["result_run_id"],
                    final_report_id=row["final_report_id"],
                    error=error,
                    llm_cost_usd=row["llm_cost_usd"],
                )
                self._runs[record.run_id] = record
                self._subscribers[record.run_id] = []
                if status != row["status"]:
                    # The row's on-disk status is now stale; persist the
                    # correction immediately rather than waiting for some
                    # future mutation that may never come for a dead run.
                    self._persist(record)

    def _persist(self, record: RunRecord) -> None:
        """Write ``record``'s persistable fields to the store. Called while
        holding ``self._lock`` so the in-memory record and its persisted row
        never visibly diverge to a concurrent reader. A no-op when no store
        is attached (e.g. most unit tests)."""
        if self._store is None:
            return
        self._store.upsert(
            ui_run_id=record.run_id,
            name=record.name,
            goal_text=record.goal_text,
            dataset_name=record.dataset_name,
            status=record.status,
            created_at=record.created_at,
            result_run_id=record.result_run_id,
            final_report_id=record.final_report_id,
            error=record.error,
            llm_cost_usd=record.llm_cost_usd,
            stages=record.stages,
        )

    # -- lifecycle --------------------------------------------------------------

    def create(
        self, run_id: str, goal_text: str, dataset_name: str, name: str = ""
    ) -> RunRecord:
        with self._lock:
            record = RunRecord(
                run_id=run_id, goal_text=goal_text, dataset_name=dataset_name, name=name
            )
            self._runs[run_id] = record
            self._subscribers[run_id] = []
            self._persist(record)
            return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def rename(self, run_id: str, name: str) -> bool:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            record.name = name
            self._persist(record)
            return True

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    _ACTIVE_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.AWAITING_APPROVAL})

    def cancel(self, run_id: str) -> bool:
        """Signal a run to stop gracefully. The background thread checks this
        flag periodically and exits early, transitioning the run to CANCELLED.
        Not persisted (see class docstring): a restart already fails any
        run still active, making a separate cancelled flag moot."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            if record.status not in self._ACTIVE_STATUSES:
                return False  # already finished; nothing to cancel
            record.cancelled = True
        self._publish(
            run_id,
            StreamEvent(
                type="log",
                data={
                    "level": "warning",
                    "stage": None,
                    "message": "Cancellation requested; run will stop after current operation",
                }.copy(),
            ),
        )
        return True

    def is_cancelled(self, run_id: str) -> bool:
        """Check if a run has been cancelled. Used by run threads to detect stop signals."""
        with self._lock:
            record = self._runs.get(run_id)
            return record.cancelled if record else False

    def delete(self, run_id: str) -> bool:
        """Remove a run's record. Refuses to delete a run that is still running
        (not cancelled), since its background thread would keep writing to a
        vanished record. Use cancel() first if the run is in-flight."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            # Allow deletion only if the run is finished or cancelled/rejected.
            # Do NOT allow deletion of QUEUED or RUNNING unless cancelled.
            if record.status in (RunStatus.QUEUED, RunStatus.RUNNING) and not record.cancelled:
                raise RunActiveError(
                    f"run '{run_id}' is still {record.status.value} — "
                    "call cancel() first to stop it, then delete"
                )
            for q in self._subscribers.pop(run_id, []):
                with contextlib.suppress(queue.Full):
                    q.put_nowait(StreamEvent(type="done", data={"status": "deleted"}))
            del self._runs[run_id]
            if self._store is not None:
                self._store.delete(run_id)
            return True

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
            self._persist(record)
            self._publish(run_id, StreamEvent(type="status", data={"status": status.value}))

    def set_cost(self, run_id: str, cost_usd: float) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.llm_cost_usd = round(cost_usd, 6)
            self._persist(record)

    def set_result_run_id(self, run_id: str, result_run_id: str) -> None:
        """Record the mlteam pipeline's own run id once ``run_manager``
        learns it (only knowable after the pipeline call returns -- see
        run_manager.py). Replaces the previous direct
        ``record.result_run_id = ...`` mutation in run_manager.py/demo.py,
        which silently bypassed persistence entirely; every other field
        mutation goes through a registry method for exactly this reason."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            record.result_run_id = result_run_id
            self._persist(record)

    def update_stage(
        self,
        run_id: str,
        stage: Stage,
        status: StageStatus,
        detail: str = "",
    ) -> None:
        """Deliberately NOT persisted on every call: this fires many times
        per run (once per agent-flow node transition) and a stage snapshot
        mid-run is only meaningful while the run's own thread is alive to
        keep advancing it. A run interrupted by a restart is marked FAILED
        on reload (see ``load_from_store``) regardless of which stage it
        reached, so persisting every transition would be write volume spent
        on state that gets overwritten as FAILED the moment it would ever
        be read back. ``finish``/``open_gate``/``resolve_gate`` persist the
        stage list at the points a restart would actually want to recover
        an accurate history up to."""
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
            self._persist(record)
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
            self._persist(record)
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
            self._persist(record)
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
