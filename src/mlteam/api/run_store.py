"""Persistent run registry: the ui_run_id -> pipeline run_id -> report
provenance spine survives a backend restart.

Before this module, every ``RunRecord`` lived only in ``RunRegistry``'s
process memory. A restart erased the mapping from a dashboard run id to
its pipeline run id and final report id -- along with ``goal_text``, which
exists nowhere else -- while the artifact store (a separate SQLite/blob
store) kept every artifact intact. A run that had genuinely completed
became permanently unreachable from the dashboard the moment the API
process restarted, and the demo run was re-seeded (and re-appended to the
outcome ledger) on every restart because nothing recorded it had already
run. See REPORT_INTEGRITY_ROADMAP.md Task 4 for the full incident.

Connection-per-operation (mirrors ``GuardedArtifactStore._connect``): each
method opens, uses, and closes its own ``sqlite3.Connection`` rather than
holding one open across the registry's lifetime, so this module never has
to reason about connection lifetime across the registry's own thread
lock. Callers already serialize access through ``RunRegistry._lock``; this
store does not need its own separate locking for that reason, but keeps a
generous busy-timeout in case a stray external reader (e.g. manual
inspection) holds a lock briefly.

Events, the pending gate handshake, and the cancellation flag are
DELIBERATELY NOT persisted: events are a bounded, best-effort recent-
activity log (re-created empty on the next launch is fine), and a pending
gate or cancellation flag describes a background thread that no longer
exists after a restart -- persisting either would create a phantom
"awaiting approval" state nothing can ever resolve.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from mlteam.api.events import RunStatus, StageState

_stages_adapter = TypeAdapter(list[StageState])

_SQLITE_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    ui_run_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    goal_text       TEXT NOT NULL,
    dataset_name    TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    result_run_id   TEXT,
    final_report_id TEXT,
    error           TEXT,
    llm_cost_usd    REAL NOT NULL DEFAULT 0.0,
    stages_json     TEXT NOT NULL
);
"""


class RunStore:
    """SQLite-backed persistence for run provenance. One file per store
    root, separate from the artifact store's own ``guard.db`` -- run
    bookkeeping and artifact bookkeeping are different concerns with
    different lifecycles (deleting a run record must never touch artifact
    lineage, and vice versa)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=_SQLITE_TIMEOUT_SECONDS)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def upsert(
        self,
        *,
        ui_run_id: str,
        name: str,
        goal_text: str,
        dataset_name: str,
        status: RunStatus,
        created_at: datetime,
        result_run_id: str | None,
        final_report_id: str | None,
        error: str | None,
        llm_cost_usd: float,
        stages: list[StageState],
    ) -> None:
        """Write (insert or fully replace) one run's persisted row."""
        stages_json = _stages_adapter.dump_json(stages).decode("utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    ui_run_id, name, goal_text, dataset_name, status,
                    created_at, result_run_id, final_report_id, error,
                    llm_cost_usd, stages_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ui_run_id) DO UPDATE SET
                    name=excluded.name,
                    goal_text=excluded.goal_text,
                    dataset_name=excluded.dataset_name,
                    status=excluded.status,
                    result_run_id=excluded.result_run_id,
                    final_report_id=excluded.final_report_id,
                    error=excluded.error,
                    llm_cost_usd=excluded.llm_cost_usd,
                    stages_json=excluded.stages_json
                """,
                (
                    ui_run_id,
                    name,
                    goal_text,
                    dataset_name,
                    status.value,
                    created_at.isoformat(),
                    result_run_id,
                    final_report_id,
                    error,
                    llm_cost_usd,
                    stages_json,
                ),
            )

    def delete(self, ui_run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE ui_run_id = ?", (ui_run_id,))

    def load_all(self) -> list[dict]:
        """Every persisted run row as a plain dict, ``stages`` already
        parsed into ``StageState`` objects. Ordering is unspecified; callers
        that need creation order should sort by ``created_at`` themselves."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ui_run_id, name, goal_text, dataset_name, status,
                       created_at, result_run_id, final_report_id, error,
                       llm_cost_usd, stages_json
                FROM runs
                """
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: tuple) -> dict:
    (
        ui_run_id,
        name,
        goal_text,
        dataset_name,
        status,
        created_at,
        result_run_id,
        final_report_id,
        error,
        llm_cost_usd,
        stages_json,
    ) = row
    return {
        "ui_run_id": ui_run_id,
        "name": name,
        "goal_text": goal_text,
        "dataset_name": dataset_name,
        "status": RunStatus(status),
        "created_at": datetime.fromisoformat(created_at),
        "result_run_id": result_run_id,
        "final_report_id": final_report_id,
        "error": error,
        "llm_cost_usd": llm_cost_usd,
        "stages": _stages_adapter.validate_json(stages_json),
    }
