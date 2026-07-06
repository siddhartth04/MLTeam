"""SQLite lineage index: who produced what, from what, with which config."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mlteam.contracts import ArtifactMeta

_SQLITE_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    producer_agent TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parents (
    child_id  TEXT NOT NULL REFERENCES artifacts(artifact_id),
    parent_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    PRIMARY KEY (child_id, parent_id)
);
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id      TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    PRIMARY KEY (run_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_parents_parent ON parents(parent_id);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_run ON run_artifacts(run_id);
"""


@dataclass(frozen=True)
class LineageRecord:
    artifact_id: str
    artifact_type: str
    producer_agent: str
    config_hash: str
    schema_version: str
    run_id: str
    created_at: str


class LineageIndex:
    """Append-only provenance index over artifact metadata."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connection that commits/rolls back AND closes on exit.

        ``with sqlite3.connect(...)`` alone only manages the transaction; the
        unclosed handle keeps a file lock that breaks temp-dir cleanup on
        Windows.
        """
        conn = sqlite3.connect(self._db_path, timeout=_SQLITE_TIMEOUT_SECONDS)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn
        finally:
            conn.close()

    def record(self, meta: ArtifactMeta) -> None:
        """Insert an artifact's provenance; idempotent for identical ids.

        Parent ids must already be recorded — dangling lineage edges would
        silently corrupt the provenance graph, so they fail fast instead.

        ``artifacts.run_id`` stays "first producer" provenance: content
        addressing means two runs over identical inputs mint the SAME
        artifact id, and ``INSERT OR IGNORE`` means only the first writer's
        run_id is kept there. The ``run_artifacts`` membership table below is
        written on every call regardless — a many-to-many pairing so a later
        run that reuses/dedupes an earlier run's artifact still sees it in
        its OWN lineage graph, instead of the graph silently belonging to
        whichever run happened to produce that artifact first.
        """
        created = meta.created_at
        created_str = created.isoformat() if isinstance(created, datetime) else str(created)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        meta.artifact_id,
                        meta.artifact_type,
                        meta.producer_agent,
                        meta.config_hash,
                        meta.schema_version,
                        meta.run_id,
                        created_str,
                    ),
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO parents VALUES (?, ?)",
                    [(meta.artifact_id, parent) for parent in meta.parent_ids],
                )
                conn.execute(
                    "INSERT OR IGNORE INTO run_artifacts VALUES (?, ?)",
                    (meta.run_id, meta.artifact_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"cannot record lineage for '{meta.artifact_id}': one of its "
                f"parents {meta.parent_ids} is not in the lineage index; "
                "store parent artifacts before their children"
            ) from exc

    def artifact_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        """Every artifact ever written under this run_id, including shared
        (deduped) artifacts first produced by a different run. Empty for
        pre-existing stores with no membership rows recorded yet — callers
        should fall back to the legacy ``artifacts.run_id`` filter in that
        case so old runs keep rendering."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT artifact_id FROM run_artifacts WHERE run_id = ?", (run_id,)
            ).fetchall()
        return tuple(row[0] for row in rows)

    def get_record(self, artifact_id: str) -> LineageRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_id, artifact_type, producer_agent, config_hash,"
                " schema_version, run_id, created_at FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return LineageRecord(*row) if row else None

    def parents(self, artifact_id: str) -> tuple[str, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT parent_id FROM parents WHERE child_id = ? ORDER BY parent_id",
                (artifact_id,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def children(self, artifact_id: str) -> tuple[str, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT child_id FROM parents WHERE parent_id = ? ORDER BY child_id",
                (artifact_id,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def ancestry(self, artifact_id: str) -> tuple[str, ...]:
        """All transitive ancestors of an artifact (recursive CTE)."""
        query = """
        WITH RECURSIVE lineage(aid) AS (
            SELECT parent_id FROM parents WHERE child_id = :root
            UNION
            SELECT p.parent_id FROM parents p JOIN lineage l ON p.child_id = l.aid
        )
        SELECT aid FROM lineage ORDER BY aid
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"root": artifact_id}).fetchall()
        return tuple(row[0] for row in rows)

    def list_ids(self, artifact_type: str | None = None) -> tuple[str, ...]:
        with self._connect() as conn:
            if artifact_type is None:
                rows = conn.execute(
                    "SELECT artifact_id FROM artifacts ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT artifact_id FROM artifacts WHERE artifact_type = ? ORDER BY created_at",
                    (artifact_type,),
                ).fetchall()
        return tuple(row[0] for row in rows)
