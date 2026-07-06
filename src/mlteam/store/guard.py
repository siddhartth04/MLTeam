"""Store-level holdout access guard.

The frozen holdout must never be visible to any agent or iteration loop until
final reporting. This is enforced HERE, at the store boundary — not by
convention. Protected ids are persisted in SQLite so protection survives
process restarts, and every access attempt (granted or denied) is logged so
the EvaluationReport can assert ``holdout_untouched`` from evidence.

Ordering guarantee: protection is registered BEFORE the artifact is written
to the inner store, so there is no window in which a stored holdout is
readable. ``repair()`` additionally re-scans the inner store on open to
re-protect anything a crash may have left unregistered.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import structlog

from mlteam.contracts import Artifact, ArtifactRef, DatasetVersion, utcnow
from mlteam.contracts.trace import TraceEventType
from mlteam.observability.tracing import TraceWriter
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_SQLITE_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS protected_items (
    item_id      TEXT PRIMARY KEY,
    reason       TEXT NOT NULL,
    protected_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS access_log (
    timestamp TEXT NOT NULL,
    item_id   TEXT NOT NULL,
    granted   INTEGER NOT NULL,
    context   TEXT NOT NULL
);
"""


class GuardAccessContext(StrEnum):
    """Where an access attempt came from; compared in holdout_untouched()."""

    DEFAULT = "default"
    MEMBERSHIP_PROBE = "membership_probe"
    FINAL_REPORTING = "final_reporting"


class HoldoutAccessError(PermissionError):
    """Raised when anything tries to read a protected holdout artifact/blob."""


@dataclass(frozen=True)
class GuardAccessRecord:
    timestamp: str
    item_id: str
    granted: bool
    context: str


class GuardedArtifactStore:
    """ArtifactStore wrapper that blocks reads of protected items.

    Artifacts declare their own protected items via
    ``Artifact.guard_protected_items()`` (e.g. a DatasetVersion with
    role=HOLDOUT protects its artifact id and data blob), so new artifact
    types can opt into protection without touching this module. Reads of
    protected items raise ``HoldoutAccessError`` unless made through the view
    yielded by ``final_reporting_access``.

    ``exists()``/``blob_exists()`` on protected items return their boolean but
    are logged as membership probes — a deliberate, documented 1-bit trade-off
    so orchestrator resume logic keeps working.
    """

    def __init__(
        self,
        inner: ArtifactStore,
        db_path: Path,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._inner: ArtifactStore = inner
        self._db_path = db_path
        self._trace_writer = trace_writer
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self.repair()

    @property
    def root(self) -> Path:
        """The store root directory (delegates to the wrapped store)."""
        return self._inner.root

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connection that commits/rolls back AND closes on exit (see lineage)."""
        conn = sqlite3.connect(self._db_path, timeout=_SQLITE_TIMEOUT_SECONDS)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # -- protection registry --------------------------------------------------

    def protect(self, item_id: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO protected_items VALUES (?, ?, ?)",
                (item_id, reason, utcnow().isoformat()),
            )
        _log.info("holdout_protected", item_id=item_id, reason=reason)

    def is_protected(self, item_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM protected_items WHERE item_id = ?", (item_id,)
            ).fetchone()
        return row is not None

    def repair(self) -> int:
        """Re-protect stored artifacts a crash may have left unregistered.

        Scans DatasetVersion artifacts in the inner store (the only Phase 0
        type with protected items) and registers any missing protections.
        Returns the number of items newly protected.
        """
        repaired = 0
        for artifact_id in self._inner.list_ids(DatasetVersion.artifact_type()):
            artifact = self._inner.get(artifact_id)
            for item_id, reason in artifact.guard_protected_items():
                if not self.is_protected(item_id):
                    self.protect(item_id, reason)
                    repaired += 1
        if repaired:
            _log.warning("holdout_guard_repaired", items=repaired)
        return repaired

    # -- internals --------------------------------------------------------------

    def _record_access(self, item_id: str, granted: bool, context: GuardAccessContext) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO access_log VALUES (?, ?, ?, ?)",
                (utcnow().isoformat(), item_id, int(granted), context.value),
            )
        if self._trace_writer is not None:
            self._trace_writer.emit(
                TraceEventType.GUARD_ACCESS_ATTEMPT,
                agent="store-guard",
                message=f"{'granted' if granted else 'denied'} ({context.value})",
                artifact_id=item_id,
                granted=granted,
                context=context.value,
            )

    def _check(self, item_id: str) -> None:
        if self.is_protected(item_id):
            self._record_access(item_id, granted=False, context=GuardAccessContext.DEFAULT)
            _log.warning("holdout_access_denied", item_id=item_id)
            raise HoldoutAccessError(
                f"'{item_id}' is a protected holdout item; it is only readable "
                "inside final_reporting_access()"
            )

    # -- ArtifactStore protocol -----------------------------------------------

    def put(self, artifact: Artifact) -> ArtifactRef:
        # Protect BEFORE the write-through so a stored holdout is never
        # readable, even across a crash between the two steps.
        for item_id, reason in artifact.guard_protected_items():
            self.protect(item_id, reason)
        return self._inner.put(artifact)

    def get(self, artifact_id: str) -> Artifact:
        self._check(artifact_id)
        return self._inner.get(artifact_id)

    def exists(self, artifact_id: str) -> bool:
        if self.is_protected(artifact_id):
            self._record_access(
                artifact_id, granted=False, context=GuardAccessContext.MEMBERSHIP_PROBE
            )
        return self._inner.exists(artifact_id)

    def list_ids(self, artifact_type: str | None = None) -> tuple[str, ...]:
        return self._inner.list_ids(artifact_type)

    def put_blob(self, data: bytes) -> str:
        return self._inner.put_blob(data)

    def put_blob_file(self, path: Path) -> str:
        return self._inner.put_blob_file(path)

    def blob_path(self, blob_hash: str) -> Path:
        self._check(blob_hash)
        return self._inner.blob_path(blob_hash)

    def blob_exists(self, blob_hash: str) -> bool:
        if self.is_protected(blob_hash):
            self._record_access(
                blob_hash, granted=False, context=GuardAccessContext.MEMBERSHIP_PROBE
            )
        return self._inner.blob_exists(blob_hash)

    # -- final reporting unlock -------------------------------------------------

    @contextmanager
    def final_reporting_access(self, reason: str) -> Iterator[_UnlockedView]:
        """Yield a view that may read protected items; every read is logged."""
        _log.info("holdout_unlocked", reason=reason)
        yield _UnlockedView(self)
        _log.info("holdout_relocked", reason=reason)

    # -- evidence ----------------------------------------------------------------

    def access_log(self) -> tuple[GuardAccessRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, item_id, granted, context FROM access_log ORDER BY timestamp"
            ).fetchall()
        return tuple(
            GuardAccessRecord(timestamp=r[0], item_id=r[1], granted=bool(r[2]), context=r[3])
            for r in rows
        )

    def holdout_untouched(self) -> bool:
        """True iff no protected item was ever read outside final reporting."""
        return not any(
            record.granted and record.context != GuardAccessContext.FINAL_REPORTING.value
            for record in self.access_log()
        )

    # -- run scoping --------------------------------------------------------

    def artifact_ids_for_run(self, run_id: str) -> frozenset[str] | None:
        """The run_artifacts membership-table view of a run's artifacts, or
        None to signal "no membership rows recorded" (a store written before
        the membership table existed) so callers fall back to the legacy
        ``artifacts.run_id`` (first-writer-wins) filter instead of showing an
        empty run. This is the ONE place "does artifact X belong to run Y?"
        is decided — every run-scoped view/query must go through it rather
        than re-deriving the fallback logic at each call site."""
        ids = self._inner.lineage.artifact_ids_for_run(run_id)
        return frozenset(ids) if ids else None

    def is_in_run(self, artifact_id: str, run_id: str, scoped: frozenset[str] | None) -> bool:
        """Membership check against a scope already fetched via
        ``artifact_ids_for_run`` (fetch once per run, check many artifacts)."""
        if scoped is not None:
            return artifact_id in scoped
        record = self._inner.lineage.get_record(artifact_id)
        return record is not None and record.run_id == run_id


class _UnlockedView:
    """Read view used ONLY inside final_reporting_access; logs granted reads."""

    def __init__(self, guard: GuardedArtifactStore) -> None:
        self._guard = guard

    def get(self, artifact_id: str) -> Artifact:
        if self._guard.is_protected(artifact_id):
            self._guard._record_access(
                artifact_id, granted=True, context=GuardAccessContext.FINAL_REPORTING
            )
        return self._guard._inner.get(artifact_id)

    def blob_path(self, blob_hash: str) -> Path:
        if self._guard.is_protected(blob_hash):
            self._guard._record_access(
                blob_hash, granted=True, context=GuardAccessContext.FINAL_REPORTING
            )
        return self._guard._inner.blob_path(blob_hash)
