"""Holdout guard tests: the leakage firewall must be store-enforced."""

from __future__ import annotations

import pytest

from mlteam.contracts import DatasetRole
from mlteam.store import (
    GuardedArtifactStore,
    HoldoutAccessError,
    LocalArtifactStore,
)


def _put_holdout(store: GuardedArtifactStore, make_dataset, blob: bytes = b"holdout-rows"):
    blob_hash = store.put_blob(blob)
    dataset = make_dataset(role=DatasetRole.HOLDOUT, blob_hash=blob_hash)
    store.put(dataset)
    return dataset


def test_holdout_artifact_read_is_blocked(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    with pytest.raises(HoldoutAccessError):
        guarded_store.get(dataset.meta.artifact_id)


def test_holdout_blob_read_is_blocked(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    with pytest.raises(HoldoutAccessError):
        guarded_store.blob_path(dataset.blob_hash)


def test_non_holdout_reads_are_unaffected(guarded_store, make_dataset):
    dataset = make_dataset(role=DatasetRole.TRAIN)
    guarded_store.put(dataset)
    assert guarded_store.get(dataset.meta.artifact_id) == dataset


def test_final_reporting_access_reads_holdout(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset, blob=b"secret rows")
    with guarded_store.final_reporting_access(reason="final report") as unlocked:
        loaded = unlocked.get(dataset.meta.artifact_id)
        assert loaded == dataset
        assert unlocked.blob_path(dataset.blob_hash).read_bytes() == b"secret rows"


def test_denied_attempts_are_logged(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    with pytest.raises(HoldoutAccessError):
        guarded_store.get(dataset.meta.artifact_id)
    denied = [r for r in guarded_store.access_log() if not r.granted]
    assert len(denied) == 1
    assert denied[0].item_id == dataset.meta.artifact_id


def test_holdout_untouched_semantics(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    assert guarded_store.holdout_untouched()

    # A denied attempt does not count as touching the holdout.
    with pytest.raises(HoldoutAccessError):
        guarded_store.get(dataset.meta.artifact_id)
    assert guarded_store.holdout_untouched()

    # A final-reporting read is legitimate and keeps the assertion true.
    with guarded_store.final_reporting_access(reason="final report") as unlocked:
        unlocked.get(dataset.meta.artifact_id)
    assert guarded_store.holdout_untouched()


def test_protection_survives_process_restart(tmp_path, make_dataset):
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    dataset = _put_holdout(store, make_dataset)

    reopened = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    with pytest.raises(HoldoutAccessError):
        reopened.get(dataset.meta.artifact_id)


def test_repair_reprotects_after_simulated_crash(tmp_path, make_dataset):
    """A holdout written directly to the inner store (as if a crash skipped
    protection) must be re-protected when the guard opens."""
    root = tmp_path / "store"
    inner = LocalArtifactStore(root)
    blob_hash = inner.put_blob(b"orphaned holdout rows")
    dataset = make_dataset(role=DatasetRole.HOLDOUT, blob_hash=blob_hash)
    inner.put(dataset)  # bypasses the guard entirely

    guard = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    with pytest.raises(HoldoutAccessError):
        guard.get(dataset.meta.artifact_id)
    with pytest.raises(HoldoutAccessError):
        guard.blob_path(blob_hash)


def test_membership_probe_is_logged_but_answered(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    assert guarded_store.exists(dataset.meta.artifact_id)
    assert guarded_store.blob_exists(dataset.blob_hash)

    probes = [r for r in guarded_store.access_log() if r.context == "membership_probe"]
    assert len(probes) == 2
    assert guarded_store.holdout_untouched()


def test_guard_emits_trace_events(tmp_path, make_dataset):
    from mlteam.observability import TraceWriter, load_trace_events

    root = tmp_path / "store"
    writer = TraceWriter("run-guard", tmp_path / "traces")
    store = GuardedArtifactStore(
        LocalArtifactStore(root), db_path=root / "guard.db", trace_writer=writer
    )
    dataset = _put_holdout(store, make_dataset)
    with pytest.raises(HoldoutAccessError):
        store.get(dataset.meta.artifact_id)

    events = load_trace_events(writer.path)
    assert len(events) == 1
    assert events[0].event_type.value == "guard_access_attempt"
    assert events[0].extra["granted"] is False


def test_guard_lock_released_after_context_exit(guarded_store, make_dataset):
    dataset = _put_holdout(guarded_store, make_dataset)
    with guarded_store.final_reporting_access(reason="final report") as unlocked:
        unlocked.get(dataset.meta.artifact_id)
    # Outside the context, reads are blocked again.
    with pytest.raises(HoldoutAccessError):
        guarded_store.get(dataset.meta.artifact_id)
