"""Lineage index tests: provenance rows, parent/child edges, ancestry."""

from __future__ import annotations

from mlteam.contracts import DatasetRole
from mlteam.store import LocalArtifactStore


def test_record_written_on_put(local_store: LocalArtifactStore, make_dataset):
    dataset = make_dataset()
    local_store.put(dataset)
    record = local_store.lineage.get_record(dataset.meta.artifact_id)
    assert record is not None
    assert record.artifact_type == "DatasetVersion"
    assert record.producer_agent == "test-fixture"
    assert record.run_id == dataset.meta.run_id


def test_parent_child_edges(local_store: LocalArtifactStore, make_dataset):
    raw = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64)
    cleaned = make_dataset(
        role=DatasetRole.CLEANED,
        blob_hash="b" * 64,
        parent_ids=(raw.meta.artifact_id,),
    )
    local_store.put(raw)
    local_store.put(cleaned)

    assert local_store.lineage.parents(cleaned.meta.artifact_id) == (raw.meta.artifact_id,)
    assert local_store.lineage.children(raw.meta.artifact_id) == (cleaned.meta.artifact_id,)


def test_ancestry_is_transitive(local_store: LocalArtifactStore, make_dataset):
    raw = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64)
    cleaned = make_dataset(
        role=DatasetRole.CLEANED, blob_hash="b" * 64, parent_ids=(raw.meta.artifact_id,)
    )
    features = make_dataset(
        role=DatasetRole.FEATURES, blob_hash="c" * 64, parent_ids=(cleaned.meta.artifact_id,)
    )
    for artifact in (raw, cleaned, features):
        local_store.put(artifact)

    ancestors = local_store.lineage.ancestry(features.meta.artifact_id)
    assert set(ancestors) == {raw.meta.artifact_id, cleaned.meta.artifact_id}


def test_dangling_parent_edge_is_rejected(local_store: LocalArtifactStore, make_dataset):
    import pytest

    orphan = make_dataset(blob_hash="d" * 64, parent_ids=("never-recorded-parent",))
    with pytest.raises(ValueError, match="parents"):
        local_store.put(orphan)


def test_lineage_survives_reopen(tmp_path, make_dataset):
    root = tmp_path / "store"
    dataset = make_dataset()
    LocalArtifactStore(root).put(dataset)

    reopened = LocalArtifactStore(root)
    assert reopened.lineage.get_record(dataset.meta.artifact_id) is not None


def test_artifact_ids_for_run_returns_only_that_runs_membership(
    local_store: LocalArtifactStore, make_dataset
):
    a = make_dataset(blob_hash="a" * 64, run_id="run-1")
    b = make_dataset(blob_hash="b" * 64, run_id="run-2")
    local_store.put(a)
    local_store.put(b)

    assert local_store.lineage.artifact_ids_for_run("run-1") == (a.meta.artifact_id,)
    assert local_store.lineage.artifact_ids_for_run("run-2") == (b.meta.artifact_id,)
    assert local_store.lineage.artifact_ids_for_run("run-missing") == ()


def test_artifact_ids_for_run_includes_a_shared_deduped_artifact(
    local_store: LocalArtifactStore, make_dataset
):
    """Phase 4 (dashboard UX plan) regression: content-addressing means two
    runs over identical inputs mint the SAME artifact id. artifacts.run_id
    (first-writer-wins) would attribute it only to run-1; run_artifacts must
    show it under BOTH runs, since both runs genuinely used it."""
    shared = make_dataset(blob_hash="shared" + "0" * 58, run_id="run-1")
    local_store.put(shared)
    # A second run reuses the identical artifact (same content-addressed id).
    local_store.lineage.record(shared.meta.model_copy(update={"run_id": "run-2"}))

    run1_ids = local_store.lineage.artifact_ids_for_run("run-1")
    run2_ids = local_store.lineage.artifact_ids_for_run("run-2")
    assert shared.meta.artifact_id in run1_ids
    assert shared.meta.artifact_id in run2_ids
