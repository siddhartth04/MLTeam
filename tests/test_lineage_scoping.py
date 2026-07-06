"""Phase 4 (dashboard UX plan): lineage/artifacts scoped to the run that
actually used them, not just the run that happened to produce them first.

Root cause: content-addressing means two runs over identical inputs mint the
SAME artifact id (new_artifact excludes run_id from the hash, deliberately,
for dedup/reproducibility). The old lineage_graph/list_artifacts filter read
`artifacts.run_id`, a first-writer-wins column (INSERT OR IGNORE) -- so a
second run reusing an earlier run's artifact would see an incomplete graph,
or (worse) appear to see the OTHER run's artifacts. The run_artifacts
membership table fixes this without touching content-addressing/dedup.
"""

from __future__ import annotations

from mlteam.api.views import lineage_graph, list_artifacts
from mlteam.contracts import DatasetRole
from mlteam.store import GuardedArtifactStore, LocalArtifactStore


def test_two_runs_over_identical_input_each_see_their_full_lineage(tmp_path, make_dataset):
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    # run-1 produces a raw dataset, then a cleaned child.
    raw = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64, run_id="run-1")
    store.put(raw)
    cleaned_1 = make_dataset(
        role=DatasetRole.CLEANED,
        blob_hash="b" * 64,
        parent_ids=(raw.meta.artifact_id,),
        run_id="run-1",
    )
    store.put(cleaned_1)

    # run-2 re-runs over the SAME tiny dataset+seed: content-addressing mints
    # the identical raw artifact id (dedup), but produces its OWN cleaned
    # child (different config/content -> different id).
    raw_reused = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64, run_id="run-2")
    assert raw_reused.meta.artifact_id == raw.meta.artifact_id  # confirms the dedup premise
    store.put(raw_reused)
    cleaned_2 = make_dataset(
        role=DatasetRole.CLEANED,
        blob_hash="c" * 64,
        parent_ids=(raw.meta.artifact_id,),
        run_id="run-2",
    )
    store.put(cleaned_2)

    graph_1 = lineage_graph(store, run_id="run-1")
    graph_2 = lineage_graph(store, run_id="run-2")

    ids_1 = {n["id"] for n in graph_1["nodes"]}
    ids_2 = {n["id"] for n in graph_2["nodes"]}

    # Each run sees the shared/deduped raw artifact AND its own cleaned child.
    assert raw.meta.artifact_id in ids_1
    assert raw.meta.artifact_id in ids_2
    assert cleaned_1.meta.artifact_id in ids_1
    assert cleaned_1.meta.artifact_id not in ids_2
    assert cleaned_2.meta.artifact_id in ids_2
    assert cleaned_2.meta.artifact_id not in ids_1

    # run-1's graph must not be empty just because run-2 also touched the
    # shared artifact -- this was the exact failure mode being fixed.
    assert len(graph_1["nodes"]) >= 2
    assert len(graph_2["nodes"]) >= 2


def test_two_runs_over_identical_input_each_see_their_full_artifact_list(tmp_path, make_dataset):
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    raw = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64, run_id="run-1")
    store.put(raw)
    raw_reused = make_dataset(role=DatasetRole.RAW, blob_hash="a" * 64, run_id="run-2")
    store.put(raw_reused)

    ids_1 = {a["id"] for a in list_artifacts(store, run_id="run-1")}
    ids_2 = {a["id"] for a in list_artifacts(store, run_id="run-2")}
    assert raw.meta.artifact_id in ids_1
    assert raw.meta.artifact_id in ids_2


def test_legacy_store_with_no_membership_rows_falls_back(tmp_path, make_dataset):
    """A pre-existing store (no run_artifacts rows for old data) must still
    render via the legacy artifacts.run_id filter -- no backfill required."""
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    dataset = make_dataset(run_id="legacy-run")
    store.put(dataset)

    # Simulate a pre-membership-table row: delete it directly, leaving only
    # the legacy `artifacts` row behind.
    with store._inner.lineage._connect() as conn:
        conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", ("legacy-run",))

    graph = lineage_graph(store, run_id="legacy-run")
    assert dataset.meta.artifact_id in {n["id"] for n in graph["nodes"]}
