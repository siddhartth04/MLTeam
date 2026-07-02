"""Store tests: artifact round-trip, idempotency, blobs, missing items."""

from __future__ import annotations

import pytest

from mlteam.contracts import DatasetRole
from mlteam.store import ArtifactNotFoundError, LocalArtifactStore


def test_put_get_round_trip(local_store: LocalArtifactStore, make_dataset):
    dataset = make_dataset()
    ref = local_store.put(dataset)
    assert ref.artifact_id == dataset.meta.artifact_id
    loaded = local_store.get(ref.artifact_id)
    assert loaded == dataset


def test_put_is_idempotent(local_store: LocalArtifactStore, make_dataset):
    dataset = make_dataset()
    ref_a = local_store.put(dataset)
    ref_b = local_store.put(dataset)
    assert ref_a == ref_b
    assert local_store.list_ids("DatasetVersion") == (dataset.meta.artifact_id,)


def test_get_missing_raises(local_store: LocalArtifactStore):
    with pytest.raises(ArtifactNotFoundError):
        local_store.get("does-not-exist")


def test_exists(local_store: LocalArtifactStore, make_dataset):
    dataset = make_dataset()
    assert not local_store.exists(dataset.meta.artifact_id)
    local_store.put(dataset)
    assert local_store.exists(dataset.meta.artifact_id)


def test_blob_round_trip(local_store: LocalArtifactStore):
    payload = b"col_a,col_b\n1,2\n3,4\n"
    blob_hash = local_store.put_blob(payload)
    assert local_store.blob_exists(blob_hash)
    assert local_store.blob_path(blob_hash).read_bytes() == payload


def test_blob_content_addressed(local_store: LocalArtifactStore):
    assert local_store.put_blob(b"same") == local_store.put_blob(b"same")
    assert local_store.put_blob(b"same") != local_store.put_blob(b"different")


def test_blob_file_matches_bytes_hash(local_store: LocalArtifactStore, tmp_path):
    payload = b"x" * 4096
    source = tmp_path / "data.bin"
    source.write_bytes(payload)
    assert local_store.put_blob_file(source) == local_store.put_blob(payload)


def test_blob_missing_raises(local_store: LocalArtifactStore):
    with pytest.raises(ArtifactNotFoundError):
        local_store.blob_path("0" * 64)


def test_list_ids_filters_by_type(local_store: LocalArtifactStore, make_dataset):
    dataset = make_dataset(role=DatasetRole.CLEANED)
    local_store.put(dataset)
    assert dataset.meta.artifact_id in local_store.list_ids("DatasetVersion")
    assert local_store.list_ids("ProblemSpec") == ()
