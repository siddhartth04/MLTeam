"""Shared fixtures: temp stores and artifact factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.contracts import (
    DatasetRole,
    DatasetVersion,
    new_artifact,
)
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN_ID = "test-run-001"


@pytest.fixture()
def local_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "store")


@pytest.fixture()
def guarded_store(tmp_path: Path) -> GuardedArtifactStore:
    root = tmp_path / "store"
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


@pytest.fixture()
def make_dataset():
    """Factory for DatasetVersion artifacts."""

    def _make(
        role: DatasetRole = DatasetRole.RAW,
        blob_hash: str = "0" * 64,
        parent_ids: tuple[str, ...] = (),
        run_id: str = RUN_ID,
    ) -> DatasetVersion:
        return new_artifact(
            DatasetVersion,
            producer_agent="test-fixture",
            run_id=run_id,
            parent_ids=parent_ids,
            blob_hash=blob_hash,
            n_rows=100,
            n_cols=5,
            role=role,
            target_column="target",
        )

    return _make
