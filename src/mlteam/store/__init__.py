"""Artifact store: typed artifacts + blobs, lineage, and the holdout guard."""

from pathlib import Path

from mlteam.store.guard import (
    GuardAccessContext,
    GuardAccessRecord,
    GuardedArtifactStore,
    HoldoutAccessError,
)
from mlteam.store.interface import ArtifactNotFoundError, ArtifactStore
from mlteam.store.lineage import LineageIndex, LineageRecord
from mlteam.store.local import LocalArtifactStore


def open_store(root: Path) -> GuardedArtifactStore:
    """Open (or create) the default guarded local store at ``root``."""
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactStore",
    "GuardAccessContext",
    "GuardAccessRecord",
    "GuardedArtifactStore",
    "HoldoutAccessError",
    "LineageIndex",
    "LineageRecord",
    "LocalArtifactStore",
    "open_store",
]
