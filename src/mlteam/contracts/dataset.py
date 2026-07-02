"""DatasetVersion: an immutable, content-hashed reference to tabular data."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from mlteam.contracts.base import Artifact


class DatasetRole(StrEnum):
    RAW = "raw"
    CLEANED = "cleaned"
    FEATURES = "features"
    TRAIN = "train"
    HOLDOUT = "holdout"


class DatasetVersion(Artifact):
    """Pointer to a dataset blob in the store plus its identity and role.

    ``role == HOLDOUT`` triggers store-level access protection: the guard
    blocks all reads until final reporting explicitly unlocks it.
    """

    blob_hash: str = Field(min_length=1, description="Content hash of the data blob in the store")
    file_format: str = "parquet"
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    role: DatasetRole
    target_column: str | None = None
    pandera_schema_blob: str | None = Field(
        default=None, description="Blob hash of the serialized Pandera schema, if generated"
    )

    def guard_protected_items(self) -> tuple[tuple[str, str], ...]:
        if self.role is DatasetRole.HOLDOUT:
            return (
                (self.meta.artifact_id, "holdout dataset artifact"),
                (self.blob_hash, "holdout dataset blob"),
            )
        return ()
