"""Dataset I/O between Polars frames and the artifact store.

All datasets are stored as canonical parquet blobs; a DatasetVersion artifact
points at the blob. Reading a holdout dataset through the guarded store fails
by design — the guard protects the blob as well as the artifact.
"""

from __future__ import annotations

import io

import polars as pl

from mlteam.contracts import DatasetRole, DatasetVersion, new_artifact
from mlteam.store.interface import ArtifactStore


def read_dataset(store: ArtifactStore, dataset: DatasetVersion) -> pl.DataFrame:
    """Load a dataset's parquet blob into a Polars frame."""
    return pl.read_parquet(store.blob_path(dataset.blob_hash))


def store_dataframe(
    store: ArtifactStore,
    df: pl.DataFrame,
    *,
    role: DatasetRole,
    producer_agent: str,
    run_id: str,
    parent_ids: tuple[str, ...] = (),
    target_column: str | None = None,
) -> DatasetVersion:
    """Write a frame as a parquet blob and register its DatasetVersion."""
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    blob_hash = store.put_blob(buffer.getvalue())
    dataset = new_artifact(
        DatasetVersion,
        producer_agent=producer_agent,
        run_id=run_id,
        parent_ids=parent_ids,
        blob_hash=blob_hash,
        file_format="parquet",
        n_rows=df.height,
        n_cols=df.width,
        role=role,
        target_column=target_column,
    )
    store.put(dataset)
    return dataset
