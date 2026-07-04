"""Ingestion: user file → validated, content-hashed raw DatasetVersion."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from mlteam.agents.data_io import store_dataframe
from mlteam.contracts import DatasetRole, DatasetVersion
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

def _read_csv(path: Path) -> pl.DataFrame:
    # try_parse_dates so date columns get real Date/Datetime dtypes — the
    # profiler's DATETIME_DETECTED warning and the forecasting path (time
    # column detection, temporal split) all key off the dtype.
    #
    # Real-world CSVs (Kaggle especially) routinely mix numeric values with
    # bucket labels in one column (e.g. a "Balcony" count column containing
    # "> 10"). Polars' strict parser raises ComputeError on the first such
    # value past its inference window, killing ingestion outright. Retry with
    # ignore_errors=True (nulls out unparseable cells in an otherwise-typed
    # column instead of crashing) rather than forcing every column to string
    # — that would silently defeat numeric profiling/imputation everywhere.
    try:
        return pl.read_csv(path, try_parse_dates=True)
    except pl.exceptions.ComputeError as exc:
        _log.warning("csv_strict_parse_failed_retrying_lenient", path=str(path), error=str(exc))
        return pl.read_csv(path, try_parse_dates=True, ignore_errors=True)


_READERS = {
    ".csv": _read_csv,
    ".parquet": pl.read_parquet,
}


def _normalize_nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Canonicalize float NaN to Polars null so missingness has ONE representation."""
    float_cols = [c for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    return df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])


def ingest_file(
    store: ArtifactStore,
    path: Path,
    *,
    target_column: str,
    run_id: str,
    allow_missing_target: bool = False,
) -> DatasetVersion:
    """Read a CSV/parquet file, validate basics, store as raw DatasetVersion.

    The source file is never mutated; its contents become a content-addressed
    parquet blob in the store. ``allow_missing_target=True`` is for the
    pipeline path where the data-triage stage runs next and drops unlabeled
    rows with a logged, reported justification; direct callers keep the
    strict default.
    """
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(f"unsupported dataset format '{path.suffix}'; expected one of {supported}")

    df = reader(path)
    df = _normalize_nan_to_null(df)
    if df.height == 0:
        raise ValueError(f"dataset is empty: {path}")
    if target_column not in df.columns:
        raise ValueError(f"target column '{target_column}' not found; available: {df.columns}")
    if not allow_missing_target and df[target_column].null_count() > 0:
        raise ValueError(
            f"target column '{target_column}' has {df[target_column].null_count()} "
            "missing values; rows without labels must be resolved before ingestion"
        )

    dataset = store_dataframe(
        store,
        df,
        role=DatasetRole.RAW,
        producer_agent="ingest",
        run_id=run_id,
        target_column=target_column,
    )
    _log.info(
        "dataset_ingested",
        path=str(path),
        artifact_id=dataset.meta.artifact_id,
        n_rows=dataset.n_rows,
        n_cols=dataset.n_cols,
    )
    return dataset
