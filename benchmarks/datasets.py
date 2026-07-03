"""Benchmark dataset acquisition: OpenML fetch + deterministic synthesis.

Datasets are cached as parquet under benchmarks/data/ (gitignored); the
manifest pins OpenML ids so cached content is stable and content hashes
reproduce across machines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import structlog

from benchmarks.manifest import BenchmarkDataset

_log = structlog.get_logger(__name__)

DATA_DIR = Path(__file__).parent / "data"
_SYNTH_SEED = 29
_SYNTH_ROWS = 800


def make_temporal_churn_frame(n_rows: int = _SYNTH_ROWS, seed: int = _SYNTH_SEED) -> pl.DataFrame:
    """Deterministic churn dataset with a REAL datetime column."""
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1)
    signup_offsets = rng.integers(0, 730, n_rows)
    signup_dates = [start + timedelta(days=int(d)) for d in signup_offsets]
    weekly_logins = rng.poisson(3.0, n_rows).astype(float)
    monthly_spend = np.round(rng.gamma(2.0, 25.0, n_rows), 2)
    plan = rng.choice(["basic", "plus", "enterprise"], n_rows, p=[0.6, 0.3, 0.1])

    # Later signups churn more (product got worse), low engagement churns more.
    recency = signup_offsets / 730.0
    logit = -1.5 + 1.2 * recency - 0.35 * weekly_logins + np.where(plan == "basic", 0.8, 0.0)
    churned = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    return pl.DataFrame(
        {
            "signup_date": signup_dates,
            "weekly_logins": weekly_logins,
            "monthly_spend": monthly_spend,
            "plan": plan,
            "churned": churned,
        }
    )


def _fetch_openml_frame(openml_id: int) -> pl.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    return pl.from_pandas(bunch.frame)


def ensure_dataset(entry: BenchmarkDataset, data_dir: Path | None = None) -> Path:
    """Return a cached parquet path for a benchmark dataset, fetching if needed."""
    data_dir = data_dir if data_dir is not None else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{entry.name}.parquet"
    if path.exists():
        return path

    if entry.source == "synthetic":
        frame = make_temporal_churn_frame()
    elif entry.source == "openml":
        if entry.openml_id is None:
            raise ValueError(f"benchmark '{entry.name}' declares openml source without an id")
        _log.info("fetching_openml_dataset", name=entry.name, openml_id=entry.openml_id)
        frame = _fetch_openml_frame(entry.openml_id)
    else:
        raise ValueError(f"unknown benchmark source '{entry.source}'")

    if entry.target_column not in frame.columns:
        raise ValueError(
            f"benchmark '{entry.name}': target '{entry.target_column}' not in "
            f"columns {frame.columns}"
        )
    frame.write_parquet(path)
    _log.info("benchmark_dataset_cached", name=entry.name, rows=frame.height, path=str(path))
    return path
