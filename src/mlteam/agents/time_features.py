"""Deterministic forecasting features: target lags, rolling stats, calendar.

Applied BEFORE the temporal split, which is leakage-safe by construction:
every engineered value for a row uses only strictly-earlier rows (all lag and
rolling expressions are shifted by >= 1) or the row's own timestamp. Holdout
rows therefore carry real lagged history — exactly the information a
production forecaster has when predicting the future.

The transform is stateless (no statistics learned from data), so computing it
once on the full frame does not leak holdout information into training.
"""

from __future__ import annotations

import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.contracts import DatasetRole
from mlteam.contracts.dataset import DatasetVersion
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

LAGS: tuple[int, ...] = (1, 2, 3, 7)
ROLLING_WINDOWS: tuple[int, ...] = (3, 7)


def add_forecast_features(
    df: pl.DataFrame, *, time_column: str, target_column: str
) -> pl.DataFrame:
    """Sort by time and add lag/rolling/calendar columns; drop warmup rows.

    The leading rows whose largest lag reaches before the series start are
    dropped (they would carry nulls); training on imputed fake history would
    be worse than losing a handful of rows.
    """
    if time_column not in df.columns:
        raise ValueError(f"time_column '{time_column}' is not in the dataset")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' is not in the dataset")
    if not df[target_column].dtype.is_numeric():
        raise ValueError(
            f"forecasting target '{target_column}' must be numeric, "
            f"got {df[target_column].dtype}"
        )

    df = df.sort(time_column)
    time_dtype = df.schema[time_column]
    if time_dtype not in (pl.Date, pl.Datetime):
        raise ValueError(
            f"time_column '{time_column}' must be Date/Datetime, got {time_dtype}"
        )
    n_dupes = len(df) - df[time_column].n_unique()
    if n_dupes:
        # Panel data (several entities per timestamp) would silently corrupt
        # lags: a global shift(1) pulls a NEIGHBORING ENTITY's value, not this
        # entity's past. Fail loud until per-entity grouping is supported.
        raise ValueError(
            f"time_column '{time_column}' has {n_dupes} duplicate timestamps — "
            "panel/multi-entity forecasting is not supported yet; aggregate to "
            "one row per timestamp or filter to a single entity"
        )

    # The raw time column is kept ON PURPOSE: the trainer casts it to epoch
    # seconds, which is an honest linear-trend term (CV is chronological via
    # TimeSeriesSplit, so an epoch-threshold shortcut cannot score well in
    # fold validation the way it could under shuffled CV).
    t = pl.col(time_column)
    calendar = [
        t.dt.year().alias(f"{time_column}_year"),
        t.dt.month().alias(f"{time_column}_month"),
        t.dt.day().alias(f"{time_column}_day"),
        t.dt.weekday().alias(f"{time_column}_dayofweek"),
        t.dt.ordinal_day().alias(f"{time_column}_dayofyear"),
    ]
    if time_dtype == pl.Datetime:
        calendar.append(t.dt.hour().alias(f"{time_column}_hour"))

    y = pl.col(target_column)
    lags = [k for k in LAGS if k < len(df)]
    windows = [w for w in ROLLING_WINDOWS if w + 1 < len(df)]
    lag_cols = [y.shift(k).alias(f"{target_column}_lag_{k}") for k in lags]
    # shift(1) first: the window covers t-w .. t-1, never the row's own target.
    roll_cols = [
        expr
        for w in windows
        for expr in (
            y.shift(1).rolling_mean(w).alias(f"{target_column}_rollmean_{w}"),
            y.shift(1).rolling_std(w).alias(f"{target_column}_rollstd_{w}"),
        )
    ]

    out = df.with_columns(calendar + lag_cols + roll_cols)
    warmup_cols = [f"{target_column}_lag_{k}" for k in lags] + [
        f"{target_column}_rollmean_{w}" for w in windows
    ] + [f"{target_column}_rollstd_{w}" for w in windows]
    if warmup_cols:
        out = out.drop_nulls(subset=warmup_cols)
    if len(out) < 10:
        raise ValueError(
            f"only {len(out)} rows remain after lag warmup — the series is too "
            "short to forecast with the vetted lag set"
        )
    _log.info(
        "forecast_features_added",
        lags=lags,
        rolling_windows=windows,
        rows_before=len(df),
        rows_after=len(out),
    )
    return out


def engineer_time_features(
    store: ArtifactStore,
    dataset_id: str,
    *,
    time_column: str,
    target_column: str,
    run_id: str,
) -> DatasetVersion:
    """Store a FEATURES DatasetVersion with forecast features, lineage-linked."""
    dataset = store.get(dataset_id)
    featured = add_forecast_features(
        read_dataset(store, dataset), time_column=time_column, target_column=target_column
    )
    return store_dataframe(
        store,
        featured,
        role=DatasetRole.FEATURES,
        producer_agent="time-feature-engineer",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=target_column,
    )
