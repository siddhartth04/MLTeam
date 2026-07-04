"""Split strategist: rule-based strategy choice + the frozen holdout.

The holdout DatasetVersion is written with role=HOLDOUT, which the guarded
store protects immediately — nothing downstream can read it again until
final reporting. Strategies: stratified (classification), random (regression),
temporal (forecasting — the holdout is the chronological TAIL, never shuffled,
so evaluation simulates predicting the future).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.contracts import (
    DatasetRole,
    SplitPlan,
    SplitStrategy,
    TaskType,
    content_hash,
    is_classification,
    new_artifact,
)
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)


def _choose_strategy(task_type: TaskType) -> tuple[SplitStrategy, str]:
    if task_type is TaskType.FORECASTING:
        return (
            SplitStrategy.TEMPORAL,
            "forecasting target: hold out the chronological tail so evaluation "
            "simulates predicting the future — shuffled splits would leak it",
        )
    if is_classification(task_type):
        return (
            SplitStrategy.STRATIFIED,
            "classification target: stratify to preserve class balance in every split "
            "(critical for rare-anomaly labels)",
        )
    return SplitStrategy.RANDOM, "regression target with no temporal/group structure declared"


def detect_time_column(df: pl.DataFrame) -> str:
    """The single Date/Datetime column, or a loud error if none/ambiguous."""
    temporal = [c for c, dt in df.schema.items() if dt in (pl.Date, pl.Datetime)]
    if len(temporal) == 1:
        return temporal[0]
    if not temporal:
        raise ValueError(
            "forecasting requires a Date/Datetime column but the dataset has none; "
            "declare time_column in the problem spec or add a parseable date column"
        )
    raise ValueError(
        f"forecasting time column is ambiguous — dataset has {temporal}; "
        "declare time_column in the problem spec"
    )


def _split_indices(
    y: np.ndarray, strategy: SplitStrategy, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    stratify = y if strategy is SplitStrategy.STRATIFIED else None
    train_idx, holdout_idx = train_test_split(
        indices, test_size=holdout_fraction, random_state=seed, stratify=stratify
    )
    # Sorted indices keep row order stable, so dataset content hashes are
    # independent of shuffle internals.
    return np.sort(train_idx), np.sort(holdout_idx)


def _fold_assignment_hash(
    y_train: np.ndarray, strategy: SplitStrategy, n_folds: int, seed: int
) -> str:
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if strategy is SplitStrategy.TEMPORAL:
        cv: object = TimeSeriesSplit(n_splits=n_folds)
    elif strategy is SplitStrategy.STRATIFIED:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # -1 marks rows in no test fold (the expanding-window prefix of a
    # TimeSeriesSplit is train-only).
    assignment = np.full(len(y_train), -1, dtype=np.int64)
    for fold, (_, test_idx) in enumerate(cv.split(np.zeros(len(y_train)), y_train)):
        assignment[test_idx] = fold
    return content_hash(assignment.tolist())


def plan_split(
    store: ArtifactStore,
    dataset_id: str,
    *,
    task_type: TaskType,
    run_id: str,
    seed: int,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    time_column: str | None = None,
) -> SplitPlan:
    """Split a raw dataset into train + frozen holdout and store the plan.

    Temporal (forecasting) splits sort by ``time_column`` and freeze the
    chronological tail as holdout; the stored train set stays time-sorted so
    the trainer's TimeSeriesSplit CV is valid on row order.
    """
    dataset = store.get(dataset_id)
    if dataset.target_column is None:
        raise ValueError("cannot split a dataset without a declared target column")
    df = read_dataset(store, dataset)

    strategy, rationale = _choose_strategy(task_type)
    if strategy is SplitStrategy.TEMPORAL:
        time_column = time_column or detect_time_column(df)
        if time_column not in df.columns:
            raise ValueError(f"declared time_column '{time_column}' is not in the dataset")
        df = df.sort(time_column)
        n_holdout = max(1, round(len(df) * holdout_fraction))
        # Cut on a TIMESTAMP boundary, not a row count: rows sharing the
        # boundary timestamp must all land in holdout, or "train < holdout"
        # chronology silently breaks on duplicate timestamps.
        boundary = df[time_column][len(df) - n_holdout]
        n_train = int((df[time_column] < boundary).sum())
        if n_train == 0:
            raise ValueError(
                "temporal split impossible: every row shares the boundary "
                f"timestamp in '{time_column}'"
            )
        if n_train < n_folds + 1:
            raise ValueError(
                f"forecasting series has only {n_train} training rows after the "
                f"chronological holdout — reduce n_folds to at most {n_train - 1} "
                "or provide a longer series"
            )
        train_df = df[:n_train]
        holdout_df = df[n_train:]
        y_train = train_df[dataset.target_column].cast(pl.String).to_numpy()
    else:
        time_column = None
        y = df[dataset.target_column].cast(pl.String).to_numpy()
        train_idx, holdout_idx = _split_indices(y, strategy, holdout_fraction, seed)
        train_df = df[train_idx.tolist()]
        holdout_df = df[holdout_idx.tolist()]
        y_train = y[train_idx]

    train_ds = store_dataframe(
        store,
        train_df,
        role=DatasetRole.TRAIN,
        producer_agent="split-strategist",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=dataset.target_column,
    )
    holdout_ds = store_dataframe(
        store,
        holdout_df,
        role=DatasetRole.HOLDOUT,
        producer_agent="split-strategist",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=dataset.target_column,
    )
    _log.info(
        "holdout_frozen",
        strategy=strategy.value,
        holdout_rows=len(holdout_df),
        train_rows=len(train_df),
        holdout_artifact=holdout_ds.meta.artifact_id,
    )

    plan = new_artifact(
        SplitPlan,
        producer_agent="split-strategist",
        run_id=run_id,
        parent_ids=(dataset_id, train_ds.meta.artifact_id, holdout_ds.meta.artifact_id),
        strategy=strategy,
        rationale=rationale,
        n_folds=n_folds,
        seed=seed,
        holdout_fraction=holdout_fraction,
        time_column=time_column,
        train_ref=train_ds.ref(),
        holdout_ref=holdout_ds.ref(),
        fold_assignment_hash=_fold_assignment_hash(y_train, strategy, n_folds, seed),
    )
    store.put(plan)
    return plan
