"""Split strategist: rule-based strategy choice + the frozen holdout.

The holdout DatasetVersion is written with role=HOLDOUT, which the guarded
store protects immediately — nothing downstream can read it again until
final reporting. Phase 1 implements the random/stratified rules; temporal
and group strategies join in Phase 3 with the rule engine + LLM tie-break.
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
    new_artifact,
)
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)


def _choose_strategy(task_type: TaskType) -> tuple[SplitStrategy, str]:
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        return (
            SplitStrategy.STRATIFIED,
            "classification target: stratify to preserve class balance in every split",
        )
    return SplitStrategy.RANDOM, "regression target with no temporal/group structure declared"


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
    from sklearn.model_selection import KFold, StratifiedKFold

    if strategy is SplitStrategy.STRATIFIED:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignment = np.empty(len(y_train), dtype=np.int64)
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
) -> SplitPlan:
    """Split a raw dataset into train + frozen holdout and store the plan."""
    dataset = store.get(dataset_id)
    if dataset.target_column is None:
        raise ValueError("cannot split a dataset without a declared target column")
    df = read_dataset(store, dataset)
    y = df[dataset.target_column].cast(pl.String).to_numpy()

    strategy, rationale = _choose_strategy(task_type)
    train_idx, holdout_idx = _split_indices(y, strategy, holdout_fraction, seed)

    train_ds = store_dataframe(
        store,
        df[train_idx.tolist()],
        role=DatasetRole.TRAIN,
        producer_agent="split-strategist",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=dataset.target_column,
    )
    holdout_ds = store_dataframe(
        store,
        df[holdout_idx.tolist()],
        role=DatasetRole.HOLDOUT,
        producer_agent="split-strategist",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=dataset.target_column,
    )
    _log.info(
        "holdout_frozen",
        holdout_rows=len(holdout_idx),
        train_rows=len(train_idx),
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
        train_ref=train_ds.ref(),
        holdout_ref=holdout_ds.ref(),
        fold_assignment_hash=_fold_assignment_hash(y[train_idx], strategy, n_folds, seed),
    )
    store.put(plan)
    return plan
