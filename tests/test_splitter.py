"""Splitter tests: strategy rules, frozen holdout, determinism."""

from __future__ import annotations

import pytest

from mlteam.agents import ingest_file, plan_split
from mlteam.contracts import SplitStrategy, TaskType
from mlteam.store import HoldoutAccessError

RUN = "test-split-run"


@pytest.fixture()
def raw_dataset(guarded_store, churn_csv):
    return ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)


def test_classification_uses_stratified(guarded_store, raw_dataset):
    plan = plan_split(
        guarded_store,
        raw_dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
    )
    assert plan.strategy is SplitStrategy.STRATIFIED
    assert plan.rationale


def test_split_fractions(guarded_store, raw_dataset):
    plan = plan_split(
        guarded_store,
        raw_dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        holdout_fraction=0.25,
    )
    train = guarded_store.get(plan.train_ref.artifact_id)
    total = raw_dataset.n_rows
    assert train.n_rows == pytest.approx(0.75 * total, abs=2)


def test_holdout_is_frozen_immediately(guarded_store, raw_dataset):
    plan = plan_split(
        guarded_store,
        raw_dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
    )
    with pytest.raises(HoldoutAccessError):
        guarded_store.get(plan.holdout_ref.artifact_id)


def test_split_is_deterministic(guarded_store, raw_dataset):
    kwargs = dict(
        task_type=TaskType.BINARY_CLASSIFICATION,
        seed=42,
        holdout_fraction=0.2,
        n_folds=5,
    )
    a = plan_split(guarded_store, raw_dataset.meta.artifact_id, run_id="run-a", **kwargs)
    b = plan_split(guarded_store, raw_dataset.meta.artifact_id, run_id="run-b", **kwargs)
    assert a.meta.artifact_id == b.meta.artifact_id
    assert a.fold_assignment_hash == b.fold_assignment_hash


def test_different_seed_changes_split(guarded_store, raw_dataset):
    a = plan_split(
        guarded_store,
        raw_dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=1,
    )
    b = plan_split(
        guarded_store,
        raw_dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=2,
    )
    assert a.train_ref.artifact_id != b.train_ref.artifact_id
