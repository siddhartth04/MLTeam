"""Phase 1: every experiment reaches a persisted terminal state.

Before ExperimentOutcome existed, only the winning model's evaluation was
ever stored; every other experiment's metrics lived in an in-memory dict
that vanished when the process moved on, and a training failure was only
a log line. These tests prove the fix: completed, failed, and quorum-
collapsed runs are all distinguishable from the store afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.api.views import experiments_view
from mlteam.contracts import (
    BudgetEstimate,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentState,
    TaskType,
)
from mlteam.orchestration.agentic import _check_experiment_quorum
from mlteam.orchestration.pipeline import _dummy_key, _train_all, default_experiment_specs
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-outcomes-run"
FAST_LGBM = {"n_estimators": 30}


def _fresh_store(root: Path) -> GuardedArtifactStore:
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


@pytest.fixture()
def trained_store(tmp_path: Path):
    """A tiny end-to-end pipeline run, giving a store with real outcomes."""
    from mlteam.orchestration import run_tabular_pipeline
    from synthetic import make_churn_frame

    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    store = _fresh_store(tmp_path / "store")
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config=FAST_LGBM,
    )
    return store, result


def test_experiments_view_is_baseline_matches_the_plan_not_just_dummy_template(trained_store):
    """audit L4: default_experiment_specs marks BOTH exp-dummy (dummy_baseline)
    and exp-lightgbm (lightgbm_default) as is_baseline=True in the plan. The
    view must agree — deriving is_baseline purely from
    template_id == "dummy_baseline" would incorrectly report exp-lightgbm as
    a non-baseline, contradicting the plan it came from."""
    store, result = trained_store
    rows = experiments_view(store, result.run_id)
    by_id = {r["experiment_id"]: r for r in rows}
    assert by_id["exp-dummy"]["is_baseline"] is True
    assert by_id["exp-lightgbm"]["is_baseline"] is True


def test_completed_outcome_persisted_with_metrics(trained_store):
    store, result = trained_store
    outcome_ids = store.list_ids(ExperimentOutcome.artifact_type())
    outcomes = [store.get(aid) for aid in outcome_ids]
    by_experiment = {o.experiment_id: o for o in outcomes}

    assert "exp-dummy" in by_experiment
    assert "exp-lightgbm" in by_experiment
    lgbm = by_experiment["exp-lightgbm"]
    assert lgbm.state == ExperimentState.COMPLETED
    assert lgbm.model_ref is not None
    report_metric_names = {m.name for m in result.evaluation_report.cv_metrics}
    assert {m.name for m in lgbm.cv_metrics} == report_metric_names
    assert lgbm.wall_time_seconds > 0


def test_failed_experiment_persists_error_and_does_not_kill_the_run(tmp_path: Path):
    from mlteam.agents.ingest import ingest_file
    from synthetic import make_churn_frame

    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    store = _fresh_store(tmp_path / "store")
    dataset = ingest_file(store, csv_path, target_column="churned", run_id=RUN)

    specs = (
        ExperimentSpec(
            experiment_id="exp-dummy",
            template_id="dummy_baseline",
            budget=BudgetEstimate(wall_time_seconds=2),
            is_baseline=True,
        ),
        ExperimentSpec(
            experiment_id="exp-broken",
            template_id="does_not_exist_in_registry",
            budget=BudgetEstimate(wall_time_seconds=2),
            is_baseline=False,
        ),
    )
    results, skipped = _train_all(
        store,
        dataset.meta.artifact_id,
        specs,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
        mlflow_enabled=False,
    )
    assert "exp-dummy" in results
    assert "exp-broken" not in results
    assert len(skipped) == 1
    assert skipped[0][0] == "does_not_exist_in_registry"

    outcomes = {
        store.get(aid).experiment_id: store.get(aid)
        for aid in store.list_ids(ExperimentOutcome.artifact_type())
    }
    assert outcomes["exp-dummy"].state == ExperimentState.COMPLETED
    assert outcomes["exp-broken"].state == ExperimentState.FAILED
    assert outcomes["exp-broken"].error is not None
    assert "does_not_exist_in_registry" in outcomes["exp-broken"].error
    assert outcomes["exp-broken"].model_ref is None


def test_quorum_raises_when_every_candidate_fails():
    specs = (
        ExperimentSpec(
            experiment_id="exp-dummy",
            template_id="dummy_baseline",
            budget=BudgetEstimate(wall_time_seconds=2),
            is_baseline=True,
        ),
        ExperimentSpec(
            experiment_id="exp-a",
            template_id="lightgbm_default",
            budget=BudgetEstimate(wall_time_seconds=2),
        ),
        ExperimentSpec(
            experiment_id="exp-b",
            template_id="random_forest_default",
            budget=BudgetEstimate(wall_time_seconds=2),
        ),
    )
    dummy_key = _dummy_key(specs)
    results: dict = {}  # nothing trained — total collapse
    skipped = (
        ("lightgbm_default", "boom-a"),
        ("random_forest_default", "boom-b"),
    )
    with pytest.raises(ValueError, match="all 2 candidate experiment"):
        _check_experiment_quorum(specs, results, skipped, dummy_key)


def test_quorum_allows_partial_completion():
    """_check_experiment_quorum only checks membership in `results`, so a
    plain sentinel is enough to represent "this experiment_id completed"."""
    specs = default_experiment_specs(include_tabpfn=False)[:2]  # dummy + lightgbm
    dummy_key = _dummy_key(specs)
    non_dummy_id = next(s.experiment_id for s in specs if s.experiment_id != dummy_key)
    results = {dummy_key: object(), non_dummy_id: object()}
    _check_experiment_quorum(specs, results, (), dummy_key)  # no raise


def test_experiments_view_returns_state_and_error(trained_store):
    store, result = trained_store
    rows = experiments_view(store, result.run_id)
    assert len(rows) >= 2
    assert all("state" in r for r in rows)
    completed = [r for r in rows if r["state"] == "completed"]
    assert len(completed) >= 2
    # trained_store's fixture plan (default_experiment_specs, no TabPFN) is
    # exactly the two mandatory baselines (dummy + lightgbm) with no
    # non-baseline candidate — both are correctly is_baseline=True per the
    # plan (audit L4: this must come from the plan, not a
    # template == "dummy_baseline" heuristic that would call lightgbm a
    # non-baseline and fail this very assertion).
    assert all(r["is_baseline"] for r in completed)
    assert all(r["error"] is None for r in completed)
