"""Optuna HPO tests: vetted spaces, determinism, stored best model."""

from __future__ import annotations

import pytest

from mlteam.agents import ingest_file, tune_experiment
from mlteam.contracts import TaskType

RUN = "test-hpo-run"


def tiny_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
    }


@pytest.fixture()
def dataset_id(guarded_store, churn_csv):
    return ingest_file(
        guarded_store, churn_csv, target_column="churned", run_id=RUN
    ).meta.artifact_id


def _tune(store, dataset_id, **kwargs):
    return tune_experiment(
        store,
        dataset_id,
        template_id="lightgbm_default",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
        n_trials=4,
        space=tiny_space,
        **kwargs,
    )


def test_hpo_returns_stored_best_model(guarded_store, dataset_id):
    outcome = _tune(guarded_store, dataset_id)
    assert outcome.n_trials_run == 4
    assert 10 <= outcome.best_params["n_estimators"] <= 20
    assert 0.05 <= outcome.best_params["learning_rate"] <= 0.2
    assert guarded_store.exists(outcome.result.model.meta.artifact_id)
    assert outcome.result.model.config == outcome.best_params
    assert "roc_auc" in outcome.result.metrics


def test_hpo_is_deterministic(guarded_store, dataset_id):
    first = _tune(guarded_store, dataset_id, experiment_id="hpo-a")
    second = _tune(guarded_store, dataset_id, experiment_id="hpo-b")
    assert first.best_params == second.best_params
    assert first.best_cv_score == pytest.approx(second.best_cv_score)


def test_hpo_optimizes_rmse_toward_lower_error(guarded_store):
    """Regression: lower-is-better metrics must not be inverted. The tuned
    config's RMSE must beat a deliberately-underfit config's RMSE."""
    import numpy as np
    import polars as pl

    from mlteam.agents import train_experiment
    from mlteam.agents.data_io import store_dataframe
    from mlteam.contracts import BudgetEstimate, DatasetRole, ExperimentSpec

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 300)
    df = pl.DataFrame(
        {"x": x, "x2": x**2, "noise": rng.normal(0, 1, 300), "y": 3.0 * x + rng.normal(0, 1, 300)}
    )
    dataset = store_dataframe(
        guarded_store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id="rmse-run",
        target_column="y",
    )

    def wide_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        }

    outcome = tune_experiment(
        guarded_store,
        dataset.meta.artifact_id,
        template_id="lightgbm_default",
        task_type=TaskType.REGRESSION,
        primary_metric="rmse",
        run_id="rmse-run",
        seed=1,
        n_folds=3,
        env_hash="test-env",
        n_trials=12,
        space=wide_space,
    )
    underfit = train_experiment(
        guarded_store,
        dataset.meta.artifact_id,
        ExperimentSpec(
            experiment_id="underfit",
            template_id="lightgbm_default",
            config={"n_estimators": 2, "num_leaves": 2, "learning_rate": 0.02},
            budget=BudgetEstimate(wall_time_seconds=1),
        ),
        task_type=TaskType.REGRESSION,
        run_id="rmse-run",
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    tuned_rmse = outcome.result.metrics["rmse"].mean
    assert tuned_rmse < underfit.metrics["rmse"].mean  # searched toward LOW error
    assert outcome.best_cv_score > 0  # reported in natural (positive rmse) sign


def test_hpo_rejects_untunable_template(guarded_store, dataset_id):
    with pytest.raises(ValueError, match="no vetted search space"):
        tune_experiment(
            guarded_store,
            dataset_id,
            template_id="dummy_baseline",
            task_type=TaskType.BINARY_CLASSIFICATION,
            primary_metric="roc_auc",
            run_id=RUN,
            seed=42,
            n_folds=3,
            env_hash="test-env",
        )
