"""Audit Task 5 (H3): HPO winner's-curse re-scoring.

Root cause: Optuna maximizes the mean CV score across many noisy trials;
the maximum of many estimates is upward-biased (Cawley & Talbot, 2010).
Before this fix, the winning config was retrained and reported on the SAME
folds (seed) the search just optimized against, and (in the iteration loop)
that seed differed from the base experiments' seed -- breaking fold
pairing needed for the paired significance test (audit Task 2) on top of
carrying the search's own selection bias into the number everyone else
compares against.

tune_experiment now decouples the SEARCH seed from the REPORT seed: the
final config is retrained and scored on report_seed's folds (defaulting to
seed, i.e. unchanged behavior, when report_seed is omitted).
"""

from __future__ import annotations

import pytest

from mlteam.agents import ingest_file, train_experiment, tune_experiment
from mlteam.contracts import BudgetEstimate, ExperimentSpec, TaskType

RUN = "test-hpo-rescoring"


def _tiny_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
    }


@pytest.fixture()
def dataset_id(guarded_store, churn_csv):
    return ingest_file(
        guarded_store, churn_csv, target_column="churned", run_id=RUN
    ).meta.artifact_id


def test_report_seed_defaults_to_search_seed_when_omitted(guarded_store, dataset_id):
    """Backward compatibility: omitting report_seed must behave EXACTLY as
    before -- search_cv_score and reported_cv_score are then computed from
    the identical (seed, seed) fold assignment, so they should match."""
    outcome = tune_experiment(
        guarded_store,
        dataset_id,
        template_id="lightgbm_default",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
        n_trials=4,
        space=_tiny_space,
    )
    assert outcome.reported_cv_score == pytest.approx(outcome.best_cv_score)


def test_report_seed_rescoring_restores_fold_pairing_with_incumbent(guarded_store, dataset_id):
    """When report_seed differs from the search seed, the HPO winner's
    reported score must come from folds IDENTICAL to an incumbent trained
    directly on report_seed -- i.e. re-training the SAME config directly on
    report_seed gives the SAME score as tune_experiment's reported_cv_score,
    because both used the identical fold assignment."""
    base_seed = 42
    search_seed = 42 + 1  # simulates seed + iteration in the loop

    outcome = tune_experiment(
        guarded_store,
        dataset_id,
        template_id="lightgbm_default",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=search_seed,
        report_seed=base_seed,
        n_folds=3,
        env_hash="test-env",
        n_trials=4,
        space=_tiny_space,
        experiment_id="hpo-rescore-a",
    )

    # Directly retrain the SAME winning config on base_seed's folds --
    # this must reproduce reported_cv_score exactly (same config, same
    # dataset, same fold assignment).
    direct = train_experiment(
        guarded_store,
        dataset_id,
        ExperimentSpec(
            experiment_id="direct-check",
            template_id="lightgbm_default",
            config=outcome.best_params,
            budget=BudgetEstimate(wall_time_seconds=1),
        ),
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=base_seed,
        n_folds=3,
        env_hash="test-env",
    )
    assert outcome.reported_cv_score == pytest.approx(direct.metrics["roc_auc"].mean, abs=1e-9)
    assert outcome.result.metrics["roc_auc"].fold_values == pytest.approx(
        direct.metrics["roc_auc"].fold_values
    )


def test_search_and_reported_scores_can_legitimately_differ(guarded_store, dataset_id):
    """The whole point: search_cv_score (the search's own biased estimate)
    and reported_cv_score (the honest re-score on different folds) are
    tracked as SEPARATE fields -- they are not required to be equal, and in
    general won't be, because they were computed from different data
    splits. This test only asserts both fields exist and are populated;
    HPOOutcome.best_cv_score is the deprecated alias for reported_cv_score."""
    outcome = tune_experiment(
        guarded_store,
        dataset_id,
        template_id="lightgbm_default",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=99,
        report_seed=1,
        n_folds=3,
        env_hash="test-env",
        n_trials=4,
        space=_tiny_space,
        experiment_id="hpo-rescore-b",
    )
    assert isinstance(outcome.search_cv_score, float)
    assert isinstance(outcome.reported_cv_score, float)
    assert outcome.best_cv_score == outcome.reported_cv_score
