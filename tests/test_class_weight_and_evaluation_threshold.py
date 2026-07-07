"""Audit Task 7: class_weight whitelisting (M6) + build_evaluation_report's
real OOF-based calibration/decision-threshold wiring (H7, H8).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from mlteam.agents.data_io import store_dataframe
from mlteam.agents.experiment_planner import plan_experiments
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.trainer import train_experiment
from mlteam.contracts import (
    BudgetEstimate,
    CostAsymmetry,
    DatasetRole,
    ExperimentSpec,
    TaskType,
)
from mlteam.evaluation.report import build_evaluation_report
from mlteam.evaluation.results import TrainResult
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-classweight-threshold"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


def _store(tmp_path) -> GuardedArtifactStore:
    root = tmp_path / "store"
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


def test_planner_rejects_class_weight_on_incapable_template(tmp_path, guarded_store, churn_csv):
    from mlteam.agents import ingest_file
    from mlteam.contracts import ProblemSpec, new_artifact

    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )
    guarded_store.put(spec)

    bad = json.dumps(
        {
            "experiments": [
                {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
                {
                    "experiment_id": "e1",
                    "template_id": "xgboost_default",  # does NOT accept class_weight
                    "config": {"class_weight": "balanced"},
                    "is_baseline": False,
                },
            ],
            "rationale": "test",
        }
    )
    good = json.dumps(
        {
            "experiments": [
                {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
                {
                    "experiment_id": "e1",
                    "template_id": "lightgbm_default",  # DOES accept it
                    "config": {"class_weight": "balanced"},
                    "is_baseline": False,
                },
            ],
            "rationale": "test",
        }
    )
    client, transport = _client(bad, good)
    plan = plan_experiments(
        client, guarded_store, profile=profile, problem=spec, n_rows=200, run_id=RUN
    )
    assert "does not accept class_weight" in transport.requests[1].user_content
    revised = next(e for e in plan.experiments if e.experiment_id == "e1")
    assert revised.config.get("class_weight") == "balanced"


def test_build_evaluation_report_populates_real_calibration_and_threshold(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"x": x, "target": y})
    store = _store(tmp_path)
    dataset = store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id=RUN,
        target_column="target",
    )

    model_spec = ExperimentSpec(
        experiment_id="exp-model",
        template_id="lightgbm_default",
        config={"n_estimators": 30},
        budget=BudgetEstimate(wall_time_seconds=5),
    )
    dummy_spec = ExperimentSpec(
        experiment_id="exp-dummy",
        template_id="dummy_baseline",
        budget=BudgetEstimate(wall_time_seconds=1),
        is_baseline=True,
    )
    model_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        model_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
    )
    dummy_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        dummy_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
    )
    profile = profile_dataset(store, dataset.meta.artifact_id, run_id=RUN, seed=42)

    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=TrainResult(model=model_train.model, metrics=model_train.metrics),
        dummy_result=TrainResult(model=dummy_train.model, metrics=dummy_train.metrics),
        profile=profile,
        primary_metric="roc_auc",
        n_rows=n,
        dataset_id=dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        cost_asymmetry=CostAsymmetry(false_positive_cost=1.0, false_negative_cost=5.0),
        n_folds=3,
        seed=42,
    )
    assert report.calibration.expected_calibration_error is not None
    assert report.calibration.brier_score is not None
    assert report.decision_threshold is not None
    assert 0.0 <= report.decision_threshold.threshold <= 1.0
    assert report.decision_threshold.false_negative_cost == 5.0


def test_build_evaluation_report_omits_threshold_without_cost_asymmetry(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"x": x, "target": y})
    store = _store(tmp_path)
    dataset = store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id=RUN,
        target_column="target",
    )
    model_spec = ExperimentSpec(
        experiment_id="exp-model",
        template_id="lightgbm_default",
        config={"n_estimators": 20},
        budget=BudgetEstimate(wall_time_seconds=5),
    )
    dummy_spec = ExperimentSpec(
        experiment_id="exp-dummy",
        template_id="dummy_baseline",
        budget=BudgetEstimate(wall_time_seconds=1),
        is_baseline=True,
    )
    model_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        model_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    dummy_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        dummy_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    profile = profile_dataset(store, dataset.meta.artifact_id, run_id=RUN, seed=1)

    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=TrainResult(model=model_train.model, metrics=model_train.metrics),
        dummy_result=TrainResult(model=dummy_train.model, metrics=dummy_train.metrics),
        profile=profile,
        primary_metric="roc_auc",
        n_rows=n,
        dataset_id=dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        # cost_asymmetry omitted entirely
        n_folds=3,
        seed=1,
    )
    assert report.calibration.expected_calibration_error is not None  # still measured
    assert report.decision_threshold is None  # but no threshold without a cost ratio


def test_build_evaluation_report_omits_both_without_dataset_id(tmp_path):
    """Backward compatibility: callers that don't supply dataset_id/task_type
    (none currently in the codebase, but the params are optional) get the
    exact old behavior -- an empty CalibrationSummary, no threshold, no
    OOF refit attempted at all."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"x": x, "target": y})
    store = _store(tmp_path)
    dataset = store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id=RUN,
        target_column="target",
    )
    model_spec = ExperimentSpec(
        experiment_id="exp-model",
        template_id="lightgbm_default",
        config={"n_estimators": 15},
        budget=BudgetEstimate(wall_time_seconds=5),
    )
    dummy_spec = ExperimentSpec(
        experiment_id="exp-dummy",
        template_id="dummy_baseline",
        budget=BudgetEstimate(wall_time_seconds=1),
        is_baseline=True,
    )
    model_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        model_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    dummy_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        dummy_spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    profile = profile_dataset(store, dataset.meta.artifact_id, run_id=RUN, seed=1)

    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=TrainResult(model=model_train.model, metrics=model_train.metrics),
        dummy_result=TrainResult(model=dummy_train.model, metrics=dummy_train.metrics),
        profile=profile,
        primary_metric="roc_auc",
        n_rows=n,
        # dataset_id/task_type both omitted
    )
    assert report.calibration.expected_calibration_error is None
    assert report.decision_threshold is None
