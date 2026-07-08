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


def test_regression_task_never_calls_compute_oof_for_calibration(tmp_path, monkeypatch):
    """The actual bug this test guards against: build_evaluation_report used
    to call _calibration_and_threshold -> compute_oof unconditionally
    whenever dataset_id/task_type were BOTH given, regardless of task type
    -- paying the full cost of refitting the winning model per fold, only
    to discover afterward that oof.y_proba is None for a regression
    template and discard the entire result. A live house-prices run spent
    ~25 silent minutes here refitting a 5-base-learner stacking ensemble
    for a REGRESSION task, for a calibration/threshold pair that can never
    apply to regression in the first place.

    Monkeypatches compute_oof to raise if called at all -- proving the
    is_classification guard skips the refit before ever reaching it, not
    just discards its result afterward.
    """
    import mlteam.evaluation.oof as oof_module

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "compute_oof was called for a REGRESSION task -- calibration/"
            "decision-threshold only apply to classification, so the OOF "
            "refit must be skipped analytically, not run and discarded"
        )

    monkeypatch.setattr(oof_module, "compute_oof", _must_not_be_called)

    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = x * 2.0 + rng.normal(scale=0.3, size=n)
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
        task_type=TaskType.REGRESSION,
        run_id=RUN,
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )
    dummy_train = train_experiment(
        store,
        dataset.meta.artifact_id,
        dummy_spec,
        task_type=TaskType.REGRESSION,
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
        primary_metric="rmse",
        n_rows=n,
        dataset_id=dataset.meta.artifact_id,
        task_type=TaskType.REGRESSION,
        n_folds=3,
        seed=1,
    )
    # Same observable result as before the fix (empty/None) -- the fix
    # changes HOW that result is reached (skipped, not run-then-discarded),
    # which is exactly what the monkeypatch above proves.
    assert report.calibration.expected_calibration_error is None
    assert report.decision_threshold is None


def test_classification_calibration_refit_logs_per_fold_progress(tmp_path):
    """A single house-prices run's explainability stage spent 45+ silent
    minutes in an OOF refit with nothing on the dashboard distinguishing
    "working" from "hung". The SAME refit path (evaluation.oof.compute_oof)
    backs build_evaluation_report's calibration computation for
    classification tasks -- this proves a reporter passed through reaches
    compute_oof and receives one line per CV fold."""
    from dataclasses import dataclass, field

    @dataclass
    class RecordingReporter:
        log_calls: list[tuple[str, str, str | None]] = field(default_factory=list)

        def stage(self, stage: str, status: str, detail: str = "") -> None:
            pass

        def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None:
            self.log_calls.append((message, level, stage))

    rng = np.random.default_rng(0)
    n = 150
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

    reporter = RecordingReporter()
    build_evaluation_report(
        store,
        run_id=RUN,
        model_result=TrainResult(model=model_train.model, metrics=model_train.metrics),
        dummy_result=TrainResult(model=dummy_train.model, metrics=dummy_train.metrics),
        profile=profile,
        primary_metric="roc_auc",
        n_rows=n,
        dataset_id=dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        n_folds=3,
        seed=1,
        reporter=reporter,
    )

    evaluate_logs = [msg for msg, _level, stage in reporter.log_calls if stage == "evaluate"]
    assert len(evaluate_logs) >= 3 * 2, "expected at least a start+complete line per fold"
    assert any("fold 1/3" in msg for msg in evaluate_logs)
    assert any("fold 3/3" in msg for msg in evaluate_logs)
