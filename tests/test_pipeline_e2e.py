"""End-to-end Phase 1 proofs: full pipeline, holdout untouched, reproducibility.

TabPFN is excluded here (it downloads model weights on first use); the
example script exercises it. Dummy + LightGBM prove the substrate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.contracts import TaskType
from mlteam.orchestration import run_tabular_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

FAST_LGBM = {"n_estimators": 30}


def _run(store: GuardedArtifactStore, csv_path: Path, run_id: str | None = None):
    return run_tabular_pipeline(
        store,
        csv_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=run_id,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config=FAST_LGBM,
    )


def _fresh_store(root: Path) -> GuardedArtifactStore:
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    from synthetic import make_churn_frame

    csv_path = tmp / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    store = _fresh_store(tmp / "store")
    return store, _run(store, csv_path)


def test_pipeline_produces_evaluation_report(pipeline_run):
    _, result = pipeline_run
    report = result.evaluation_report
    assert result.best_template_id == "lightgbm_default"
    assert {m.name for m in report.cv_metrics} == {"roc_auc", "accuracy", "f1", "log_loss"}
    assert all(len(m.fold_values) == 3 for m in report.cv_metrics)


def test_model_beats_dummy_on_primary_metric(pipeline_run):
    _, result = pipeline_run
    auc = next(d for d in result.evaluation_report.baseline_deltas if d.metric == "roc_auc")
    assert auc.model_value > auc.baseline_value
    assert auc.significant is True


def test_holdout_untouched_through_full_run(pipeline_run):
    store, result = pipeline_run
    assert result.evaluation_report.holdout_untouched is True
    holdout_check = next(
        c for c in result.evaluation_report.leakage_checks if c.name == "holdout_untouched"
    )
    assert holdout_check.passed is True
    assert store.holdout_untouched() is True


def test_leakage_checks_all_pass_on_clean_data(pipeline_run):
    _, result = pipeline_run
    assert all(c.passed for c in result.evaluation_report.leakage_checks)


def test_lineage_traces_report_back_to_raw_data(pipeline_run):
    store, result = pipeline_run
    ancestors = store._inner.lineage.ancestry(result.evaluation_report.meta.artifact_id)
    assert result.raw_dataset_id in ancestors
    assert result.cleaned_dataset_id in ancestors


def test_leaky_dataset_fails_leakage_checks(tmp_path, leaky_churn_csv):
    store = _fresh_store(tmp_path / "store")
    result = _run(store, leaky_churn_csv)
    near_perfect = next(
        c for c in result.evaluation_report.leakage_checks if c.name == "near_perfect_features"
    )
    assert near_perfect.passed is False
    assert "leaked_score" in near_perfect.details


def test_reproducibility_two_runs_identical(tmp_path, churn_csv):
    """Same data + config in two FRESH stores must give identical artifacts."""
    result_a = _run(_fresh_store(tmp_path / "store_a"), churn_csv, run_id="repro-a")
    result_b = _run(_fresh_store(tmp_path / "store_b"), churn_csv, run_id="repro-b")

    assert result_a.raw_dataset_id == result_b.raw_dataset_id
    assert result_a.cleaned_dataset_id == result_b.cleaned_dataset_id
    assert result_a.split_plan_id == result_b.split_plan_id
    assert result_a.train_profile_id == result_b.train_profile_id

    metrics_a = {m.name: m.fold_values for m in result_a.evaluation_report.cv_metrics}
    metrics_b = {m.name: m.fold_values for m in result_b.evaluation_report.cv_metrics}
    assert metrics_a == metrics_b

    assert (
        result_a.evaluation_report.meta.artifact_id == result_b.evaluation_report.meta.artifact_id
    )
