"""Regression: engineered feature columns must be re-derived on the holdout.

The feature executor runs post-split on the train partition only, so the
frozen holdout is stored WITHOUT the engineered columns. If the report scores
that raw holdout, every metric silently fails with "columns are missing" and
the final report shows no true out-of-sample numbers. build_final_report must
replay the (row-local, leakage-free) feature plan on the holdout first.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from mlteam.agents.feature_executor import apply_feature_plan_to_frame
from mlteam.contracts import FeatureOpType, FeaturePlan, FeatureProposal, new_artifact

RUN = "test-holdout-features"


def _plan(dataset_ref) -> FeaturePlan:
    return new_artifact(
        FeaturePlan,
        producer_agent="feature-engineer",
        run_id=RUN,
        dataset_ref=dataset_ref,
        proposals=(
            FeatureProposal(
                name="ratio_ab",
                op_type=FeatureOpType.RATIO,
                input_columns=("a", "b"),
                params={},
                rationale="test",
            ),
            FeatureProposal(
                name="log_a",
                op_type=FeatureOpType.LOG_TRANSFORM,
                input_columns=("a",),
                params={},
                rationale="test",
            ),
        ),
    )


def test_apply_feature_plan_replays_row_local_ops(local_store):
    from mlteam.agents.data_io import store_dataframe
    from mlteam.contracts import DatasetRole

    train = pl.DataFrame({"a": [10.0, 20.0, 30.0], "b": [2.0, 4.0, 5.0], "y": [0, 1, 0]})
    ds = store_dataframe(
        local_store,
        train,
        role=DatasetRole.CLEANED,
        producer_agent="t",
        run_id=RUN,
        target_column="y",
    )
    plan = _plan(ds.ref())

    # A DIFFERENT frame (stands in for the holdout) gets the same columns.
    holdout = pl.DataFrame({"a": [40.0, 50.0], "b": [8.0, 10.0], "y": [1, 0]})
    featured = apply_feature_plan_to_frame(holdout, plan)
    assert "ratio_ab" in featured.columns
    assert "log_a" in featured.columns
    assert featured["ratio_ab"].to_list() == [5.0, 5.0]


def test_full_report_scores_holdout_with_engineered_features(tmp_path: Path):
    """End-to-end: a churn run whose winning model was trained on engineered
    features must still produce non-empty holdout metrics in the report."""
    from mlteam.llm import FixtureTransport, LLMClient
    from mlteam.orchestration import AutoApproveGate, run_iterative_pipeline
    from mlteam.orchestration.budget import RunBudget
    from mlteam.reporting import ReportInputs, build_final_report
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    fixture = Path(__file__).parent / "fixtures" / "llm" / "iterative_churn_approved.json"
    csv = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(fixture), model="claude-sonnet-4-6")

    result = run_iterative_pipeline(
        store,
        csv,
        goal_text="reduce churn",
        client=client,
        gate=AutoApproveGate(),
        budget=RunBudget(max_iterations=2),
        seed=42,
        n_folds=3,
        hpo_trials=6,
    )
    pipeline = result.initial.pipeline
    # The fixture's feature plan applies real engineered columns.
    assert pipeline.feature_dataset_id is not None
    assert result.initial.feature_plan_id is not None

    report = build_final_report(
        store,
        ReportInputs(
            problem=result.initial.problem_spec,
            model=result.best_model,
            evaluation=result.final_report,
            critique=result.final_critique,
            split_plan=store.get(pipeline.split_plan_id),
            goal_text="reduce churn",
            best_template_id=pipeline.best_template_id,
            best_experiment_id=result.best_experiment_id,
            feature_plan=store.get(result.initial.feature_plan_id),
        ),
        run_id=result.initial.run_id,
    )
    # The bug: without the replay, the holdout table renders "unavailable".
    technical_html = store.blob_path(report.technical_report_blob).read_text("utf-8")
    assert "Holdout metrics unavailable" not in technical_html
    assert "Holdout (out-of-sample) metrics" in technical_html
    # The primary metric must actually appear as a scored holdout row.
    assert "roc_auc" in technical_html


def test_full_report_without_feature_plan_still_scores_holdout(tmp_path: Path):
    """No feature plan (feature_plan=None) must not break holdout scoring —
    the raw holdout already matches what the model was trained on."""
    from mlteam.contracts import (
        CritiqueReport,
        ProblemSpec,
        TaskType,
        Verdict,
    )
    from mlteam.orchestration import run_tabular_pipeline
    from mlteam.reporting import ReportInputs, build_final_report
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    csv = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    result = run_tabular_pipeline(
        store,
        csv,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config={"n_estimators": 30},
    )
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=result.run_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )
    store.put(spec)
    critique = new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=result.run_id,
        evaluation_ref=result.evaluation_report.ref(),
        verdict=Verdict.APPROVE,
    )
    store.put(critique)
    report = build_final_report(
        store,
        ReportInputs(
            problem=spec,
            model=store.get(result.evaluation_report.model_ref.artifact_id),
            evaluation=result.evaluation_report,
            critique=critique,
            split_plan=store.get(result.split_plan_id),
            goal_text="reduce churn",
            best_template_id=result.best_template_id,
            best_experiment_id="exp-lightgbm",
            feature_plan=None,
        ),
        run_id=result.run_id,
    )
    technical_html = store.blob_path(report.technical_report_blob).read_text("utf-8")
    assert "Holdout metrics unavailable" not in technical_html
