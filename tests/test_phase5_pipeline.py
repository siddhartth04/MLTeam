"""Explainability, deployment, and final-report tests over a real trained model.

A single module-scoped deterministic pipeline run produces the model + profile
+ split + evaluation; the Phase 5 components are exercised on those artifacts.
"""

from __future__ import annotations

import pytest

from mlteam.agents import (
    build_deployment_bundle,
    critique_evaluation,
    explain_model,
    materialize_bundle,
    understand_requirements,
)
from mlteam.contracts import TaskType
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import AutoApproveGate, run_tabular_pipeline
from mlteam.reporting import ReportInputs, build_final_report

RUN = "phase5-run"
FAST_LGBM = {"n_estimators": 30}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = tmp_path_factory.mktemp("phase5")
    csv_path = tmp / "churn.csv"
    make_churn_frame(n_rows=300).write_csv(csv_path)
    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config=FAST_LGBM,
    )
    best = result.train_results[
        next(k for k in result.train_results if k != "exp-dummy" and "dummy" not in k)
    ]
    return store, result, best.model


# -- Explainability --------------------------------------------------------------


@pytest.fixture(scope="module")
def explainability(trained):
    store, result, model = trained
    report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        n_folds=3,
        run_id=RUN,
    )
    return store, report


def test_explainability_ranks_features(explainability):
    _, report = explainability
    assert len(report.importances) >= 1
    # importances are sorted descending
    values = [fi.importance for fi in report.importances]
    assert values == sorted(values, reverse=True)


def test_explainability_produces_permutation_figure(explainability):
    store, report = explainability
    kinds = {f.kind for f in report.figures}
    assert "permutation_importance" in kinds
    for fig in report.figures:
        assert store.blob_exists(fig.blob_hash)


def test_explainability_holdout_untouched(explainability):
    store, _ = explainability
    assert store.holdout_untouched() is True


# -- Deployment ------------------------------------------------------------------


@pytest.fixture(scope="module")
def deployment(trained):
    store, result, model = trained
    profile = store.get(result.train_profile_id)
    problem = understand_requirements(
        LLMClient(
            FixtureTransport.from_texts(
                '{"task_type": "binary_classification", "target_column": "churned",'
                ' "primary_metric": "roc_auc"}'
            ),
            model="claude-sonnet-4-6",
        ),
        store,
        goal_text="predict churn",
        dataset_columns=["churned", "tenure_months"],
        run_id=RUN,
    )

    from mlteam.agents import read_dataset

    features = [
        c
        for c in read_dataset(store, store.get(result.cleaned_dataset_id)).columns
        if c != "churned"
    ]
    bundle = build_deployment_bundle(
        store,
        model=model,
        profile=profile,
        problem=problem,
        train_dataset_id=result.cleaned_dataset_id,
        feature_names=features,
        gate=AutoApproveGate(),
        run_id=RUN,
    )
    return store, model, bundle


def test_deployment_bundle_has_all_blobs(deployment):
    store, _, bundle = deployment
    assert bundle.human_signed_off is True
    for blob in (
        bundle.inference_schema_blob,
        bundle.app_blob,
        bundle.dockerfile_blob,
        bundle.drift_baseline_blob,
    ):
        assert blob and store.blob_exists(blob)


def test_materialized_bundle_is_runnable_layout(deployment, tmp_path):
    store, model, bundle = deployment
    target = tmp_path / "deploy"
    materialize_bundle(store, bundle, model, target)
    assert (target / "serve.py").exists()
    assert (target / "Dockerfile").exists()
    assert (target / "requirements.txt").exists()
    assert (target / "model" / "pipeline.joblib").exists()
    assert (target / "model" / "inference_schema.json").exists()


def test_materialize_refuses_unsigned_bundle(deployment, tmp_path):
    store, model, bundle = deployment
    unsigned = bundle.model_copy(update={"human_signed_off": False})
    with pytest.raises(ValueError, match="Gate 3"):
        materialize_bundle(store, unsigned, model, tmp_path / "no")


def test_evidently_drift_report_against_baseline(deployment):
    """Drift report runs against the deployment's stored training baseline."""
    import polars as pl

    from mlteam.agents.deployer import evidently_drift_report

    store, _, bundle = deployment
    # Using the training features as "current" data is a valid drift computation.
    baseline = pl.read_parquet(store.blob_path(bundle.drift_baseline_blob)).to_pandas()
    result = evidently_drift_report(store, bundle, baseline)
    assert "metrics" in result


# -- Final report ----------------------------------------------------------------


def test_final_report_reads_holdout_and_renders(trained, explainability):
    store, result, model = trained
    _, explain_report = explainability

    from mlteam.contracts import ProblemSpec, new_artifact

    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
        human_confirmed=True,
    )
    critique = critique_evaluation(
        LLMClient(
            FixtureTransport.from_texts(
                '{"findings": [], "verdict": "approve", "iteration_actions": []}'
            ),
            model="claude-sonnet-4-6",
        ),
        store,
        report=result.evaluation_report,
        primary_metric="roc_auc",
        run_id=RUN,
    )
    inputs = ReportInputs(
        problem=problem,
        model=model,
        evaluation=result.evaluation_report,
        critique=critique,
        split_plan=store.get(result.split_plan_id),
        goal_text="identify customers likely to churn",
        best_template_id=result.best_template_id,
        best_experiment_id=model.experiment_id,
        explainability=explain_report,
    )
    final = build_final_report(store, inputs, run_id=RUN)

    business = store.blob_path(final.business_report_blob).read_text(encoding="utf-8")
    technical = store.blob_path(final.technical_report_blob).read_text(encoding="utf-8")
    assert "<!doctype html>" in business.lower()
    assert "Holdout" in technical
    assert "roc_auc" in final.headline_metrics  # holdout metric captured
    assert final.headline_metrics["roc_auc"] > 0.0

    # The embedded Plotly JSON must survive rendering intact (autoescape must
    # NOT have turned its quotes into &#34; inside the <script> block).
    assert "Plotly.newPlot" in technical
    assert "&#34;" not in _script_bodies(technical)

    # Headline cards visually distinguish true holdout (out-of-sample) from
    # CV-mean (in-sample) evidence (audit L6) rather than one undifferentiated
    # grid a reader could mistake for uniformly out-of-sample.
    assert "metric-card--holdout" in business
    assert "out-of-sample" in business
    assert "cross-validation" in business


def _script_bodies(html: str) -> str:
    import re

    return "\n".join(re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL))


def test_report_access_logged_as_final_reporting(trained):
    """Reading the holdout for the report must be the ONLY granted access,
    and must keep holdout_untouched True (it's the sanctioned unlock)."""
    store, _, _ = trained
    granted = [r for r in store.access_log() if r.granted]
    assert all(r.context == "final_reporting" for r in granted)
    assert store.holdout_untouched() is True
