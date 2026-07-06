"""Phase 6: residual analysis + plain-language insight narration.

The platform produces excellent structured evidence but until now shipped no
plain-language findings, and regression runs had no error analysis at all.
These tests prove the residual slices land for regression, and that the
narrator turns evidence into findings while never blocking the run on a bad
generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from mlteam.agents.explainer import explain_model
from mlteam.agents.insight_narrator import narrate_insights
from mlteam.contracts import ProblemSpec, TaskType, new_artifact
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import run_tabular_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-insights"


def _fresh_store(root: Path) -> GuardedArtifactStore:
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


def _regression_run(tmp_path: Path):
    """A regression run where one region is deliberately hard to predict."""
    rng = np.random.default_rng(0)
    n = 240
    region = rng.choice(["north", "south", "east"], n)
    size = rng.uniform(50, 200, n)
    # 'east' carries a large offset the model will struggle to fit -> worst slice
    price = 3.0 * size + np.where(region == "east", 500, 0) + rng.normal(0, 20, n)
    df = pl.DataFrame({"region": region, "size": size, "price": np.round(price, 2)})
    csv = tmp_path / "homes.csv"
    df.write_csv(csv)
    store = _fresh_store(tmp_path / "store")
    result = run_tabular_pipeline(
        store,
        csv,
        target_column="price",
        task_type=TaskType.REGRESSION,
        primary_metric="rmse",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config={"n_estimators": 30},
    )
    model = store.get(result.evaluation_report.model_ref.artifact_id)
    return store, result, model


def test_residual_analysis_produces_figures_and_worst_slice(tmp_path: Path):
    store, result, model = _regression_run(tmp_path)
    report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_is_classification=False,
        run_id=result.run_id,
        seed=42,
    )
    figure_kinds = {f.kind for f in report.figures}
    assert "residual_hist" in figure_kinds
    assert "residual_vs_predicted" in figure_kinds

    assert len(report.residual_slices) >= 2
    assert all(s.feature == "region" for s in report.residual_slices)
    # Worst slice is first, and it should be the deliberately-hard 'east' region.
    assert report.residual_slices[0].value == "east"
    assert report.residual_slices[0].mae > report.residual_slices[-1].mae


def test_classification_run_has_no_residual_slices(tmp_path: Path):
    from synthetic import make_churn_frame

    csv = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv)
    store = _fresh_store(tmp_path / "store")
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
    model = store.get(result.evaluation_report.model_ref.artifact_id)
    report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_is_classification=True,
        run_id=result.run_id,
        seed=42,
    )
    assert report.residual_slices == ()
    assert not any(f.kind.startswith("residual") for f in report.figures)


def _spec() -> ProblemSpec:
    return new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="rmse",
    )


def test_narrator_builds_findings_and_caveats(tmp_path: Path):
    store, result, model = _regression_run(tmp_path)
    explain_report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_is_classification=False,
        run_id=result.run_id,
        seed=42,
    )
    client = LLMClient(
        FixtureTransport.from_texts(
            json.dumps(
                {
                    "findings": ["Size is the dominant driver of price.", "East is hardest."],
                    "caveats": ["Measured on historical data."],
                }
            )
        ),
        model="claude-sonnet-4-6",
    )
    narrative = narrate_insights(
        client,
        store,
        problem=_spec(),
        evaluation=result.evaluation_report,
        explainability=explain_report,
        run_id=result.run_id,
    )
    assert narrative is not None
    assert len(narrative.findings) == 2
    assert len(narrative.caveats) == 1
    assert store.exists(narrative.meta.artifact_id)


def test_narrator_skips_loudly_on_bad_generation(tmp_path: Path):
    store, result, model = _regression_run(tmp_path)
    explain_report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_is_classification=False,
        run_id=result.run_id,
        seed=42,
    )
    # Invalid JSON on every attempt -> LLMValidationError -> None, never raises.
    client = LLMClient(
        FixtureTransport.from_texts("not json", "still not json", "nope"),
        model="claude-sonnet-4-6",
    )
    narrative = narrate_insights(
        client,
        store,
        problem=_spec(),
        evaluation=result.evaluation_report,
        explainability=explain_report,
        run_id=result.run_id,
    )
    assert narrative is None
