"""Task 3 of REPORT_INTEGRITY_ROADMAP.md: ``lint_rendered_metric_names`` is
defense-in-depth over ``validate_report_inputs`` (Task 2) -- it re-checks
the metric NAMES a report template actually rendered against the task
type's legal set.

This module's decisive test is
``test_llm_narrated_prose_using_metric_english_words_is_never_flagged``:
the original implementation scanned full rendered HTML for forbidden
tokens as whole words, and shipped a false positive to production the
first time it ran against a REAL run -- an LLM-narrated insight used the
ordinary English word "accuracy" ("prioritizing prediction accuracy for
negotiation") in a regression report, which a word-boundary regex cannot
distinguish from the classification metric of the same name. The run had
completed its entire two-hour pipeline (training, evaluation, critic,
explainability, Gate 3 approval) and failed only at the final report
step. This module proves the redesign -- checking the actual metric-name
values a template rendered, never the page text -- closes that class of
false positive by construction: free-text narration is simply never in
the input list checked.
"""

from __future__ import annotations

import pytest

from mlteam.contracts import TaskType
from mlteam.reporting.contract import ReportContractError, lint_rendered_metric_names


def test_regression_report_rendering_a_classification_metric_raises():
    with pytest.raises(ReportContractError, match="accuracy"):
        lint_rendered_metric_names(
            ["rmse", "mae", "accuracy"], task_type=TaskType.REGRESSION, primary_metric="rmse"
        )


def test_classification_report_rendering_a_regression_metric_raises():
    with pytest.raises(ReportContractError, match="rmse"):
        lint_rendered_metric_names(
            ["roc_auc", "rmse"], task_type=TaskType.BINARY_CLASSIFICATION, primary_metric="roc_auc"
        )


def test_report_not_rendering_its_own_primary_metric_raises():
    with pytest.raises(ReportContractError, match="primary_metric"):
        lint_rendered_metric_names(
            ["mae", "r2"], task_type=TaskType.REGRESSION, primary_metric="rmse"
        )


def test_clean_regression_metric_names_pass():
    lint_rendered_metric_names(
        ["rmse", "mae", "r2"], task_type=TaskType.REGRESSION, primary_metric="rmse"
    )  # no raise


def test_clean_classification_metric_names_pass():
    lint_rendered_metric_names(
        ["roc_auc", "pr_auc", "f1_macro"],
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
    )  # no raise


def test_empty_metric_list_does_not_raise_a_spurious_primary_metric_violation():
    """An empty list (e.g. a report section that legitimately renders no
    metrics table) must not be treated as "missing" the primary metric --
    there is nothing to check, which is different from checking and
    failing."""
    lint_rendered_metric_names([], task_type=TaskType.REGRESSION, primary_metric="rmse")


def test_llm_narrated_prose_using_metric_english_words_is_never_flagged():
    """The actual production incident, reduced to its essence: the ordinary
    English word "accuracy" appearing in LLM-narrated prose about a
    regression model must never reach this function's input in the first
    place, because callers pass only the literal metric-name values a
    template rendered (headline card labels, CV/holdout table rows,
    baseline-delta rows) -- never free-text insight/critic strings. This
    test documents that contract: passing the correct (metrics-only) list
    for a report whose prose elsewhere says "prioritizing prediction
    accuracy for negotiation" does not raise, because that sentence is
    never part of what's checked.
    """
    # This IS what builder.py actually passes: real metric names only, not
    # the narrative text that happens to contain the word "accuracy".
    rendered_metric_names = ["rmse", "mae", "r2"]
    lint_rendered_metric_names(
        rendered_metric_names, task_type=TaskType.REGRESSION, primary_metric="rmse"
    )  # no raise -- "accuracy" the English word was never in this list


def test_build_final_report_survives_insight_prose_containing_metric_english_words(
    tmp_path_factory,
):
    """End-to-end reproduction of the actual production incident: an
    InsightNarrative whose findings/caveats use the English word "accuracy"
    in ordinary prose about a REGRESSION run must not fail report
    generation. This is the exact failure that took down a completed,
    Gate-3-approved, two-hour house-prices pipeline run in production --
    build_final_report raised ReportContractError at the very last step
    because the (now-removed) HTML-scanning linter matched "accuracy"
    inside narrated prose. Uses a real trained pipeline (not hand-built
    artifacts): build_final_report scores a genuine holdout dataset through
    the serialized sklearn pipeline, which minimal fake artifacts can't
    provide.
    """
    import polars as pl

    from mlteam.contracts import (
        CritiqueReport,
        InsightNarrative,
        ProblemSpec,
        Verdict,
        new_artifact,
    )
    from mlteam.orchestration import run_tabular_pipeline
    from mlteam.reporting.builder import ReportInputs, build_final_report
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = tmp_path_factory.mktemp("metric-lint-regression")
    csv_path = tmp / "prices.csv"
    frame = (
        make_churn_frame(n_rows=200)
        .with_columns((pl.col("monthly_charges") * 12).alias("annual_spend"))
        .drop("churned")
    )
    frame.write_csv(csv_path)

    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    run_id = "metric-lint-regression-run"
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="annual_spend",
        task_type=TaskType.REGRESSION,
        primary_metric="rmse",
        run_id=run_id,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
    )
    model = next(r.model for k, r in result.train_results.items() if "dummy" not in k)
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=run_id,
        task_type=TaskType.REGRESSION,
        target_column="annual_spend",
        primary_metric="rmse",
        human_confirmed=True,
    )
    # The exact production shape: an LLM narrated a finding using the
    # ordinary English word "accuracy", about a REGRESSION run.
    insights = new_artifact(
        InsightNarrative,
        producer_agent="test",
        run_id=run_id,
        model_ref=model.ref(),
        findings=("The model prioritizes prediction accuracy for negotiation-ready pricing.",),
        caveats=("Accuracy may vary for listings far outside the training price range.",),
    )
    store.put(insights)

    critique = new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=run_id,
        evaluation_ref=result.evaluation_report.ref(),
        verdict=Verdict.APPROVE,
    )
    store.put(critique)

    inputs = ReportInputs(
        problem=problem,
        model=model,
        evaluation=result.evaluation_report,
        critique=critique,
        split_plan=store.get(result.split_plan_id),
        goal_text="predict annual spend",
        best_template_id=result.best_template_id,
        best_experiment_id=model.experiment_id,
        insights=insights,
    )

    # Must NOT raise: the incident this test reproduces.
    build_final_report(store, inputs, run_id=run_id)
