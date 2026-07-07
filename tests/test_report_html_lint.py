"""Task 3 of REPORT_INTEGRITY_ROADMAP.md: ``lint_report_html`` is
defense-in-depth over ``validate_report_inputs`` (Task 2) -- it re-checks
the RENDERED HTML for wrong-task-type metric vocabulary, catching
contamination introduced by a future template edit or narrated-insight
text regardless of how it got there.

The decisive test here is ``test_real_regression_report_has_no_false_positives``:
a genuine regression report's technical template always prints a
"Calibration" section header and a "Not measured for this run." caption
(see mlteam/reporting/templates.py) -- if the linter matched prose words
instead of metric-name tokens, every regression report would fail. This
proves it doesn't.
"""

from __future__ import annotations

import pytest

from mlteam.contracts import TaskType
from mlteam.reporting.contract import ReportContractError, lint_report_html


def test_regression_html_with_classification_metric_raises():
    html = "<html><body><h2>rmse</h2><p>roc_auc: 0.91</p></body></html>"
    with pytest.raises(ReportContractError, match="roc_auc"):
        lint_report_html(html, task_type=TaskType.REGRESSION, primary_metric="rmse")


def test_classification_html_with_regression_metric_raises():
    html = "<html><body><h2>roc_auc</h2><p>rmse: 12.3</p></body></html>"
    with pytest.raises(ReportContractError, match="rmse"):
        lint_report_html(html, task_type=TaskType.BINARY_CLASSIFICATION, primary_metric="roc_auc")


def test_report_missing_its_own_primary_metric_raises():
    html = "<html><body><h2>Results</h2><p>Everything looks fine.</p></body></html>"
    with pytest.raises(ReportContractError, match="primary_metric"):
        lint_report_html(html, task_type=TaskType.REGRESSION, primary_metric="rmse")


def test_clean_regression_html_passes():
    html = "<html><body><h2>rmse</h2><p>mae: 1.2, r2: 0.8</p></body></html>"
    lint_report_html(html, task_type=TaskType.REGRESSION, primary_metric="rmse")  # no raise


def test_clean_classification_html_passes():
    html = "<html><body><h2>roc_auc</h2><p>pr_auc: 0.7, f1_macro: 0.6</p></body></html>"
    lint_report_html(
        html, task_type=TaskType.BINARY_CLASSIFICATION, primary_metric="roc_auc"
    )  # no raise


def test_calibration_section_header_is_not_a_false_positive_on_regression():
    """The literal bug class this guards against: a naive substring/prose
    match on "calibration" would fire on the technical template's ALWAYS-
    PRESENT "Calibration" section header, even on a genuine regression
    report where calibration was correctly never measured."""
    html = (
        "<html><body><h2>Calibration</h2>"
        "<p class='muted'>Not measured for this run.</p>"
        "<h2>Cross-validation metrics</h2><p>rmse: 12.3</p></body></html>"
    )
    lint_report_html(html, task_type=TaskType.REGRESSION, primary_metric="rmse")  # no raise


def test_matches_whole_metric_tokens_not_substrings():
    """'r2' as a forbidden token for classification must not fire on
    unrelated text that happens to contain "r2" as a substring (e.g. a
    hypothetical column name or id) -- only a standalone word boundary
    match counts."""
    html = "<html><body><h2>roc_auc</h2><p>experiment ver2_final: 0.9</p></body></html>"
    lint_report_html(html, task_type=TaskType.BINARY_CLASSIFICATION, primary_metric="roc_auc")


def test_real_regression_report_has_no_false_positives(trained_regression_report):
    """Integration proof: build_final_report already calls lint_report_html
    internally (Task 3 wiring) -- a genuine regression report from a real
    trained pipeline must render successfully, i.e. the linter did not
    reject its own always-present Calibration/decision-threshold "not
    applicable" text."""
    business_html, technical_html = trained_regression_report
    # Would have raised inside build_final_report already if it were going
    # to; re-running here documents the exact invariant under test.
    lint_report_html(business_html, task_type=TaskType.REGRESSION, primary_metric="rmse")
    lint_report_html(technical_html, task_type=TaskType.REGRESSION, primary_metric="rmse")


@pytest.fixture(scope="module")
def trained_regression_report(tmp_path_factory):
    # A regression target derived from an existing synthetic generator's
    # numeric column keeps this fixture self-contained without inventing a
    # second dataset generator just for this test module.
    import polars as pl

    from mlteam.agents import critique_evaluation
    from mlteam.contracts import ProblemSpec, new_artifact
    from mlteam.llm import FixtureTransport, LLMClient
    from mlteam.orchestration import run_tabular_pipeline
    from mlteam.reporting import ReportInputs, build_final_report
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = tmp_path_factory.mktemp("lint-regression")
    csv_path = tmp / "prices.csv"
    frame = make_churn_frame(n_rows=200).with_columns(
        (pl.col("monthly_charges") * 12).alias("annual_spend")
    ).drop("churned")
    frame.write_csv(csv_path)

    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    run_id = "lint-regression-run"
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
    model = next(
        r.model for k, r in result.train_results.items() if "dummy" not in k
    )
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=run_id,
        task_type=TaskType.REGRESSION,
        target_column="annual_spend",
        primary_metric="rmse",
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
        primary_metric="rmse",
        run_id=run_id,
    )
    inputs = ReportInputs(
        problem=problem,
        model=model,
        evaluation=result.evaluation_report,
        critique=critique,
        split_plan=store.get(result.split_plan_id),
        goal_text="predict annual spend",
        best_template_id=result.best_template_id,
        best_experiment_id=model.experiment_id,
    )
    report = build_final_report(store, inputs, run_id=run_id)
    business_html = store.blob_path(report.business_report_blob).read_text(encoding="utf-8")
    technical_html = store.blob_path(report.technical_report_blob).read_text(encoding="utf-8")
    return business_html, technical_html
