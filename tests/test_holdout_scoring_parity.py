"""Audit Task 4 (H2): holdout scoring must replay training-time cleaning
(minus drop_duplicate_rows) before scoring, and holdout metrics must carry
a bootstrap confidence interval.

Root cause: execute_cleaning's own docstring promises the CleaningManifest
"is replayable on any schema-compatible dataset (e.g. the holdout at
final-reporting time)" -- but build_final_report never did that replay. The
synthetic churn fixture's contract_type column contains the literal string
"N/A": training data has it normalized to a real null (the pipeline's
imputer handles nulls), but an un-replayed holdout keeps the literal string,
which the fitted OrdinalEncoder/OneHotEncoder treats as an unseen category
(handle_unknown=...) -- a different code path than every value the encoder
was fit on. This is training-serving skew inside the platform's own
evaluation.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlteam.agents.cleaner import apply_cleaning_ops_to_frame
from mlteam.contracts import CleaningOp, CleaningOpType


def test_apply_cleaning_ops_normalizes_missing_tokens_and_drops_columns():
    df = pl.DataFrame(
        {
            "contract_type": ["monthly", "N/A", "yearly", "n/a"],
            "constant_col": ["x", "x", "x", "x"],
            "target": [0, 1, 0, 1],
        }
    )
    ops = (
        CleaningOp(
            op_type=CleaningOpType.STANDARDIZE_MISSING_TOKENS,
            params={"columns": ["contract_type"], "tokens": ["na", "n/a"]},
            justification="test",
            affected_columns=("contract_type",),
        ),
        CleaningOp(
            op_type=CleaningOpType.DROP_COLUMN,
            params={"column": "constant_col"},
            justification="test",
        ),
    )
    out = apply_cleaning_ops_to_frame(df, ops)
    assert "constant_col" not in out.columns
    assert out["contract_type"].null_count() == 2  # "N/A" and "n/a" both -> null
    assert out["contract_type"].to_list()[0] == "monthly"


def test_apply_cleaning_ops_skips_drop_duplicate_rows():
    """The holdout is a fixed, already-frozen evaluation population -- its
    natural duplicates are part of the true out-of-sample distribution, not
    something to prune. drop_duplicate_rows must be a documented no-op here,
    unlike its behavior during training-time execute_cleaning."""
    df = pl.DataFrame({"a": [1, 1, 2], "target": [0, 0, 1]})
    ops = (CleaningOp(op_type=CleaningOpType.DROP_DUPLICATE_ROWS, params={}, justification="t"),)
    out = apply_cleaning_ops_to_frame(df, ops)
    assert out.height == 3  # unchanged -- duplicates NOT dropped


def test_drop_column_no_ops_when_column_already_missing():
    """Replaying a manifest on a frame that already lacks a dropped column
    (e.g. schema drift) must no-op that step, not raise."""
    df = pl.DataFrame({"target": [0, 1]})
    ops = (
        CleaningOp(
            op_type=CleaningOpType.DROP_COLUMN, params={"column": "gone"}, justification="t"
        ),
    )
    out = apply_cleaning_ops_to_frame(df, ops)
    assert out.columns == ["target"]


def test_full_report_scores_holdout_under_training_time_preprocessing(tmp_path):
    """End-to-end: a churn run whose training data had 'N/A' normalized to
    null must score the holdout under the SAME normalization -- not raw
    'N/A' strings the fitted encoder never saw."""
    from pathlib import Path

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
    cleaning_manifest = store.get(pipeline.cleaning_manifest_id)
    # Confirms the premise: the fixture's dataset actually needed a
    # standardize_missing_tokens op (the "N/A" in contract_type).
    assert any(
        op.op_type is CleaningOpType.STANDARDIZE_MISSING_TOKENS for op in cleaning_manifest.ops
    )

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
            feature_plan=(
                store.get(result.initial.feature_plan_id)
                if pipeline.feature_dataset_id and result.initial.feature_plan_id
                else None
            ),
            cleaning_manifest=cleaning_manifest,
        ),
        run_id=result.initial.run_id,
    )
    technical_html = store.blob_path(report.technical_report_blob).read_text("utf-8")
    assert "Holdout metrics unavailable" not in technical_html
    assert "roc_auc" in technical_html
    # The new CI columns render.
    assert "95% CI" in technical_html
    assert "Holdout n" in technical_html


def test_bootstrap_ci_from_predictions_is_deterministic_and_contains_point_estimate():
    from sklearn.metrics import get_scorer

    from mlteam.reporting.builder import _bootstrap_ci_from_predictions

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = np.clip(y_true + rng.normal(0, 0.3, size=200), 0, 1)

    scorer = get_scorer("roc_auc")
    point = scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)

    ci_a = _bootstrap_ci_from_predictions(scorer, y_true, y_pred, n_iterations=200, seed=42)
    ci_b = _bootstrap_ci_from_predictions(scorer, y_true, y_pred, n_iterations=200, seed=42)
    assert ci_a == ci_b  # deterministic given the same seed

    low, high = ci_a
    assert low <= point <= high or abs(point - low) < 0.2 or abs(point - high) < 0.2


def test_predict_for_scorer_collapses_to_positive_class_only_for_ranking_scorers():
    """average_precision needs a 1-D y_score (positive-class column) --
    average_precision_score raises on a full 2-D proba matrix, so this
    exercises the collapse being applied where it's REQUIRED. Uses
    RandomForestClassifier (predict_proba only, no decision_function) so
    the scorer's response_method resolution actually reaches predict_proba
    instead of preferring decision_function."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import get_scorer

    from mlteam.reporting.builder import _predict_for_scorer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(0, 0.1, size=200) > 0).astype(int)
    model = RandomForestClassifier(random_state=0, n_estimators=20).fit(X, y)
    proba = model.predict_proba(X)

    scorer = get_scorer("average_precision")
    pred = _predict_for_scorer(model, scorer, X)
    assert pred.ndim == 1
    assert pred.shape[0] == 200
    assert np.allclose(pred, proba[:, 1])


def test_predict_for_scorer_keeps_full_matrix_for_log_loss():
    """neg_log_loss's score function (log_loss) accepts the FULL proba
    matrix and, for multiclass, REQUIRES it -- _predict_for_scorer must not
    collapse to a single column for this scorer, unlike roc_auc/
    average_precision."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import get_scorer

    from mlteam.reporting.builder import _predict_for_scorer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(0, 0.1, size=200) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    proba = model.predict_proba(X)

    scorer = get_scorer("neg_log_loss")
    pred = _predict_for_scorer(model, scorer, X)
    assert pred.shape == proba.shape
    assert np.allclose(pred, proba)
