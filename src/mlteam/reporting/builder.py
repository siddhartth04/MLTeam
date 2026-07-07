"""Report generator: business + technical HTML with embedded Plotly figures.

This is the ONLY component that reads the frozen holdout — and only through the
store guard's ``final_reporting_access``. It computes true out-of-sample metrics
on the holdout, renders both reports, stores them as blobs, and returns a
FinalReport artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from mlteam.agents.trainer import _temporal_to_epoch
from mlteam.contracts import (
    CleaningManifest,
    CritiqueReport,
    DatasetVersion,
    EvaluationReport,
    ExplainabilityReport,
    FeaturePlan,
    FinalReport,
    InsightNarrative,
    ProblemSpec,
    SplitPlan,
    TrainedModel,
    new_artifact,
    utcnow,
)
from mlteam.evaluation.metrics import scoring_for, to_natural_sign
from mlteam.reporting.figures import load_figure_json
from mlteam.reporting.templates import base_css, business_template, technical_template
from mlteam.serving.model_io import load_pipeline
from mlteam.store.guard import GuardedArtifactStore

_log = structlog.get_logger(__name__)

_BOOTSTRAP_ITERATIONS = 1000
_BOOTSTRAP_SEED = 20260101  # fixed -- CIs must be reproducible run to run
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


@dataclass(frozen=True)
class ReportInputs:
    problem: ProblemSpec
    model: TrainedModel
    evaluation: EvaluationReport
    critique: CritiqueReport
    split_plan: SplitPlan
    goal_text: str
    best_template_id: str
    best_experiment_id: str
    explainability: ExplainabilityReport | None = None
    insights: InsightNarrative | None = None
    # The row-local feature plan the model was trained with, if any. Replayed on
    # the frozen holdout at scoring time so its engineered columns exist there
    # too — without it, every holdout metric silently drops on "columns missing".
    feature_plan: FeaturePlan | None = None
    # The cleaning manifest applied to training data. Replayed on the holdout
    # (minus drop_duplicate_rows) BEFORE the feature plan, so the pipeline
    # scores the holdout under the SAME preprocessing it was trained under
    # instead of raw values (e.g. an un-normalized "N/A" string) the encoder
    # never saw at fit time.
    cleaning_manifest: CleaningManifest | None = None
    iterations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


# sklearn score functions whose signature requires a 1-D `y_score` (the
# positive-class probability/decision value) rather than the full
# (n, n_classes) matrix predict_proba returns. Every _SCORING entry using
# either of these is scoped to strictly-binary task types
# (BINARY_CLASSIFICATION, ANOMALY_DETECTION) — this is NOT a general "2
# columns means binary" heuristic, which would silently mis-handle
# multiclass log_loss (whose score function DOES want the full matrix).
_NEEDS_POSITIVE_CLASS_COLUMN = frozenset({"roc_auc_score", "average_precision_score"})


def _predict_for_scorer(pipeline: Any, scorer: Any, df: Any) -> np.ndarray:
    """Compute the prediction array a scorer needs, ONCE.

    sklearn scorers request one of predict / predict_proba /
    decision_function depending on the metric (e.g. roc_auc needs
    probabilities, rmse needs point predictions). Getting this once and
    bootstrap-resampling the RESULT (rather than re-invoking the full
    pipeline's .predict() inside every bootstrap iteration) turns an
    O(n_iterations) full-pipeline-inference cost into O(1) inference +
    O(n_iterations) cheap array resampling — the difference between a
    report taking milliseconds and one taking over a minute.
    """
    response_methods = scorer._response_method
    if isinstance(response_methods, str):
        response_methods = (response_methods,)
    for method_name in response_methods:
        method = getattr(pipeline, method_name, None)
        if method is None:
            continue
        prediction = method(df)
        if (
            method_name == "predict_proba"
            and scorer._score_func.__name__ in _NEEDS_POSITIVE_CLASS_COLUMN
            and prediction.ndim == 2
            and prediction.shape[1] == 2
        ):
            return prediction[:, 1]  # positive-class column; these two need 1-D y_score
        return prediction
    raise ValueError(
        f"pipeline exposes none of {response_methods} needed by scorer {scorer!r}"
    )


def _bootstrap_ci_from_predictions(
    scorer: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iterations: int = _BOOTSTRAP_ITERATIONS,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Seeded percentile bootstrap CI, resampling cached predictions.

    Resamples ROWS (with replacement) from the already-computed predictions
    and recomputes the metric on each resample — this describes sampling
    uncertainty in the metric given the size of the frozen holdout, which a
    single point estimate cannot convey (a 48-row holdout's third decimal
    place is noise, not precision). Deterministic: same seed -> same CI on
    every report generation for the same holdout + model.
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        scores[i] = scorer._sign * scorer._score_func(y_true[idx], y_pred[idx], **scorer._kwargs)
    low, high = np.percentile(scores, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(low), float(high)


def _holdout_metrics(store: GuardedArtifactStore, inputs: ReportInputs) -> list[dict[str, Any]]:
    """Score the model on the frozen holdout via the guarded unlock path."""
    pipeline = load_pipeline(store, inputs.model)
    from sklearn.metrics import get_scorer

    with store.final_reporting_access("final HTML report") as unlocked:
        holdout: DatasetVersion = unlocked.get(inputs.split_plan.holdout_ref.artifact_id)
        import polars as pl

        frame = pl.read_parquet(unlocked.blob_path(holdout.blob_hash))

    # Replay training-time preprocessing in the SAME order training applied
    # it (clean, then feature-engineer) so the holdout is scored under
    # identical preprocessing to training, not raw values the pipeline's
    # fitted encoders/imputers never saw.
    if inputs.cleaning_manifest is not None:
        from mlteam.agents.cleaner import apply_cleaning_ops_to_frame

        frame = apply_cleaning_ops_to_frame(frame, inputs.cleaning_manifest.ops)
    if inputs.feature_plan is not None:
        from mlteam.agents.feature_executor import apply_feature_plan_to_frame

        frame = apply_feature_plan_to_frame(frame, inputs.feature_plan)
    df = _temporal_to_epoch(frame).to_pandas()

    target = df.pop(inputs.problem.target_column)
    n_holdout = len(df)
    y_true = target.to_numpy()
    scoring = scoring_for(inputs.problem.task_type)
    metrics: list[dict[str, Any]] = []
    for name, scorer_name in scoring.items():
        try:
            scorer = get_scorer(scorer_name)
            y_pred = _predict_for_scorer(pipeline, scorer, df)
            raw = float(scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs))
            value = to_natural_sign(name, scorer_name, [raw])[0]
            ci_low_raw, ci_high_raw = _bootstrap_ci_from_predictions(scorer, y_true, y_pred)
            ci_low, ci_high = sorted(
                to_natural_sign(name, scorer_name, [ci_low_raw, ci_high_raw])
            )
            metrics.append(
                {
                    "name": name,
                    "value": round(value, 6),
                    "ci_low": round(ci_low, 6),
                    "ci_high": round(ci_high, 6),
                    "n": n_holdout,
                }
            )
        except Exception as exc:  # some scorers need proba/other shapes
            _log.info("holdout_metric_skipped", metric=name, error=str(exc))
    return metrics


def _headline(inputs: ReportInputs, holdout_metrics: list[dict[str, Any]]) -> list[dict]:
    primary = inputs.problem.primary_metric
    cards: list[dict] = []
    hp = next((m for m in holdout_metrics if m["name"] == primary), None)
    if hp is not None:
        cards.append({"value": f"{hp['value']:.3f}", "label": f"{primary} (holdout)"})
    for m in inputs.evaluation.cv_metrics:
        cards.append({"value": f"{m.mean:.3f}", "label": f"{m.name} (CV mean)"})
    return cards[:4]


def _render(template_str: str, **context: Any) -> str:
    from jinja2 import Environment

    env = Environment(autoescape=True)
    return env.from_string(template_str).render(css=base_css(), **context)


def build_final_report(
    store: GuardedArtifactStore, inputs: ReportInputs, *, run_id: str
) -> FinalReport:
    """Render + store both reports and return a FinalReport artifact."""
    holdout_metrics = _holdout_metrics(store, inputs)
    headline_cards = _headline(inputs, holdout_metrics)
    generated_at = utcnow().isoformat(timespec="seconds")

    safety_checks = [{"name": c.name, "passed": c.passed} for c in inputs.evaluation.leakage_checks]
    # Re-check the guard's LIVE state (the report itself is the only sanctioned
    # holdout read; anything else touching it before now would show here).
    safety_checks.append(
        {
            "name": "holdout untouched during development",
            "passed": inputs.evaluation.holdout_untouched and store.holdout_untouched(),
        }
    )

    business_figs = []
    technical_figs = []
    top_features: list[dict] = []
    figure_blobs: list[str] = []
    if inputs.explainability is not None:
        from markupsafe import Markup

        top_features = [{"feature": fi.feature} for fi in inputs.explainability.importances[:5]]
        for fig in inputs.explainability.figures:
            figure_blobs.append(fig.blob_hash)
            # Mark the Plotly JSON safe ONLY for the <script> embed: autoescape
            # would turn its quotes into &#34; and a browser never HTML-decodes
            # <script> text, breaking the chart. The JSON is produced by us, not
            # user text, so this is safe; all other fields stay autoescaped.
            fig_json = Markup(load_figure_json(store, fig.blob_hash))
            technical_figs.append(fig_json)
            if fig.kind in ("permutation_importance", "shap_summary"):
                business_figs.append(fig_json)

    verdict = inputs.critique.verdict.value
    verdict_label = verdict.replace("_", " ")

    business_html = _render(
        business_template(),
        title=f"Model report: {inputs.problem.target_column}",
        run_id=run_id,
        generated_at=generated_at,
        goal=inputs.goal_text,
        verdict=verdict,
        verdict_label=verdict_label,
        headline_sentence=_headline_sentence(inputs, holdout_metrics),
        headline_metrics=headline_cards,
        insight_findings=list(inputs.insights.findings) if inputs.insights else [],
        insight_caveats=list(inputs.insights.caveats) if inputs.insights else [],
        top_features=top_features,
        business_figures=business_figs,
        safety_checks=safety_checks,
    )
    technical_html = _render(
        technical_template(),
        title=f"Technical report: {inputs.problem.target_column}",
        run_id=run_id,
        generated_at=generated_at,
        problem={
            "task_type": inputs.problem.task_type.value,
            "target_column": inputs.problem.target_column,
            "primary_metric": inputs.problem.primary_metric,
        },
        best_template=inputs.best_template_id,
        best_experiment=inputs.best_experiment_id,
        cv_metrics=[
            {
                "name": m.name,
                "mean": f"{m.mean:.6f}",
                "std": f"{m.std:.6f}",
                "folds": ", ".join(f"{v:.4f}" for v in m.fold_values),
            }
            for m in inputs.evaluation.cv_metrics
        ],
        holdout_metrics=holdout_metrics,
        baseline_deltas=[
            {
                "baseline": d.baseline_name,
                "metric": d.metric,
                "baseline_value": f"{d.baseline_value:.4f}",
                "model_value": f"{d.model_value:.4f}",
                "delta": f"{d.delta:+.4f}",
                "significant": d.significant,
                "significance_method": d.significance_method,
                "p_value": None if d.p_value is None else round(d.p_value, 4),
            }
            for d in inputs.evaluation.baseline_deltas
        ],
        leakage_checks=[
            {"name": c.name, "passed": c.passed, "details": c.details}
            for c in inputs.evaluation.leakage_checks
        ],
        calibration=(
            {
                "expected_calibration_error": (
                    inputs.evaluation.calibration.expected_calibration_error
                ),
                "brier_score": inputs.evaluation.calibration.brier_score,
                "method": inputs.evaluation.calibration.method,
            }
            if inputs.evaluation.calibration.expected_calibration_error is not None
            else None
        ),
        decision_threshold=(
            {
                "threshold": inputs.evaluation.decision_threshold.threshold,
                "expected_cost": inputs.evaluation.decision_threshold.expected_cost,
                "cost_at_default_threshold": (
                    inputs.evaluation.decision_threshold.cost_at_default_threshold
                ),
                "false_positive_cost": inputs.evaluation.decision_threshold.false_positive_cost,
                "false_negative_cost": inputs.evaluation.decision_threshold.false_negative_cost,
                "false_positives": inputs.evaluation.decision_threshold.false_positives,
                "false_negatives": inputs.evaluation.decision_threshold.false_negatives,
                "true_positives": inputs.evaluation.decision_threshold.true_positives,
                "true_negatives": inputs.evaluation.decision_threshold.true_negatives,
            }
            if inputs.evaluation.decision_threshold is not None
            else None
        ),
        verdict=verdict,
        verdict_label=verdict_label,
        findings=[
            {"severity": f.severity.value, "category": f.category, "evidence": f.evidence}
            for f in inputs.critique.findings
        ],
        technical_figures=technical_figs,
        iterations=[
            {
                "index": it["index"],
                "verdict": it["verdict"],
                "best": it["best"],
                "new": ", ".join(it.get("new", [])),
            }
            for it in inputs.iterations
        ],
        repro={
            "seed": inputs.model.seed,
            "env_hash": inputs.model.env_hash,
            "train_data": inputs.model.train_data_ref.artifact_id,
        },
    )

    business_blob = store.put_blob(business_html.encode("utf-8"))
    technical_blob = store.put_blob(technical_html.encode("utf-8"))
    headline_dict = {m["name"]: float(m["value"]) for m in holdout_metrics}

    report = new_artifact(
        FinalReport,
        producer_agent="report-generator",
        run_id=run_id,
        parent_ids=(
            inputs.evaluation.meta.artifact_id,
            inputs.critique.meta.artifact_id,
            inputs.model.meta.artifact_id,
        ),
        business_report_blob=business_blob,
        technical_report_blob=technical_blob,
        figure_blobs=tuple(figure_blobs),
        headline_metrics=headline_dict,
    )
    store.put(report)
    _log.info(
        "final_report_built",
        business_blob=business_blob,
        technical_blob=technical_blob,
        holdout_metrics=len(holdout_metrics),
    )
    return report


def _headline_sentence(inputs: ReportInputs, holdout_metrics: list[dict[str, Any]]) -> str:
    primary = inputs.problem.primary_metric
    hp = next((m for m in holdout_metrics if m["name"] == primary), None)
    delta = next((d for d in inputs.evaluation.baseline_deltas if d.metric == primary), None)
    if hp is None or delta is None:
        return "See the metrics below for measured performance."
    # "significantly outperforms" is only used when a real statistical test
    # (the paired Nadeau-Bengio corrected t-test) actually ran and passed —
    # the heuristic fallback path says "improves on" even when it judged the
    # gap significant, since it is a documented rule of thumb, not a test.
    if delta.significant and delta.significance_method == "nadeau_bengio_paired_t":
        verb = "significantly outperforms"
        p_clause = f" (p={delta.p_value:.4f})" if delta.p_value is not None else ""
    else:
        verb = "improves on"
        p_clause = ""
    return (
        f"On held-out data the model scores {hp['value']:.3f} on {primary}, and it "
        f"{verb} a naive baseline ({delta.baseline_value:.3f} → {delta.model_value:.3f}){p_clause}."
    )
