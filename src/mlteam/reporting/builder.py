"""Report generator: business + technical HTML with embedded Plotly figures.

This is the ONLY component that reads the frozen holdout — and only through the
store guard's ``final_reporting_access``. It computes true out-of-sample metrics
on the holdout, renders both reports, stores them as blobs, and returns a
FinalReport artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from mlteam.agents.trainer import _temporal_to_epoch
from mlteam.contracts import (
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
    iterations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _holdout_metrics(store: GuardedArtifactStore, inputs: ReportInputs) -> list[dict[str, Any]]:
    """Score the model on the frozen holdout via the guarded unlock path."""
    pipeline = load_pipeline(store, inputs.model)
    from sklearn.metrics import get_scorer

    with store.final_reporting_access("final HTML report") as unlocked:
        holdout: DatasetVersion = unlocked.get(inputs.split_plan.holdout_ref.artifact_id)
        import polars as pl

        frame = pl.read_parquet(unlocked.blob_path(holdout.blob_hash))

    # Re-derive the engineered feature columns on the holdout so the pipeline
    # sees the same schema it was trained on. Row-local ops only, so this is
    # leakage-free (see apply_feature_plan_to_frame).
    if inputs.feature_plan is not None:
        from mlteam.agents.feature_executor import apply_feature_plan_to_frame

        frame = apply_feature_plan_to_frame(frame, inputs.feature_plan)
    df = _temporal_to_epoch(frame).to_pandas()

    target = df.pop(inputs.problem.target_column)
    scoring = scoring_for(inputs.problem.task_type)
    metrics: list[dict[str, Any]] = []
    for name, scorer_name in scoring.items():
        try:
            raw = float(get_scorer(scorer_name)(pipeline, df, target))
            value = to_natural_sign(name, scorer_name, [raw])[0]
            metrics.append({"name": name, "value": round(value, 6)})
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
            }
            for d in inputs.evaluation.baseline_deltas
        ],
        leakage_checks=[
            {"name": c.name, "passed": c.passed, "details": c.details}
            for c in inputs.evaluation.leakage_checks
        ],
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
    verb = "significantly outperforms" if delta.significant else "improves on"
    return (
        f"On held-out data the model scores {hp['value']:.3f} on {primary}, and it "
        f"{verb} a naive baseline ({delta.baseline_value:.3f} → {delta.model_value:.3f})."
    )
