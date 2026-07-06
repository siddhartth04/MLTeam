"""Deployment agent: assemble a serving bundle behind Human Gate 3.

Bundle = full serialized pipeline + the inference Pandera schema (same one
that killed train/serve skew) + generated FastAPI app + Dockerfile + an
Evidently drift baseline (the training feature distribution). Gate 3 is the
pre-deployment sign-off; nothing is materialized to disk until it passes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import structlog

from mlteam.agents.data_io import read_dataset
from mlteam.contracts import (
    CritiqueReport,
    DataProfile,
    DeploymentBundle,
    EvaluationReport,
    ExperimentOutcome,
    ExperimentState,
    ExplainabilityReport,
    ProblemSpec,
    TrainedModel,
    new_artifact,
)
from mlteam.evaluation.metrics import is_higher_better
from mlteam.orchestration.gates import ApprovalGate, GateName, require_approval
from mlteam.serving.app_template import render_app
from mlteam.serving.dockerfile import render_dockerfile, render_requirements
from mlteam.serving.schema import build_inference_schema_spec, load_spec, serialize_spec
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_TOP_FEATURES = 5


def _metric_mean(outcome: ExperimentOutcome, metric: str) -> float | None:
    return next((m.mean for m in outcome.cv_metrics if m.name == metric), None)


def _signed_improvement(metric: str, better: float, worse: float) -> float:
    """Positive whenever ``better`` actually is better, regardless of the
    metric's own direction (so a dossier reader never has to remember which
    metrics are higher-is-better)."""
    return (better - worse) if is_higher_better(metric) else (worse - better)


def build_deployment_dossier(
    *,
    model: TrainedModel,
    problem: ProblemSpec,
    train_dataset_id: str,
    split_plan_id: str | None,
    outcomes: tuple[ExperimentOutcome, ...] = (),
    evaluation: EvaluationReport | None = None,
    explainability: ExplainabilityReport | None = None,
    critique: CritiqueReport | None = None,
) -> dict[str, Any]:
    """Assemble the evidence a human needs to answer "why trust this model?"

    Pure and deterministic — no LLM, no store access beyond what's already
    been loaded by the caller — so it's independently testable and never
    adds latency or cost to Gate 3.
    """
    primary = problem.primary_metric
    completed = [o for o in outcomes if o.state == ExperimentState.COMPLETED]
    dummy = next((o for o in completed if o.template_id == "dummy_baseline"), None)
    candidates = [o for o in completed if o.template_id != "dummy_baseline"]
    ranked = sorted(
        (o for o in candidates if _metric_mean(o, primary) is not None),
        key=lambda o: _metric_mean(o, primary),
        reverse=is_higher_better(primary),
    )

    comparison = [
        {
            "experiment_id": o.experiment_id,
            "template_id": o.template_id,
            "state": o.state.value,
            "primary_metric": primary,
            "mean": _metric_mean(o, primary),
            "std": next((m.std for m in o.cv_metrics if m.name == primary), None),
            "is_winner": o.experiment_id == model.experiment_id,
        }
        for o in completed
    ] + [
        {
            "experiment_id": o.experiment_id,
            "template_id": o.template_id,
            "state": o.state.value,
            "primary_metric": primary,
            "mean": None,
            "std": None,
            "is_winner": False,
        }
        for o in outcomes
        if o.state != ExperimentState.COMPLETED
    ]

    winner_margin: dict[str, Any] | None = None
    if len(ranked) >= 1:
        winner = ranked[0]
        winner_value = _metric_mean(winner, primary)
        vs_runner_up = None
        if len(ranked) >= 2:
            runner_up = ranked[1]
            runner_up_value = _metric_mean(runner_up, primary)
            vs_runner_up = {
                "template_id": runner_up.template_id,
                "improvement": _signed_improvement(primary, winner_value, runner_up_value),
            }
        vs_dummy = None
        if dummy is not None:
            dummy_value = _metric_mean(dummy, primary)
            if dummy_value is not None:
                vs_dummy = _signed_improvement(primary, winner_value, dummy_value)
        winner_margin = {
            "vs_runner_up": vs_runner_up,
            "vs_dummy": vs_dummy,
        }

    top_features = []
    if explainability is not None:
        top_features = [
            {"feature": f.feature, "importance": f.importance}
            for f in sorted(explainability.importances, key=lambda f: f.importance, reverse=True)[
                :_TOP_FEATURES
            ]
        ]

    critique_summary = None
    if critique is not None:
        critique_summary = {
            "verdict": critique.verdict.value,
            "findings": [
                {"severity": f.severity.value, "category": f.category, "evidence": f.evidence}
                for f in critique.findings
            ],
        }

    return {
        "comparison": comparison,
        "winner_margin": winner_margin,
        "top_features": top_features,
        "critique": critique_summary,
        "lineage": {
            "model_id": model.meta.artifact_id,
            "dataset_id": train_dataset_id,
            "split_plan_id": split_plan_id,
        },
        # No "rollback strategy" here on purpose: there is no model registry
        # yet to roll back through, and fabricating one would be worse than
        # omitting it.
    }


def _drift_baseline_parquet(store: ArtifactStore, train_dataset_id: str) -> bytes:
    """Evidently reference dataset = the training features (as parquet)."""
    dataset = store.get(train_dataset_id)
    df = read_dataset(store, dataset)
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()


def build_deployment_bundle(
    store: ArtifactStore,
    *,
    model: TrainedModel,
    profile: DataProfile,
    problem: ProblemSpec,
    train_dataset_id: str,
    feature_names: list[str],
    gate: ApprovalGate,
    run_id: str,
    split_plan_id: str | None = None,
    outcomes: tuple[ExperimentOutcome, ...] = (),
    evaluation: EvaluationReport | None = None,
    explainability: ExplainabilityReport | None = None,
    critique: CritiqueReport | None = None,
) -> DeploymentBundle:
    """Assemble and store a DeploymentBundle after Gate 3 sign-off.

    ``outcomes``/``evaluation``/``explainability``/``critique`` are optional
    so a bare-minimum caller still gets a bundle; when given, Gate 3 shows
    the full comparison table, winner margin, top features, and critique
    verdict instead of just artifact hashes — "why trust deploying this?"
    answered with evidence, not bare ids.
    """
    schema_spec = build_inference_schema_spec(
        profile, target_column=problem.target_column, feature_names=feature_names
    )
    schema_blob = store.put_blob(serialize_spec(schema_spec))
    app_source = render_app(problem.task_type).encode("utf-8")
    app_blob = store.put_blob(app_source)
    dockerfile_blob = store.put_blob(render_dockerfile().encode("utf-8"))
    drift_blob = store.put_blob(_drift_baseline_parquet(store, train_dataset_id))

    dossier = build_deployment_dossier(
        model=model,
        problem=problem,
        train_dataset_id=train_dataset_id,
        split_plan_id=split_plan_id,
        outcomes=outcomes,
        evaluation=evaluation,
        explainability=explainability,
        critique=critique,
    )

    require_approval(
        gate,
        GateName.DEPLOYMENT,
        {
            "model_experiment_id": model.experiment_id,
            "template_id": model.template_id,
            "task_type": problem.task_type.value,
            "served_features": [c["name"] for c in schema_spec["columns"]],
            "artifacts": {
                "inference_schema_blob": schema_blob,
                "app_blob": app_blob,
                "dockerfile_blob": dockerfile_blob,
                "drift_baseline_blob": drift_blob,
            },
            "dossier": dossier,
        },
    )

    bundle = new_artifact(
        DeploymentBundle,
        producer_agent="deployer",
        run_id=run_id,
        parent_ids=(model.meta.artifact_id, profile.meta.artifact_id),
        model_ref=model.ref(),
        inference_schema_blob=schema_blob,
        app_blob=app_blob,
        dockerfile_blob=dockerfile_blob,
        drift_baseline_blob=drift_blob,
        human_signed_off=True,
    )
    store.put(bundle)
    _log.info("deployment_bundle_built", bundle=bundle.meta.artifact_id)
    return bundle


def materialize_bundle(
    store: ArtifactStore, bundle: DeploymentBundle, model: TrainedModel, target_dir: Path
) -> None:
    """Write a signed-off bundle to disk as a runnable serving directory."""
    if not bundle.human_signed_off:
        raise ValueError("refusing to materialize a bundle without Gate 3 sign-off")
    target_dir.mkdir(parents=True, exist_ok=True)
    model_dir = target_dir / "model"
    model_dir.mkdir(exist_ok=True)

    (target_dir / "serve.py").write_bytes(store.blob_path(bundle.app_blob).read_bytes())
    (target_dir / "Dockerfile").write_bytes(store.blob_path(bundle.dockerfile_blob).read_bytes())
    (target_dir / "requirements.txt").write_text(render_requirements(), encoding="utf-8")
    (model_dir / "pipeline.joblib").write_bytes(store.blob_path(model.pipeline_blob).read_bytes())
    (model_dir / "inference_schema.json").write_bytes(
        store.blob_path(bundle.inference_schema_blob).read_bytes()
    )
    _log.info("deployment_bundle_materialized", target=str(target_dir))


def evidently_drift_report(
    store: ArtifactStore, bundle: DeploymentBundle, current_frame: Any
) -> dict:
    """Run an Evidently data-drift report of current data vs the stored baseline.

    Returns the report as a dict; the reference is the deployment's baseline so
    drift is measured against exactly what the model was trained on.
    """
    import polars as pl
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if bundle.drift_baseline_blob is None:
        raise ValueError("bundle has no drift baseline")
    reference = pl.read_parquet(store.blob_path(bundle.drift_baseline_blob)).to_pandas()
    schema_spec = load_spec(store.blob_path(bundle.inference_schema_blob).read_bytes())
    served = [c["name"] for c in schema_spec["columns"]]
    ref = reference[[c for c in served if c in reference.columns]]
    cur = current_frame[[c for c in served if c in current_frame.columns]]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)
    return snapshot.dict()
