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
    DataProfile,
    DeploymentBundle,
    ProblemSpec,
    TrainedModel,
    new_artifact,
)
from mlteam.orchestration.gates import ApprovalGate, GateName, require_approval
from mlteam.serving.app_template import render_app
from mlteam.serving.dockerfile import render_dockerfile, render_requirements
from mlteam.serving.schema import build_inference_schema_spec, load_spec, serialize_spec
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)


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
) -> DeploymentBundle:
    """Assemble and store a DeploymentBundle after Gate 3 sign-off."""
    schema_spec = build_inference_schema_spec(
        profile, target_column=problem.target_column, feature_names=feature_names
    )
    schema_blob = store.put_blob(serialize_spec(schema_spec))
    app_source = render_app(problem.task_type).encode("utf-8")
    app_blob = store.put_blob(app_source)
    dockerfile_blob = store.put_blob(render_dockerfile().encode("utf-8"))
    drift_blob = store.put_blob(_drift_baseline_parquet(store, train_dataset_id))

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
