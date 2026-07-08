"""Training: template + config → CV metrics + serialized full pipeline.

Everything is seeded and single-threaded where it affects determinism.
Preprocessing lives inside the pipeline, so it is re-fit within every CV
fold — never on full data.
"""

from __future__ import annotations

import io
from pathlib import Path

import structlog

from mlteam.agents.data_io import read_dataset
from mlteam.contracts import (
    ExperimentSpec,
    MetricSummary,
    TaskType,
    file_content_hash,
    new_artifact,
)
from mlteam.contracts.model import TrainedModel
from mlteam.evaluation.metrics import scoring_for, summarize_folds, to_natural_sign
from mlteam.evaluation.results import TrainResult
from mlteam.store.interface import ArtifactStore
from mlteam.templates.registry import ColumnRoles, get_template, wrap_target_transform

_log = structlog.get_logger(__name__)

__all__ = ["TrainResult", "compute_env_hash", "train_experiment"]


def compute_env_hash(lock_path: Path | None = None) -> str:
    """Hash of the locked dependency environment, for reproducibility records."""
    path = lock_path if lock_path is not None else Path("uv.lock")
    if path.exists():
        return file_content_hash(path)
    return "unlocked-environment"


def _temporal_to_epoch(df: object) -> object:
    """Cast Date/Datetime columns to epoch seconds (Int64).

    A fixed, stateless transform (no statistics learned), so it is safe
    outside CV folds. Proper datetime decomposition arrives with the Phase 3
    feature-engineering ops.
    """
    import polars as pl

    assert isinstance(df, pl.DataFrame)
    temporal = [c for c, dt in df.schema.items() if dt in (pl.Date, pl.Datetime)]
    if not temporal:
        return df
    _log.info("temporal_columns_cast_to_epoch", columns=temporal)
    return df.with_columns(
        [pl.col(c).cast(pl.Datetime).dt.epoch(time_unit="s").alias(c) for c in temporal]
    )


def _column_roles(df: object, target_column: str) -> ColumnRoles:
    import polars as pl

    assert isinstance(df, pl.DataFrame)
    numeric, categorical = [], []
    for col in df.columns:
        if col == target_column:
            continue
        (numeric if df[col].dtype.is_numeric() else categorical).append(col)
    return ColumnRoles(numeric=tuple(numeric), categorical=tuple(categorical))


def _make_cv(task_type: TaskType, n_folds: int, seed: int) -> object:
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if task_type is TaskType.FORECASTING:
        # Rows arrive time-sorted from the temporal splitter; expanding-window
        # CV means every fold trains strictly on the past. Never shuffle.
        return TimeSeriesSplit(n_splits=n_folds)
    if task_type is TaskType.REGRESSION:
        return KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def _cross_validate(
    pipeline: object,
    features: object,
    target: object,
    task_type: TaskType,
    n_folds: int,
    seed: int,
) -> dict[str, MetricSummary]:
    from sklearn.model_selection import cross_validate

    scoring = scoring_for(task_type)
    raw = cross_validate(
        pipeline,
        features,
        target,
        cv=_make_cv(task_type, n_folds, seed),
        scoring={name: scorer for name, scorer in scoring.items()},
        n_jobs=1,
        error_score="raise",
    )
    summaries = {}
    for name, scorer in scoring.items():
        values = to_natural_sign(name, scorer, [float(v) for v in raw[f"test_{name}"]])
        summaries[name] = summarize_folds(name, values)
    return summaries


def train_experiment(
    store: ArtifactStore,
    dataset_id: str,
    spec: ExperimentSpec,
    *,
    task_type: TaskType,
    run_id: str,
    seed: int,
    n_folds: int,
    env_hash: str,
    mlflow_enabled: bool = False,
) -> TrainResult:
    """Cross-validate a template, fit it on all train data, store the model."""
    import joblib

    dataset = store.get(dataset_id)
    if dataset.target_column is None:
        raise ValueError("training dataset has no declared target column")
    df = _temporal_to_epoch(read_dataset(store, dataset))
    roles = _column_roles(df, dataset.target_column)

    frame = df.to_pandas()
    target = frame.pop(dataset.target_column)

    template = get_template(spec.template_id)
    if not template.supports(task_type):
        raise ValueError(f"template '{spec.template_id}' does not support {task_type}")
    if spec.target_transform is not None and (target < 0).any():
        raise ValueError(
            f"target_transform='{spec.target_transform}' requires a non-negative "
            f"target, but experiment '{spec.experiment_id}' has "
            f"{int((target < 0).sum())} negative value(s)"
        )
    pipeline = wrap_target_transform(
        template.build(task_type, roles, spec.config, seed), spec.target_transform, task_type
    )

    metrics = _cross_validate(pipeline, frame, target, task_type, n_folds, seed)
    pipeline.fit(frame, target)

    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    pipeline_blob = store.put_blob(buffer.getvalue())

    mlflow_run_id = None
    if mlflow_enabled:
        mlflow_run_id = _log_to_mlflow(spec, dataset.blob_hash, metrics, seed)

    model = new_artifact(
        TrainedModel,
        producer_agent="trainer",
        run_id=run_id,
        parent_ids=(dataset_id,),
        experiment_id=spec.experiment_id,
        template_id=spec.template_id,
        pipeline_blob=pipeline_blob,
        config=dict(spec.config),
        seed=seed,
        env_hash=env_hash,
        train_data_ref=dataset.ref(),
        mlflow_run_id=mlflow_run_id,
        target_transform=spec.target_transform,
    )
    store.put(model)
    _log.info(
        "experiment_trained",
        experiment_id=spec.experiment_id,
        template=spec.template_id,
        metrics={name: summary.mean for name, summary in metrics.items()},
    )
    return TrainResult(model=model, metrics=metrics)


def _log_to_mlflow(
    spec: ExperimentSpec,
    data_hash: str,
    metrics: dict[str, MetricSummary],
    seed: int,
) -> str:
    import mlflow

    with mlflow.start_run(run_name=spec.experiment_id) as active:
        mlflow.set_tag("mlteam.template_id", spec.template_id)
        mlflow.set_tag("mlteam.data_hash", data_hash)
        mlflow.log_param("seed", seed)
        for key, value in spec.config.items():
            mlflow.log_param(key, value)
        for name, summary in metrics.items():
            mlflow.log_metric(f"cv_{name}_mean", summary.mean)
            mlflow.log_metric(f"cv_{name}_std", summary.std)
        return active.info.run_id
