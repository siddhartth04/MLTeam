"""Deterministic tabular pipeline (Phase 1): no LLM anywhere.

ingest → profile(raw) → split (freeze holdout) → profile(train) →
clean(train) → baselines (dummy + LightGBM [+ TabPFN]) → EvaluationReport.

Cleaning decisions are derived from the TRAIN profile only — the raw profile
exists for the record and split rules, but nothing learned from holdout rows
influences modeling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from mlteam.agents.cleaner import execute_cleaning, propose_cleaning_ops
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.splitter import plan_split
from mlteam.agents.trainer import TrainResult, compute_env_hash, train_experiment
from mlteam.contracts import (
    BudgetEstimate,
    EvaluationReport,
    ExperimentPlan,
    ExperimentSpec,
    TaskType,
    new_artifact,
)
from mlteam.evaluation.metrics import better_than
from mlteam.evaluation.report import build_evaluation_report
from mlteam.store.guard import GuardedArtifactStore

_log = structlog.get_logger(__name__)

_DUMMY = "dummy_baseline"


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def default_experiment_specs(
    include_tabpfn: bool = True,
    lgbm_config: dict | None = None,
) -> tuple[ExperimentSpec, ...]:
    """Phase 1 baseline set. Dummy always first — the plan validator insists."""
    specs = [
        ExperimentSpec(
            experiment_id="exp-dummy",
            template_id=_DUMMY,
            budget=BudgetEstimate(wall_time_seconds=2),
            is_baseline=True,
        ),
        ExperimentSpec(
            experiment_id="exp-lightgbm",
            template_id="lightgbm_default",
            config=dict(lgbm_config or {}),
            budget=BudgetEstimate(wall_time_seconds=30),
            is_baseline=True,
        ),
    ]
    if include_tabpfn:
        specs.append(
            ExperimentSpec(
                experiment_id="exp-tabpfn",
                template_id="tabpfn_default",
                budget=BudgetEstimate(wall_time_seconds=180),
                is_baseline=True,
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    raw_dataset_id: str
    raw_profile_id: str
    split_plan_id: str
    train_profile_id: str
    cleaned_dataset_id: str
    cleaning_manifest_id: str
    experiment_plan_id: str
    evaluation_report: EvaluationReport
    best_template_id: str
    train_results: dict[str, TrainResult] = field(default_factory=dict)
    skipped_experiments: tuple[tuple[str, str], ...] = ()
    # Set when the feature-plan executor produced an engineered dataset;
    # training/explaining/iterating all use modeling_dataset_id.
    feature_dataset_id: str | None = None

    @property
    def modeling_dataset_id(self) -> str:
        return self.feature_dataset_id or self.cleaned_dataset_id


def _train_all(
    store: GuardedArtifactStore,
    dataset_id: str,
    specs: tuple[ExperimentSpec, ...],
    *,
    task_type: TaskType,
    run_id: str,
    seed: int,
    n_folds: int,
    env_hash: str,
    mlflow_enabled: bool,
) -> tuple[dict[str, TrainResult], tuple[tuple[str, str], ...]]:
    """Train every spec; non-dummy failures are skipped LOUDLY, not fatally.

    Results are keyed by experiment_id — several experiments may share a
    template (e.g. default and tuned LightGBM).
    """
    results: dict[str, TrainResult] = {}
    skipped: list[tuple[str, str]] = []
    for spec in specs:
        try:
            results[spec.experiment_id] = train_experiment(
                store,
                dataset_id,
                spec,
                task_type=task_type,
                run_id=run_id,
                seed=seed,
                n_folds=n_folds,
                env_hash=env_hash,
                mlflow_enabled=mlflow_enabled,
            )
        except Exception as exc:
            if spec.template_id == _DUMMY:
                raise  # without the dummy floor no evaluation is meaningful
            _log.warning(
                "experiment_skipped",
                template=spec.template_id,
                error=str(exc),
                hint="baseline skipped; remaining baselines still run",
            )
            skipped.append((spec.template_id, str(exc)))
    return results, tuple(skipped)


def _dummy_key(specs: tuple[ExperimentSpec, ...]) -> str:
    """experiment_id of the dummy-baseline spec (the sanity floor)."""
    for spec in specs:
        if spec.template_id == _DUMMY:
            return spec.experiment_id
    raise RuntimeError("experiment specs contain no dummy_baseline")


def _select_best(results: dict[str, TrainResult], primary_metric: str, dummy_key: str) -> str:
    candidates = {eid: r for eid, r in results.items() if eid != dummy_key}
    if not candidates:
        raise RuntimeError("no non-dummy experiment trained successfully; cannot produce a report")
    best_id, best = next(iter(candidates.items()))
    for experiment_id, result in candidates.items():
        if better_than(
            primary_metric,
            result.metrics[primary_metric].mean,
            best.metrics[primary_metric].mean,
        ):
            best_id, best = experiment_id, result
    return best_id


def run_tabular_pipeline(
    store: GuardedArtifactStore,
    data_path: Path,
    *,
    target_column: str,
    task_type: TaskType,
    primary_metric: str,
    run_id: str | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    include_tabpfn: bool = True,
    lgbm_config: dict | None = None,
    mlflow_enabled: bool = False,
    env_lock_path: Path | None = None,
    time_column: str | None = None,
) -> PipelineResult:
    """Run the full deterministic pipeline and return every artifact handle."""
    run_id = run_id or new_run_id()
    raw = ingest_file(store, data_path, target_column=target_column, run_id=run_id)
    modeling_id = raw.meta.artifact_id
    if task_type is TaskType.FORECASTING:
        from mlteam.agents.data_io import read_dataset
        from mlteam.agents.splitter import detect_time_column
        from mlteam.agents.time_features import engineer_time_features

        if time_column is None:
            time_column = detect_time_column(read_dataset(store, raw))
        featured = engineer_time_features(
            store,
            raw.meta.artifact_id,
            time_column=time_column,
            target_column=target_column,
            run_id=run_id,
        )
        modeling_id = featured.meta.artifact_id
    raw_profile = profile_dataset(store, modeling_id, run_id=run_id, seed=seed)
    split = plan_split(
        store,
        modeling_id,
        task_type=task_type,
        run_id=run_id,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_folds=n_folds,
        time_column=time_column,
    )
    train_profile = profile_dataset(store, split.train_ref.artifact_id, run_id=run_id, seed=seed)
    cleaned, manifest = execute_cleaning(
        store,
        split.train_ref.artifact_id,
        propose_cleaning_ops(train_profile),
        run_id=run_id,
    )

    specs = default_experiment_specs(include_tabpfn=include_tabpfn, lgbm_config=lgbm_config)
    plan = new_artifact(
        ExperimentPlan,
        producer_agent="experiment-planner",
        run_id=run_id,
        parent_ids=(cleaned.meta.artifact_id,),
        experiments=specs,
        rationale="Phase 1 fixed baseline set: dummy floor, LightGBM default, TabPFN",
    )
    store.put(plan)

    results, skipped = _train_all(
        store,
        cleaned.meta.artifact_id,
        specs,
        task_type=task_type,
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=compute_env_hash(env_lock_path),
        mlflow_enabled=mlflow_enabled,
    )
    dummy_key = _dummy_key(specs)
    best_key = _select_best(results, primary_metric, dummy_key)
    report = build_evaluation_report(
        store,
        run_id=run_id,
        model_result=results[best_key],
        dummy_result=results[dummy_key],
        profile=train_profile,
        primary_metric=primary_metric,
    )
    return PipelineResult(
        run_id=run_id,
        raw_dataset_id=raw.meta.artifact_id,
        raw_profile_id=raw_profile.meta.artifact_id,
        split_plan_id=split.meta.artifact_id,
        train_profile_id=train_profile.meta.artifact_id,
        cleaned_dataset_id=cleaned.meta.artifact_id,
        cleaning_manifest_id=manifest.meta.artifact_id,
        experiment_plan_id=plan.meta.artifact_id,
        evaluation_report=report,
        best_template_id=results[best_key].model.template_id,
        train_results=results,
        skipped_experiments=skipped,
    )
