"""Agentic pipeline (Phase 3): LLM agents propose, deterministic code executes.

goal text -> Requirement Understanding -> GATE 1 -> ingest/profile/split ->
LLM cleaning ops -> LLM feature plan -> LLM experiment plan -> GATE 2 ->
train -> evaluate -> Critic. The iteration loop lands in Phase 4; Gate 3
(deployment) lands in Phase 5.

LLM failures degrade LOUDLY to the deterministic Phase 1 path — the pipeline
never dies because a proposal could not be validated, and every fallback is
logged and traced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

from mlteam.agents.cleaner import (
    execute_cleaning,
    propose_cleaning_ops,
    propose_cleaning_ops_llm,
)
from mlteam.agents.critic import critique_evaluation
from mlteam.agents.experiment_planner import plan_experiments
from mlteam.agents.feature_engineer import propose_features
from mlteam.agents.feature_executor import execute_feature_plan
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.requirement import understand_requirements
from mlteam.agents.splitter import detect_time_column, plan_split
from mlteam.agents.time_features import engineer_time_features
from mlteam.agents.trainer import compute_env_hash
from mlteam.contracts import CritiqueReport, ProblemSpec, TaskType
from mlteam.evaluation.report import build_evaluation_report
from mlteam.llm.client import LLMClient, LLMValidationError
from mlteam.orchestration.gates import ApprovalGate, GateName, require_approval
from mlteam.orchestration.pipeline import (
    PipelineResult,
    _dummy_key,
    _select_best,
    _train_all,
    default_experiment_specs,
    new_run_id,
)
from mlteam.store.guard import GuardedArtifactStore

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AgenticResult:
    run_id: str
    problem_spec: ProblemSpec
    pipeline: PipelineResult
    critique: CritiqueReport
    feature_plan_id: str | None
    llm_cost_usd: float


def _read_columns(data_path: Path) -> list[str]:
    if data_path.suffix.lower() == ".parquet":
        return list(pl.read_parquet_schema(data_path).keys())
    # Only column NAMES are needed here — read as all-string so no row's
    # value (e.g. a "> 10" bucket label in an otherwise-numeric column) can
    # ever fail dtype inference before ingestion gets a chance to normalize it.
    return pl.read_csv(data_path, n_rows=1, infer_schema=False).columns


def run_agentic_pipeline(
    store: GuardedArtifactStore,
    data_path: Path,
    *,
    goal_text: str,
    client: LLMClient,
    gate: ApprovalGate,
    run_id: str | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    include_tabpfn: bool = False,
    mlflow_enabled: bool = False,
) -> AgenticResult:
    """Run the LLM-driven pipeline end to end (no iteration loop yet)."""
    run_id = run_id or new_run_id()

    # -- Requirement Understanding + GATE 1 ---------------------------------
    columns = _read_columns(data_path)
    spec = understand_requirements(
        client, store, goal_text=goal_text, dataset_columns=columns, run_id=run_id
    )
    require_approval(
        gate,
        GateName.PROBLEM_SPEC,
        {
            "task_type": spec.task_type.value,
            "target_column": spec.target_column,
            "primary_metric": spec.primary_metric,
            "cost_asymmetry": spec.cost_asymmetry.model_dump(),
            "assumptions": list(spec.assumptions),
        },
    )
    # Confirmation mints a NEW artifact (content changed -> new content hash),
    # with the unconfirmed draft as its lineage parent.
    from mlteam.contracts import new_artifact

    spec = new_artifact(
        ProblemSpec,
        producer_agent="gate1-confirmation",
        run_id=run_id,
        parent_ids=(spec.meta.artifact_id,),
        task_type=spec.task_type,
        target_column=spec.target_column,
        prediction_horizon=spec.prediction_horizon,
        time_column=spec.time_column,
        primary_metric=spec.primary_metric,
        secondary_metrics=spec.secondary_metrics,
        business_constraints=spec.business_constraints,
        cost_asymmetry=spec.cost_asymmetry,
        assumptions=spec.assumptions,
        human_confirmed=True,
    )
    store.put(spec)

    # -- Deterministic substrate: ingest, [time features], profile, split -----
    raw = ingest_file(store, data_path, target_column=spec.target_column, run_id=run_id)
    modeling_id = raw.meta.artifact_id
    time_column = spec.time_column
    if spec.task_type is TaskType.FORECASTING:
        # Lag/rolling/calendar features are computed pre-split: every value is
        # backward-looking, so holdout rows carry real history — no leakage.
        from mlteam.agents.data_io import read_dataset

        if time_column is None:
            time_column = detect_time_column(read_dataset(store, raw))
        featured = engineer_time_features(
            store,
            raw.meta.artifact_id,
            time_column=time_column,
            target_column=spec.target_column,
            run_id=run_id,
        )
        modeling_id = featured.meta.artifact_id
    raw_profile = profile_dataset(store, modeling_id, run_id=run_id, seed=seed)
    split = plan_split(
        store,
        modeling_id,
        task_type=spec.task_type,
        run_id=run_id,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_folds=n_folds,
        time_column=time_column,
    )
    train_profile = profile_dataset(store, split.train_ref.artifact_id, run_id=run_id, seed=seed)

    # -- LLM cleaning proposal (deterministic fallback, loudly) ---------------
    try:
        ops = propose_cleaning_ops_llm(client, train_profile, target_column=spec.target_column)
    except LLMValidationError as exc:
        _log.warning("cleaning_proposer_fallback", error=str(exc))
        ops = propose_cleaning_ops(train_profile)
    cleaned, manifest = execute_cleaning(store, split.train_ref.artifact_id, ops, run_id=run_id)

    # -- LLM feature plan + deterministic execution -----------------------------
    feature_plan_id: str | None = None
    feature_dataset_id: str | None = None
    modeling_dataset = cleaned
    try:
        feature_plan = propose_features(
            client, store, profile=train_profile, problem=spec, run_id=run_id
        )
        feature_plan_id = feature_plan.meta.artifact_id
        featured_ds, _feature_set = execute_feature_plan(
            store, cleaned.meta.artifact_id, feature_plan, run_id=run_id
        )
        if featured_ds is not None:
            modeling_dataset = featured_ds
            feature_dataset_id = featured_ds.meta.artifact_id
    except LLMValidationError as exc:
        _log.warning("feature_proposer_skipped", error=str(exc))

    # -- LLM experiment plan + GATE 2 ----------------------------------------
    try:
        plan = plan_experiments(
            client,
            store,
            profile=train_profile,
            problem=spec,
            n_rows=modeling_dataset.n_rows,
            run_id=run_id,
        )
        specs = plan.experiments
        plan_rationale = plan.rationale
        plan_id = plan.meta.artifact_id
    except LLMValidationError as exc:
        _log.warning("experiment_planner_fallback", error=str(exc))
        specs = default_experiment_specs(include_tabpfn=include_tabpfn)
        plan_rationale = "deterministic fallback plan (LLM planner failed validation)"
        plan_id = ""
    total = sum(s.budget.wall_time_seconds for s in specs)
    require_approval(
        gate,
        GateName.TRAINING_PLAN,
        {
            "experiments": [
                {
                    "experiment_id": s.experiment_id,
                    "template_id": s.template_id,
                    "config": s.config,
                    "estimated_seconds": s.budget.wall_time_seconds,
                    "is_baseline": s.is_baseline,
                }
                for s in specs
            ],
            "estimated_total_seconds": total,
            "llm_cost_so_far_usd": client.total_cost_usd,
            "rationale": plan_rationale,
        },
    )

    # -- Train, evaluate, critique -------------------------------------------
    results, skipped = _train_all(
        store,
        modeling_dataset.meta.artifact_id,
        tuple(specs),
        task_type=spec.task_type,
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=compute_env_hash(),
        mlflow_enabled=mlflow_enabled,
    )
    dummy_key = _dummy_key(tuple(specs))
    best_key = _select_best(results, spec.primary_metric, dummy_key)
    report = build_evaluation_report(
        store,
        run_id=run_id,
        model_result=results[best_key],
        dummy_result=results[dummy_key],
        profile=train_profile,
        primary_metric=spec.primary_metric,
    )
    critique = critique_evaluation(
        client, store, report=report, primary_metric=spec.primary_metric, run_id=run_id
    )

    pipeline = PipelineResult(
        run_id=run_id,
        raw_dataset_id=raw.meta.artifact_id,
        raw_profile_id=raw_profile.meta.artifact_id,
        split_plan_id=split.meta.artifact_id,
        train_profile_id=train_profile.meta.artifact_id,
        cleaned_dataset_id=cleaned.meta.artifact_id,
        cleaning_manifest_id=manifest.meta.artifact_id,
        experiment_plan_id=plan_id,
        feature_dataset_id=feature_dataset_id,
        evaluation_report=report,
        best_template_id=results[best_key].model.template_id,
        train_results=results,
        skipped_experiments=skipped,
    )
    return AgenticResult(
        run_id=run_id,
        problem_spec=spec,
        pipeline=pipeline,
        critique=critique,
        feature_plan_id=feature_plan_id,
        llm_cost_usd=client.total_cost_usd,
    )
