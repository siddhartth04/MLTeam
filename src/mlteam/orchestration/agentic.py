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
from mlteam.agents.experiment_planner import estimate_budget, plan_experiments
from mlteam.agents.feature_engineer import propose_features
from mlteam.agents.feature_executor import execute_feature_plan
from mlteam.agents.ingest import ingest_file
from mlteam.agents.preview import column_preview
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.requirement import understand_requirements
from mlteam.agents.splitter import detect_time_column, plan_split
from mlteam.agents.time_features import engineer_time_features
from mlteam.agents.trainer import compute_env_hash
from mlteam.agents.triage import triage_dataset
from mlteam.contracts import CritiqueReport, ExperimentSpec, ProblemSpec, TaskType
from mlteam.evaluation.report import build_evaluation_report
from mlteam.evaluation.results import TrainResult, list_experiment_outcomes
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
from mlteam.orchestration.progress import NULL_REPORTER, ProgressReporter
from mlteam.store.guard import GuardedArtifactStore

_log = structlog.get_logger(__name__)

_MIN_QUORUM_FRACTION = 0.5


def _check_experiment_quorum(
    specs: tuple[ExperimentSpec, ...],
    results: dict[str, TrainResult],
    skipped: tuple[tuple[str, str], ...],
    dummy_key: str,
) -> None:
    """Refuse to silently crown a survivor when the candidate field collapsed.

    ``planned`` excludes the dummy floor (it always trains or the run already
    aborted in ``_train_all``); ``completed`` is how many of those actually
    produced a model. Zero completions is fatal — there is nothing to select
    from and no evidence to evaluate. Below-half completions is not fatal
    (partial evidence can still be a valid, if degraded, result) but must be
    loud: the persisted FAILED outcomes make every failure visible on the
    leaderboard, and this warning makes the degradation visible in the logs
    a human is likely to read at Gate 2 / the critique stage.
    """
    planned = [s for s in specs if s.experiment_id != dummy_key]
    completed = [s for s in planned if s.experiment_id in results]
    if not planned:
        return
    if not completed:
        failures = "; ".join(f"{template}: {error}" for template, error in skipped)
        raise ValueError(
            f"all {len(planned)} candidate experiment(s) failed — nothing to "
            f"evaluate or deploy: {failures}"
        )
    if len(completed) < _MIN_QUORUM_FRACTION * len(planned):
        failed_ids = {s.experiment_id for s in planned} - {s.experiment_id for s in completed}
        _log.warning(
            "experiment_quorum_degraded",
            planned=len(planned),
            completed=len(completed),
            failed=sorted(failed_ids),
            hint="fewer than half the planned experiments completed; check the "
            "leaderboard for FAILED outcomes before trusting the winner",
        )


# Cheap probes the planner reads BEFORE choosing the real candidate slate.
# dummy_baseline and lightgbm_default reuse straight into the Gate-2 plan (via
# _train_all's reuse-by-identical-key check) whenever the plan keeps their
# config as the vetted default, so screening is close to free in wall time.
_SCREENING_TEMPLATES = ("dummy_baseline", "linear_default", "lightgbm_default")


def _screening_experiment_id(template: str) -> str:
    return f"screen-{template.replace('_default', '').replace('_baseline', '')}"


def screening_experiment_ids() -> tuple[str, ...]:
    """The fixed experiment_ids the screening round always plans — used both
    to run screening and, independently, to audit that it actually completed."""
    return tuple(_screening_experiment_id(t) for t in _SCREENING_TEMPLATES)


def _run_screening(
    store: GuardedArtifactStore,
    dataset_id: str,
    *,
    task_type: TaskType,
    primary_metric: str,
    n_rows: int,
    run_id: str,
    seed: int,
    n_folds: int,
    env_hash: str,
    mlflow_enabled: bool,
) -> list[dict[str, object]]:
    """Train the cheap probe trio and return their results as plain dicts for
    the planner payload. Never raises for a non-dummy probe failure — a
    linear-model failure (e.g. all-categorical data) is itself informative
    and simply produces one fewer screening result, not a broken run."""
    specs = tuple(
        ExperimentSpec(
            experiment_id=_screening_experiment_id(template),
            template_id=template,
            budget=estimate_budget(template, n_rows),
            is_baseline=(template == "dummy_baseline"),
        )
        for template in _SCREENING_TEMPLATES
    )
    _results, _skipped = _train_all(
        store,
        dataset_id,
        specs,
        task_type=task_type,
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=env_hash,
        mlflow_enabled=mlflow_enabled,
    )
    screen_ids = {s.experiment_id for s in specs}
    outcomes = [
        o for o in list_experiment_outcomes(store, run_id) if o.experiment_id in screen_ids
    ]
    return [
        {
            "template": o.template_id,
            "primary_metric_mean": next(
                (round(m.mean, 6) for m in o.cv_metrics if m.name == primary_metric), None
            ),
            "primary_metric_std": next(
                (round(m.std, 6) for m in o.cv_metrics if m.name == primary_metric), None
            ),
            "wall_time_seconds": round(o.wall_time_seconds, 2),
            "state": o.state.value,
        }
        for o in outcomes
    ]


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
    reporter: ProgressReporter = NULL_REPORTER,
) -> AgenticResult:
    """Run the LLM-driven pipeline end to end (no iteration loop yet)."""
    run_id = run_id or new_run_id()

    # -- Requirement Understanding + GATE 1 ---------------------------------
    reporter.stage("requirement", "active")
    columns = _read_columns(data_path)
    previews = column_preview(data_path)
    spec = understand_requirements(
        client,
        store,
        goal_text=goal_text,
        dataset_columns=columns,
        run_id=run_id,
        column_profile=previews,
    )
    reporter.stage("requirement", "done", detail=f"task: {spec.task_type.value}")
    reporter.log(f"Understood goal as {spec.task_type.value} on '{spec.target_column}'")
    require_approval(
        gate,
        GateName.PROBLEM_SPEC,
        {
            "task_type": spec.task_type.value,
            "target_column": spec.target_column,
            "primary_metric": spec.primary_metric,
            "cost_asymmetry": spec.cost_asymmetry.model_dump(),
            "assumptions": list(spec.assumptions),
            "reasoning": spec.reasoning.model_dump() if spec.reasoning else None,
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

    # -- Deterministic substrate: ingest, triage, [time features], split ------
    reporter.stage("ingest", "active")
    raw = ingest_file(
        store,
        data_path,
        target_column=spec.target_column,
        run_id=run_id,
        allow_missing_target=True,  # triage drops unlabeled rows with a report
    )
    reporter.stage("ingest", "done", detail=f"{raw.n_rows} rows")
    reporter.log(f"Ingested {raw.n_rows} rows, {raw.n_cols} columns", stage="ingest")

    triaged, triage_report = triage_dataset(
        store,
        raw.meta.artifact_id,
        target_column=spec.target_column,
        task_type=spec.task_type,
        run_id=run_id,
    )
    reporter.log(
        f"Triage: {triage_report.rows_before} -> {triage_report.rows_after} rows "
        f"({len(triage_report.remedies)} remedie(s))",
        stage="ingest",
    )
    modeling_id = triaged.meta.artifact_id
    time_column = spec.time_column
    if spec.task_type is TaskType.FORECASTING:
        # Lag/rolling/calendar features are computed pre-split: every value is
        # backward-looking, so holdout rows carry real history — no leakage.
        from mlteam.agents.data_io import read_dataset

        if time_column is None:
            time_column = detect_time_column(read_dataset(store, triaged))
        featured = engineer_time_features(
            store,
            triaged.meta.artifact_id,
            time_column=time_column,
            target_column=spec.target_column,
            run_id=run_id,
        )
        modeling_id = featured.meta.artifact_id

    reporter.stage("profile", "active")
    raw_profile = profile_dataset(store, modeling_id, run_id=run_id, seed=seed)
    reporter.stage("profile", "done", detail=f"{len(raw_profile.columns)} columns")
    reporter.log(f"Profiled {len(raw_profile.columns)} column(s)", stage="profile")

    reporter.stage("split", "active")
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
    reporter.stage("split", "done", detail=f"{n_folds} folds")
    reporter.log(f"Split into train/holdout with {n_folds} CV folds", stage="split")
    train_profile = profile_dataset(store, split.train_ref.artifact_id, run_id=run_id, seed=seed)

    # -- LLM cleaning proposal (deterministic fallback, loudly) ---------------
    reporter.stage("clean", "active")
    try:
        ops = propose_cleaning_ops_llm(client, train_profile, target_column=spec.target_column)
    except LLMValidationError as exc:
        _log.warning("cleaning_proposer_fallback", error=str(exc))
        reporter.log(f"Cleaning proposer fell back to deterministic ops: {exc}", level="warning")
        ops = propose_cleaning_ops(train_profile)
    cleaned, manifest = execute_cleaning(store, split.train_ref.artifact_id, ops, run_id=run_id)
    reporter.stage("clean", "done", detail=f"{len(manifest.ops)} op(s)")
    reporter.log(f"Applied {len(manifest.ops)} cleaning operation(s)", stage="clean")

    # -- LLM feature plan + deterministic execution -----------------------------
    reporter.stage("features", "active")
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
        reporter.stage("features", "done", detail=f"{len(feature_plan.proposals)} proposal(s)")
        reporter.log(f"Proposed {len(feature_plan.proposals)} feature(s)", stage="features")
    except LLMValidationError as exc:
        _log.warning("feature_proposer_skipped", error=str(exc))
        reporter.stage("features", "done", detail="skipped")
        reporter.log(f"Feature proposer skipped: {exc}", level="warning", stage="features")

    # -- Screening round: cheap probes BEFORE the plan is drafted -------------
    # A senior engineer runs cheap probes first, reads them, then spends the
    # budget — the planner used to choose blind. dummy/linear/lightgbm here
    # use the SAME env_hash as the real training call below, so the plan's
    # dummy_baseline/lightgbm_default specs reuse these results outright
    # (via _train_all's identical-key check) whenever the plan keeps their
    # vetted defaults, at zero additional training cost.
    reporter.stage("plan", "active")
    env_hash = compute_env_hash()
    reporter.log("Running screening probes (dummy, linear, lightgbm)", stage="plan")
    screening_results = _run_screening(
        store,
        modeling_dataset.meta.artifact_id,
        task_type=spec.task_type,
        primary_metric=spec.primary_metric,
        n_rows=modeling_dataset.n_rows,
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=env_hash,
        mlflow_enabled=mlflow_enabled,
    )

    # -- LLM experiment plan + GATE 2 ----------------------------------------
    try:
        plan = plan_experiments(
            client,
            store,
            profile=train_profile,
            problem=spec,
            n_rows=modeling_dataset.n_rows,
            run_id=run_id,
            store_root=store.root,
            screening_results=screening_results,
        )
        specs = plan.experiments
        plan_rationale = plan.rationale
        plan_id = plan.meta.artifact_id
        excluded_templates = plan.excluded_templates
    except LLMValidationError as exc:
        _log.warning("experiment_planner_fallback", error=str(exc))
        specs = default_experiment_specs(include_tabpfn=include_tabpfn)
        plan_rationale = "deterministic fallback plan (LLM planner failed validation)"
        plan_id = ""
        excluded_templates = ()
    total = sum(s.budget.wall_time_seconds for s in specs)
    reporter.stage("plan", "done", detail=f"{len(specs)} experiment(s) planned")
    reporter.log(f"Planned {len(specs)} experiment(s), ~{total:.0f}s estimated", stage="plan")
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
            "excluded_templates": [
                {"template_id": e.template_id, "reason": e.reason} for e in excluded_templates
            ],
            "screening_results": screening_results,
            "data_triage": {
                "rows_before": triage_report.rows_before,
                "rows_after": triage_report.rows_after,
                "remedies": [
                    f"{r.action.value} on '{r.column}': {r.affected_rows} row(s)"
                    for r in triage_report.remedies
                ],
            },
        },
    )

    # -- Train, evaluate, critique -------------------------------------------
    reporter.stage("train", "active")
    results, skipped = _train_all(
        store,
        modeling_dataset.meta.artifact_id,
        tuple(specs),
        task_type=spec.task_type,
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=env_hash,
        mlflow_enabled=mlflow_enabled,
    )
    dummy_key = _dummy_key(tuple(specs))
    _check_experiment_quorum(tuple(specs), results, skipped, dummy_key)
    best_key = _select_best(results, spec.primary_metric, dummy_key)
    reporter.stage("train", "done", detail=f"{len(results)} completed, {len(skipped)} skipped")
    reporter.log(f"Trained {len(results)} experiment(s); winner so far: {best_key}", stage="train")

    reporter.stage("evaluate", "active")
    report = build_evaluation_report(
        store,
        run_id=run_id,
        model_result=results[best_key],
        dummy_result=results[dummy_key],
        profile=train_profile,
        primary_metric=spec.primary_metric,
    )
    reporter.stage("evaluate", "done")
    reporter.log(f"Evaluation report built for winner {best_key}", stage="evaluate")

    reporter.stage("critic", "active")
    all_outcomes = list_experiment_outcomes(store, run_id)
    planned_ids = screening_experiment_ids() + tuple(s.experiment_id for s in specs)
    critique = critique_evaluation(
        client,
        store,
        report=report,
        primary_metric=spec.primary_metric,
        run_id=run_id,
        outcomes=all_outcomes,
        planned_experiment_ids=planned_ids,
        specs=tuple(specs),
        excluded_templates=excluded_templates,
        winner_experiment_id=best_key,
        dummy_experiment_id=dummy_key,
    )
    reporter.stage("critic", "done", detail=critique.verdict.value)
    reporter.log(f"Critic verdict: {critique.verdict.value}", stage="critic")

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
        triage_report_id=triage_report.meta.artifact_id,
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
