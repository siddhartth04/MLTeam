"""The planner iteration loop: Plan -> Execute -> Evaluate -> Critique -> Improve.

Runs on top of the agentic pipeline head (Gates 1 & 2 included) and is capped
by an explicit RunBudget. Each ITERATE verdict triggers (a) budget-aware Optuna
tuning of the best tunable template and (b) an LLM plan revision addressing
the critique actions — both loud-fallback, never silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from mlteam.agents.critic import critique_evaluation
from mlteam.agents.experiment_planner import revise_experiment_plan
from mlteam.agents.hpo import SEARCH_SPACES, SearchSpace, tune_experiment
from mlteam.agents.trainer import TrainResult, compute_env_hash
from mlteam.contracts import (
    CritiqueReport,
    EvaluationReport,
    ExperimentState,
    TrainedModel,
    Verdict,
)
from mlteam.evaluation.metrics import better_than
from mlteam.evaluation.report import build_evaluation_report
from mlteam.evaluation.results import list_experiment_outcomes, store_experiment_outcome
from mlteam.llm.client import LLMClient, LLMValidationError
from mlteam.orchestration.agentic import AgenticResult, run_agentic_pipeline
from mlteam.orchestration.budget import BudgetTracker, RunBudget
from mlteam.orchestration.gates import ApprovalGate
from mlteam.orchestration.pipeline import _train_all
from mlteam.orchestration.progress import NULL_REPORTER, ProgressReporter
from mlteam.store.guard import GuardedArtifactStore

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IterationRecord:
    index: int
    evaluation_report_id: str
    critique_id: str
    verdict: Verdict
    best_experiment_id: str
    new_experiment_ids: tuple[str, ...]


@dataclass(frozen=True)
class IterativeResult:
    status: str  # "approved" | "halted" | "budget_exhausted"
    stop_reason: str
    initial: AgenticResult
    iterations: tuple[IterationRecord, ...]
    final_report: EvaluationReport
    final_critique: CritiqueReport
    best_experiment_id: str
    best_model: TrainedModel
    budget_snapshot: dict[str, Any] = field(default_factory=dict)


def _results_summary(results: dict[str, TrainResult], primary_metric: str) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": eid,
            "template_id": r.model.template_id,
            "config": r.model.config,
            primary_metric: round(r.metrics[primary_metric].mean, 6),
        }
        for eid, r in sorted(results.items())
    ]


def _best_key(results: dict[str, TrainResult], primary_metric: str, dummy_key: str) -> str:
    best = None
    for eid, result in results.items():
        if eid == dummy_key:
            continue
        if best is None or better_than(
            primary_metric,
            result.metrics[primary_metric].mean,
            results[best].metrics[primary_metric].mean,
        ):
            best = eid
    if best is None:
        raise RuntimeError("no candidate experiments to select from")
    return best


def run_iterative_pipeline(
    store: GuardedArtifactStore,
    data_path: Path,
    *,
    goal_text: str,
    client: LLMClient,
    gate: ApprovalGate,
    budget: RunBudget | None = None,
    run_id: str | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    hpo_trials: int = 15,
    hpo_space_override: SearchSpace | None = None,
    mlflow_enabled: bool = False,
    reporter: ProgressReporter = NULL_REPORTER,
) -> IterativeResult:
    """Full loop: agentic head, then critique-driven improvement under budget.

    ``run_id``, when given, is passed straight through to
    ``run_agentic_pipeline`` instead of letting it mint its own via
    ``new_run_id()``. This lets a caller (the API's run_manager) generate
    the pipeline run id BEFORE calling this function and register it with
    the dashboard immediately -- previously the pipeline's own run id was
    only learnable from ``IterativeResult.initial.run_id`` after this
    entire call returned, so experiments/lineage/artifacts were unresolvable
    (correctly returning empty/pending, per Task 1 of
    REPORT_INTEGRITY_ROADMAP.md, rather than leaking) for a run's ENTIRE
    in-flight duration -- often over an hour for a large dataset.
    """
    budget = budget or RunBudget()
    tracker = BudgetTracker(budget, client)

    initial = run_agentic_pipeline(
        store,
        data_path,
        goal_text=goal_text,
        client=client,
        gate=gate,
        run_id=run_id,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_folds=n_folds,
        mlflow_enabled=mlflow_enabled,
        reporter=reporter,
    )
    spec = initial.problem_spec
    primary = spec.primary_metric
    run_id = initial.run_id
    dataset_id = initial.pipeline.modeling_dataset_id
    n_rows = store.get(dataset_id).n_rows
    env_hash = compute_env_hash()

    results: dict[str, TrainResult] = dict(initial.pipeline.train_results)
    # Use the plan's AUTHORITATIVE dummy experiment id (audit Task 8): with
    # screening results merged into train_results, more than one entry can
    # have template_id == "dummy_baseline" (e.g. "screen-dummy" alongside
    # the plan's own "exp-dummy"), so scanning for the first match is no
    # longer well-defined.
    dummy_key = initial.pipeline.dummy_experiment_id or next(
        eid for eid, r in results.items() if r.model.template_id == "dummy_baseline"
    )
    report = initial.pipeline.evaluation_report
    critique = initial.critique
    records: list[IterationRecord] = []
    train_profile = store.get(initial.pipeline.train_profile_id)

    status, stop_reason = "approved", "critic approved"
    while True:
        if critique.verdict is Verdict.APPROVE:
            status, stop_reason = "approved", "critic approved"
            reporter.stage("iterate", "done", detail=f"{len(records)} iteration(s)")
            break
        if critique.verdict is Verdict.HALT:
            status, stop_reason = "halted", "critic halted"
            reporter.stage("iterate", "done", detail="halted")
            break
        ok, reason = tracker.can_continue()
        if not ok:
            status, stop_reason = "budget_exhausted", reason
            _log.warning("iteration_budget_exhausted", reason=reason)
            reporter.stage("iterate", "done", detail=reason)
            break

        tracker.record_iteration()
        iteration = tracker.iterations_used
        _log.info("iteration_started", index=iteration, budget=tracker.snapshot())
        reporter.stage("iterate", "active", detail=f"iteration {iteration}")
        reporter.log(f"Starting iteration {iteration}: {critique.verdict.value}", stage="iterate")
        new_ids: list[str] = []

        # (a) Budget-aware HPO on the best tunable template.
        best_now = _best_key(results, primary, dummy_key)
        tunable = results[best_now].model.template_id
        hpo_id = f"hpo-{tunable}-iter{iteration}"
        if hpo_trials > 0 and tunable in SEARCH_SPACES and hpo_id not in results:
            started = time.monotonic()
            try:
                hpo_outcome = tune_experiment(
                    store,
                    dataset_id,
                    template_id=tunable,
                    task_type=spec.task_type,
                    primary_metric=primary,
                    run_id=run_id,
                    seed=seed + iteration,
                    # Re-score the winning config on the BASE seed's folds —
                    # the same folds every other experiment this run used —
                    # so the reported score isn't the search's own biased
                    # estimate and remains fold-paired with incumbents for
                    # significance testing (see agents/hpo.py, audit Task 5).
                    report_seed=seed,
                    n_folds=n_folds,
                    env_hash=env_hash,
                    n_trials=hpo_trials,
                    space=hpo_space_override,
                    experiment_id=hpo_id,
                    mlflow_enabled=mlflow_enabled,
                    target_transform=results[best_now].model.target_transform,
                )
                results[hpo_id] = hpo_outcome.result
                new_ids.append(hpo_id)
                store_experiment_outcome(
                    store,
                    experiment_id=hpo_id,
                    template_id=tunable,
                    state=ExperimentState.COMPLETED,
                    run_id=run_id,
                    result=hpo_outcome.result,
                    wall_time_seconds=time.monotonic() - started,
                )
            except Exception as exc:
                store_experiment_outcome(
                    store,
                    experiment_id=hpo_id,
                    template_id=tunable,
                    state=ExperimentState.FAILED,
                    run_id=run_id,
                    wall_time_seconds=time.monotonic() - started,
                    error=str(exc),
                )
                _log.warning(
                    "hpo_retrain_skipped",
                    template=tunable,
                    error=str(exc),
                    hint="HPO retrain failed; continuing without it this iteration",
                )

        # (b) LLM plan revision addressing the critique actions.
        try:
            revision = revise_experiment_plan(
                client,
                problem=spec,
                critique_actions=critique.iteration_actions,
                prior_results=_results_summary(results, primary),
                prior_experiment_ids=set(results),
                n_rows=store.get(dataset_id).n_rows,
            )
            revised_results, _ = _train_all(
                store,
                dataset_id,
                revision.specs,
                task_type=spec.task_type,
                run_id=run_id,
                seed=seed,
                n_folds=n_folds,
                env_hash=env_hash,
                mlflow_enabled=mlflow_enabled,
                reporter=reporter,
            )
            results.update(revised_results)
            new_ids.extend(revised_results)
        except LLMValidationError as exc:
            _log.warning("plan_revision_skipped", error=str(exc))

        # Re-evaluate the new best against the dummy floor, re-critique.
        best_now = _best_key(results, primary, dummy_key)
        report = build_evaluation_report(
            store,
            run_id=run_id,
            model_result=results[best_now],
            dummy_result=results[dummy_key],
            profile=train_profile,
            primary_metric=primary,
            n_rows=n_rows,
            dataset_id=dataset_id,
            task_type=spec.task_type,
            cost_asymmetry=spec.cost_asymmetry,
            n_folds=n_folds,
            seed=seed,
            reporter=reporter,
        )
        # No terminal-state audit here: the "planned" set is this iteration's
        # dynamically-revised candidates, not a fixed plan to check completeness
        # against (that check applies to the initial full plan, already done
        # in agentic.py). Outcomes and the winner/dummy margin are still real,
        # complete evidence and enrich the re-critique regardless.
        critique = critique_evaluation(
            client,
            store,
            report=report,
            primary_metric=primary,
            run_id=run_id,
            outcomes=list_experiment_outcomes(store, run_id),
            winner_experiment_id=best_now,
            dummy_experiment_id=dummy_key,
        )
        records.append(
            IterationRecord(
                index=iteration,
                evaluation_report_id=report.meta.artifact_id,
                critique_id=critique.meta.artifact_id,
                verdict=critique.verdict,
                best_experiment_id=best_now,
                new_experiment_ids=tuple(new_ids),
            )
        )
        _log.info(
            "iteration_finished",
            index=iteration,
            verdict=critique.verdict.value,
            best=best_now,
        )
        reporter.log(
            f"Iteration {iteration} finished: {critique.verdict.value} (best: {best_now})",
            stage="iterate",
        )

    final_best = _best_key(results, primary, dummy_key)
    return IterativeResult(
        status=status,
        stop_reason=stop_reason,
        initial=initial,
        iterations=tuple(records),
        final_report=report,
        final_critique=critique,
        best_experiment_id=final_best,
        best_model=results[final_best].model,
        budget_snapshot=tracker.snapshot(),
    )
