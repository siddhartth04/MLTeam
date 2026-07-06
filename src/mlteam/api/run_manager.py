"""Background run manager: executes the full pipeline and publishes progress.

Wires the registry to the pipeline: a registry-backed ApprovalGate surfaces the
three human gates in the dashboard (blocking the run thread until resolved), and
stage transitions + the LLM cost meter stream live. Runs the iterative loop,
then explainability, then (auto-signed) deployment bundle, then the final report.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

import structlog

from mlteam.api.events import LogEvent, RunStatus, Stage, StageStatus
from mlteam.api.registry import REGISTRY, RunRegistry
from mlteam.contracts import ProblemSpec, TrainedModel
from mlteam.llm.client import LLMClient
from mlteam.llm.transport import Transport
from mlteam.orchestration.gates import ApprovalGate, GateDecision, GateName, GateRejectedError
from mlteam.orchestration.pipeline import PipelineResult
from mlteam.store import ArtifactStore, open_store
from mlteam.store.ledger import record_outcome

_log = structlog.get_logger(__name__)

_GATE_TO_STAGE = {
    GateName.PROBLEM_SPEC: Stage.GATE_SPEC,
    GateName.TRAINING_PLAN: Stage.GATE_PLAN,
    GateName.DEPLOYMENT: Stage.GATE_DEPLOY,
}


def _record_ledger_outcome(
    store: ArtifactStore,
    *,
    spec: ProblemSpec,
    pipeline: PipelineResult,
    best_model: TrainedModel,
) -> None:
    """Best-effort write to the outcome ledger — never fails the run."""
    try:
        from mlteam.agents.deployer import build_deployment_dossier
        from mlteam.evaluation.results import list_experiment_outcomes

        train_profile = store.get(pipeline.train_profile_id)
        n_features = len(train_profile.columns)
        n_categorical = sum(1 for c in train_profile.columns if c.dtype == "String")
        outcomes = list_experiment_outcomes(store, pipeline.run_id)
        dossier = build_deployment_dossier(
            model=best_model,
            problem=spec,
            train_dataset_id=pipeline.modeling_dataset_id,
            split_plan_id=pipeline.split_plan_id,
            outcomes=outcomes,
        )
        margin = None
        if dossier["winner_margin"] and dossier["winner_margin"]["vs_runner_up"]:
            margin = dossier["winner_margin"]["vs_runner_up"]["improvement"]
        record_outcome(
            store.root,
            task_type=spec.task_type.value,
            n_rows=store.get(pipeline.modeling_dataset_id).n_rows,
            n_features=n_features,
            categorical_share=n_categorical / n_features if n_features else 0.0,
            winner_template=best_model.template_id,
            margin_vs_second=margin,
            primary_metric=spec.primary_metric,
        )
    except Exception:  # ledger is a nice-to-have; never breaks a completed run
        _log.warning("ledger_write_skipped", exc_info=True)


class RegistryGate:
    """ApprovalGate that surfaces gates in the dashboard and blocks on approval."""

    def __init__(self, registry: RunRegistry, ui_run_id: str) -> None:
        self._registry = registry
        self._ui_run_id = ui_run_id

    def request(self, gate: GateName, summary: dict[str, Any]) -> GateDecision:
        stage = _GATE_TO_STAGE[gate]
        request = self._registry.open_gate(self._ui_run_id, stage, summary)
        request.event.wait()  # blocks the run thread until the UI resolves it
        return GateDecision(approved=request.approved, notes=request.notes, approver="dashboard")


def launch_run(
    *,
    ui_run_id: str,
    data_path: Path,
    goal_text: str,
    dataset_name: str,
    transport: Transport,
    store_root: Path,
    model: str = "claude-sonnet-4-6",
    seed: int = 42,
    n_folds: int = 3,
    registry: RunRegistry = REGISTRY,
) -> threading.Thread:
    """Create the run record and start it on a daemon thread."""
    registry.create(ui_run_id, goal_text=goal_text, dataset_name=dataset_name)
    thread = threading.Thread(
        target=_execute,
        kwargs=dict(
            ui_run_id=ui_run_id,
            data_path=data_path,
            goal_text=goal_text,
            transport=transport,
            store_root=store_root,
            model=model,
            seed=seed,
            n_folds=n_folds,
            registry=registry,
        ),
        daemon=True,
        name=f"run-{ui_run_id}",
    )
    thread.start()
    return thread


def _execute(
    *,
    ui_run_id: str,
    data_path: Path,
    goal_text: str,
    transport: Transport,
    store_root: Path,
    model: str,
    seed: int,
    n_folds: int,
    registry: RunRegistry,
) -> None:
    # Imported here to keep module import light and avoid heavy deps at import.
    from mlteam.agents.deployer import build_deployment_bundle
    from mlteam.agents.explainer import explain_model
    from mlteam.contracts import is_classification
    from mlteam.orchestration import run_iterative_pipeline
    from mlteam.orchestration.budget import RunBudget
    from mlteam.reporting import ReportInputs, build_final_report

    store = open_store(store_root)
    gate: ApprovalGate = RegistryGate(registry, ui_run_id)
    client = LLMClient(transport, model=model, trace=None)
    registry.set_status(ui_run_id, RunStatus.RUNNING)

    def stage(s: Stage, status: StageStatus, detail: str = "") -> None:
        registry.update_stage(ui_run_id, s, status, detail)

    def note(message: str, *, level: str = "info", s: Stage | None = None, **extra: Any) -> None:
        registry.log(ui_run_id, LogEvent(level=level, stage=s, message=message, extra=extra))
        registry.set_cost(ui_run_id, client.total_cost_usd)

    try:
        note("Run started", level="success")
        stage(Stage.REQUIREMENT, StageStatus.ACTIVE)

        # The iterative pipeline runs Gates 1 & 2 internally via `gate`; we mark
        # the surrounding stages around it for the flow diagram.
        result = _run_with_stage_tracking(
            registry,
            ui_run_id,
            lambda: run_iterative_pipeline(
                store,
                data_path,
                goal_text=goal_text,
                client=client,
                gate=gate,
                budget=RunBudget(max_iterations=2),
                seed=seed,
                n_folds=n_folds,
                hpo_trials=8,
            ),
        )
        registry.set_cost(ui_run_id, client.total_cost_usd)
        spec = result.initial.problem_spec
        pipeline = result.initial.pipeline
        best_key = result.best_experiment_id
        best_model = result.best_model
        note(
            f"Critic verdict: {result.final_critique.verdict.value} (status {result.status})",
            level="success" if result.status == "approved" else "warning",
            s=Stage.CRITIC,
        )

        # Explainability.
        stage(Stage.EXPLAIN, StageStatus.ACTIVE)
        explain_report = explain_model(
            store,
            best_model,
            train_dataset_id=pipeline.modeling_dataset_id,
            task_is_classification=is_classification(spec.task_type),
            run_id=result.initial.run_id,
            seed=seed,
        )
        stage(Stage.EXPLAIN, StageStatus.DONE, f"top: {explain_report.importances[0].feature}")
        note("Explainability figures generated", level="success", s=Stage.EXPLAIN)

        # Plain-language insight narration (best-effort; never blocks the run).
        from mlteam.agents.insight_narrator import narrate_insights

        insights = narrate_insights(
            client,
            store,
            problem=spec,
            evaluation=result.final_report,
            explainability=explain_report,
            run_id=result.initial.run_id,
        )
        if insights is not None:
            note(f"Insights narrated ({len(insights.findings)} findings)", s=Stage.EXPLAIN)

        # Deployment bundle (Gate 3 via the dashboard).
        stage(Stage.DEPLOY, StageStatus.ACTIVE)
        from mlteam.agents import read_dataset
        from mlteam.evaluation.results import list_experiment_outcomes

        features = [
            c
            for c in read_dataset(store, store.get(pipeline.modeling_dataset_id)).columns
            if c != spec.target_column
        ]
        try:
            build_deployment_bundle(
                store,
                model=best_model,
                profile=store.get(pipeline.train_profile_id),
                problem=spec,
                train_dataset_id=pipeline.modeling_dataset_id,
                feature_names=features,
                gate=gate,
                run_id=result.initial.run_id,
                split_plan_id=pipeline.split_plan_id,
                outcomes=list_experiment_outcomes(store, result.initial.run_id),
                evaluation=result.final_report,
                explainability=explain_report,
                critique=result.final_critique,
            )
            stage(Stage.DEPLOY, StageStatus.DONE)
            note("Deployment bundle signed off", level="success", s=Stage.DEPLOY)
        except GateRejectedError:
            stage(Stage.DEPLOY, StageStatus.SKIPPED, "deployment declined")
            note("Deployment declined at Gate 3", level="warning", s=Stage.GATE_DEPLOY)

        # Final report.
        stage(Stage.REPORT, StageStatus.ACTIVE)
        report = build_final_report(
            store,
            ReportInputs(
                problem=spec,
                model=best_model,
                evaluation=result.final_report,
                critique=result.final_critique,
                split_plan=store.get(pipeline.split_plan_id),
                goal_text=goal_text,
                best_template_id=pipeline.best_template_id,
                best_experiment_id=best_key,
                explainability=explain_report,
                insights=insights,
                feature_plan=(
                    store.get(result.initial.feature_plan_id)
                    if pipeline.feature_dataset_id and result.initial.feature_plan_id
                    else None
                ),
                iterations=tuple(
                    {
                        "index": it.index,
                        "verdict": it.verdict.value,
                        "best": it.best_experiment_id,
                        "new": list(it.new_experiment_ids),
                    }
                    for it in result.iterations
                ),
            ),
            run_id=result.initial.run_id,
        )
        stage(Stage.REPORT, StageStatus.DONE)
        note("Final report generated", level="success", s=Stage.REPORT)

        _record_ledger_outcome(store, spec=spec, pipeline=pipeline, best_model=best_model)

        record = registry.get(ui_run_id)
        if record is not None:
            record.result_run_id = result.initial.run_id
        registry.finish(
            ui_run_id, status=RunStatus.COMPLETED, final_report_id=report.meta.artifact_id
        )

    except GateRejectedError as exc:
        registry.log(ui_run_id, LogEvent(level="warning", message=f"Run rejected at a gate: {exc}"))
        registry.finish(ui_run_id, status=RunStatus.REJECTED, error=str(exc))
    except Exception as exc:  # any failure ends the run cleanly and visibly
        _log.exception("run_failed", ui_run_id=ui_run_id)
        registry.log(
            ui_run_id,
            LogEvent(
                level="error",
                message=f"Run failed: {exc}",
                extra={"trace": traceback.format_exc()[-1500:]},
            ),
        )
        registry.finish(ui_run_id, status=RunStatus.FAILED, error=str(exc))


def _run_with_stage_tracking(registry: RunRegistry, ui_run_id: str, fn: Any) -> Any:
    """Mark the requirement->critic stages active/done around the loop call.

    The iterative pipeline is a single call; for a truthful flow diagram we
    optimistically light the deterministic stages, then let the gate/critic
    stages settle from the loop's own logging. This keeps the UI lively without
    threading a callback through every agent.
    """
    for s in (
        Stage.INGEST,
        Stage.PROFILE,
        Stage.SPLIT,
        Stage.CLEAN,
        Stage.FEATURES,
        Stage.PLAN,
        Stage.TRAIN,
        Stage.EVALUATE,
        Stage.CRITIC,
    ):
        registry.update_stage(ui_run_id, s, StageStatus.PENDING)
    registry.update_stage(ui_run_id, Stage.REQUIREMENT, StageStatus.DONE)
    result = fn()
    for s in (
        Stage.INGEST,
        Stage.PROFILE,
        Stage.SPLIT,
        Stage.CLEAN,
        Stage.FEATURES,
        Stage.PLAN,
        Stage.TRAIN,
        Stage.EVALUATE,
        Stage.CRITIC,
    ):
        registry.update_stage(ui_run_id, s, StageStatus.DONE)
    if result.iterations:
        registry.update_stage(
            ui_run_id, Stage.ITERATE, StageStatus.DONE, f"{len(result.iterations)} iteration(s)"
        )
    else:
        registry.update_stage(ui_run_id, Stage.ITERATE, StageStatus.SKIPPED, "not needed")
    return result
