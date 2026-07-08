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
from mlteam.contracts import ProblemSpec, TrainedModel, Verdict
from mlteam.llm.client import LLMClient
from mlteam.llm.transport import Transport
from mlteam.orchestration.gates import ApprovalGate, GateDecision, GateName, GateRejectedError
from mlteam.orchestration.pipeline import PipelineResult
from mlteam.orchestration.progress import ProgressReporter
from mlteam.store import ArtifactStore, open_store
from mlteam.store.ledger import record_outcome

_log = structlog.get_logger(__name__)

_GATE_TO_STAGE = {
    GateName.PROBLEM_SPEC: Stage.GATE_SPEC,
    GateName.TRAINING_PLAN: Stage.GATE_PLAN,
    GateName.DEPLOYMENT: Stage.GATE_DEPLOY,
}


class RegistryReporter:
    """ProgressReporter that forwards live stage/log events to the run
    registry (and ticks the LLM cost meter on every call) so the dashboard's
    SSE stream reflects real progress as the pipeline runs."""

    def __init__(self, registry: RunRegistry, ui_run_id: str, client: LLMClient) -> None:
        self._registry = registry
        self._ui_run_id = ui_run_id
        self._client = client

    def stage(self, stage: str, status: str, detail: str = "") -> None:
        self._registry.update_stage(self._ui_run_id, Stage(stage), StageStatus(status), detail)
        self._registry.set_cost(self._ui_run_id, self._client.total_cost_usd)

    def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None:
        self._registry.log(
            self._ui_run_id,
            LogEvent(level=level, stage=Stage(stage) if stage else None, message=message),
        )
        self._registry.set_cost(self._ui_run_id, self._client.total_cost_usd)


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


def _planned_experiment_ids(store: ArtifactStore, pipeline: PipelineResult) -> tuple[str, ...]:
    """Screening ids + the initial plan's spec ids, for the Gate 3 dossier's
    terminal-state audit. Empty (no audit surfaced) when the LLM planner fell
    back to the deterministic default plan, which has no stored ExperimentPlan
    artifact to read specs from."""
    from mlteam.orchestration.agentic import screening_experiment_ids

    if not pipeline.experiment_plan_id:
        return ()
    plan = store.get(pipeline.experiment_plan_id)
    return screening_experiment_ids() + tuple(s.experiment_id for s in plan.experiments)


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
    # Defaults mirror APISettings' own defaults (audit finding M4): 5, not 3
    # -- k=3 is the weakest defensible CV, and combined with the paired
    # significance test (Task 2) it makes a genuinely large improvement come
    # out borderline-non-significant purely from having too few folds to
    # establish confidence, not from the improvement being small. Callers
    # that care should pass the live APISettings values explicitly rather
    # than rely on these literals staying in sync.
    n_folds: int = 5,
    hpo_trials: int = 8,
    max_iterations: int = 2,
    max_llm_cost_usd: float = 5.0,
    allow_deploy_on_halt: bool = False,
    name: str = "",
    registry: RunRegistry = REGISTRY,
) -> threading.Thread:
    """Create the run record and start it on a daemon thread."""
    registry.create(ui_run_id, goal_text=goal_text, dataset_name=dataset_name, name=name)
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
            hpo_trials=hpo_trials,
            max_iterations=max_iterations,
            max_llm_cost_usd=max_llm_cost_usd,
            allow_deploy_on_halt=allow_deploy_on_halt,
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
    hpo_trials: int,
    max_iterations: int,
    max_llm_cost_usd: float,
    allow_deploy_on_halt: bool = False,
    registry: RunRegistry,
) -> None:
    # Imported here to keep module import light and avoid heavy deps at import.
    from mlteam.agents.deployer import build_deployment_bundle
    from mlteam.agents.explainer import explain_model
    from mlteam.orchestration import new_run_id, run_iterative_pipeline
    from mlteam.orchestration.budget import RunBudget
    from mlteam.reporting import ReportInputs, build_final_report

    store = open_store(store_root)
    gate: ApprovalGate = RegistryGate(registry, ui_run_id)
    client = LLMClient(transport, model=model, trace=None)
    registry.set_status(ui_run_id, RunStatus.RUNNING)
    reporter: ProgressReporter = RegistryReporter(registry, ui_run_id, client)

    # Minted HERE, before the pipeline runs, and registered immediately --
    # not learned from IterativeResult.initial.run_id after the entire
    # pipeline returns. Previously result_run_id stayed None (correctly
    # returning empty/pending per Task 1 of REPORT_INTEGRITY_ROADMAP.md,
    # never a cross-run leak) for a run's ENTIRE in-flight duration, often
    # over an hour for a large dataset -- experiments/lineage/artifacts
    # were unresolvable the whole time a run was actually progressing.
    pipeline_run_id = new_run_id()
    registry.set_result_run_id(ui_run_id, pipeline_run_id)

    def stage(s: Stage, status: StageStatus, detail: str = "") -> None:
        registry.update_stage(ui_run_id, s, status, detail)

    def note(message: str, *, level: str = "info", s: Stage | None = None, **extra: Any) -> None:
        registry.log(ui_run_id, LogEvent(level=level, stage=s, message=message, extra=extra))
        registry.set_cost(ui_run_id, client.total_cost_usd)

    try:
        note("Run started", level="success")
        stage(Stage.REQUIREMENT, StageStatus.ACTIVE)

        # Check if cancellation was requested before starting.
        if registry.is_cancelled(ui_run_id):
            raise RuntimeError("run cancelled")

        # The iterative pipeline runs Gates 1 & 2 internally via `gate`; the
        # reporter drives real per-stage transitions as each agent boundary
        # is crossed (see orchestration/progress.py).
        result = run_iterative_pipeline(
            store,
            data_path,
            goal_text=goal_text,
            client=client,
            gate=gate,
            budget=RunBudget(max_iterations=max_iterations, max_llm_cost_usd=max_llm_cost_usd),
            run_id=pipeline_run_id,
            seed=seed,
            n_folds=n_folds,
            hpo_trials=hpo_trials,
            reporter=reporter,
        )
        # Check if cancellation was requested after the pipeline completes.
        if registry.is_cancelled(ui_run_id):
            raise RuntimeError("run cancelled")
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
            task_type=spec.task_type,
            primary_metric=spec.primary_metric,
            n_folds=n_folds,
            run_id=result.initial.run_id,
            seed=seed,
            reporter=reporter,
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

        halted = result.final_critique.verdict is Verdict.HALT
        if halted and not allow_deploy_on_halt:
            # The Critic found this model unfit to ship -- offering a human
            # a deployment sign-off anyway would make Gate 3 a rubber stamp
            # rather than a real decision point (observed live 2026-07-08:
            # halt at 16:32, Gate 3 approved at 16:40, for a model whose
            # every candidate had negative R²). Skip the bundle entirely
            # rather than build one and hope a human notices the buried
            # verdict. allow_deploy_on_halt is the explicit, logged escape
            # hatch for a human who has already decided to ship anyway.
            stage(Stage.DEPLOY, StageStatus.SKIPPED, "blocked: critic verdict halt")
            note(
                "Deployment blocked: Critic verdict is HALT. Set "
                "MLTEAM_API_ALLOW_DEPLOY_ON_HALT=true to override.",
                level="warning",
                s=Stage.DEPLOY,
            )
        else:
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
                    planned_experiment_ids=_planned_experiment_ids(store, pipeline),
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
                cleaning_manifest=store.get(pipeline.cleaning_manifest_id),
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

        # result_run_id was already set at launch (pipeline_run_id, above);
        # run_agentic_pipeline's `run_id = run_id or new_run_id()` means
        # result.initial.run_id is guaranteed identical since a non-empty
        # run_id was passed in -- no second write needed here.
        registry.finish(
            ui_run_id, status=RunStatus.COMPLETED, final_report_id=report.meta.artifact_id
        )

    except GateRejectedError as exc:
        registry.log(ui_run_id, LogEvent(level="warning", message=f"Run rejected at a gate: {exc}"))
        registry.finish(ui_run_id, status=RunStatus.REJECTED, error=str(exc))
    except RuntimeError as exc:
        if "run cancelled" in str(exc):
            registry.log(ui_run_id, LogEvent(level="warning", message="Run cancelled by user"))
            registry.finish(ui_run_id, status=RunStatus.CANCELLED, error="User cancelled")
        else:
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
