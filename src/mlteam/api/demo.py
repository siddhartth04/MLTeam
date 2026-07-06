"""Demo seed: a real, completed run so the dashboard is populated on first open.

Uses the recorded LLM fixture (no API key, no network) and the synthetic churn
dataset, auto-approving all gates, and registers it in the run registry as a
completed run with a full trace and a final report.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from mlteam.api.events import LogEvent, RunStatus, Stage, StageStatus
from mlteam.api.registry import REGISTRY, RunRegistry
from mlteam.llm import FixtureTransport, LLMClient

_log = structlog.get_logger(__name__)

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "llm"
    / "iterative_churn_approved.json"
)
DEMO_UI_RUN_ID = "demo-seed"


def _demo_dataset(path: Path) -> None:
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(7)
    n = 400
    tenure = rng.integers(1, 72, n).astype(float)
    calls = rng.poisson(1.6, n).astype(float)
    charges = np.round(rng.uniform(20, 120, n), 2)
    contract = rng.choice(["monthly", "yearly", "N/A"], n, p=[0.55, 0.4, 0.05])
    logit = -1.0 - 0.03 * tenure + 0.6 * calls + np.where(contract == "monthly", 1.0, 0.0)
    churned = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    charges[rng.uniform(size=n) < 0.05] = np.nan
    df = pl.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "num_support_calls": calls,
            "monthly_charges": charges,
            "contract_type": contract,
            "region": ["north"] * n,
            "churned": churned,
        }
    )
    pl.concat([df, df.head(4)]).write_csv(path)


def seed_demo_run(
    *, store_root: Path, data_dir: Path, registry: RunRegistry = REGISTRY
) -> str | None:
    """Run the demo pipeline synchronously; returns the UI run id (or None)."""
    if registry.get(DEMO_UI_RUN_ID) is not None:
        return DEMO_UI_RUN_ID
    if not _FIXTURE.exists():
        _log.warning("demo_fixture_missing", path=str(_FIXTURE))
        return None

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "demo_churn.csv"
    if not csv_path.exists():
        _demo_dataset(csv_path)

    from mlteam.agents.deployer import build_deployment_bundle
    from mlteam.agents.explainer import explain_model
    from mlteam.contracts import is_classification
    from mlteam.orchestration import AutoApproveGate, run_iterative_pipeline
    from mlteam.orchestration.budget import RunBudget
    from mlteam.reporting import ReportInputs, build_final_report
    from mlteam.store import open_store

    registry.create(
        DEMO_UI_RUN_ID,
        goal_text="Identify customers likely to churn so retention can call them first.",
        dataset_name="demo_churn.csv",
    )
    registry.set_status(DEMO_UI_RUN_ID, RunStatus.RUNNING)

    def stage(s: Stage, status: StageStatus, detail: str = "") -> None:
        registry.update_stage(DEMO_UI_RUN_ID, s, status, detail)

    def note(msg: str, level: str = "info", s: Stage | None = None) -> None:
        registry.log(DEMO_UI_RUN_ID, LogEvent(level=level, stage=s, message=msg))

    try:
        store = open_store(store_root)
        client = LLMClient(FixtureTransport.from_file(_FIXTURE), model="claude-sonnet-4-6")
        note("Demo run started", "success")
        for s in list(Stage)[:13]:
            stage(s, StageStatus.DONE)
        result = run_iterative_pipeline(
            store,
            csv_path,
            goal_text="reduce churn by proactively contacting at-risk customers",
            client=client,
            gate=AutoApproveGate(),
            budget=RunBudget(max_iterations=2),
            seed=42,
            n_folds=3,
            hpo_trials=6,
        )
        registry.set_cost(DEMO_UI_RUN_ID, client.total_cost_usd)
        stage(Stage.ITERATE, StageStatus.DONE, f"{len(result.iterations)} iteration(s)")
        note(f"Critic verdict: {result.final_critique.verdict.value}", "success", Stage.CRITIC)

        spec = result.initial.problem_spec
        pipeline = result.initial.pipeline
        stage(Stage.EXPLAIN, StageStatus.ACTIVE)
        explain_report = explain_model(
            store,
            result.best_model,
            train_dataset_id=pipeline.modeling_dataset_id,
            task_is_classification=is_classification(spec.task_type),
            run_id=result.initial.run_id,
        )
        stage(Stage.EXPLAIN, StageStatus.DONE, f"top: {explain_report.importances[0].feature}")

        from mlteam.agents import read_dataset
        from mlteam.agents.insight_narrator import narrate_insights
        from mlteam.evaluation.results import list_experiment_outcomes

        insights = narrate_insights(
            client,
            store,
            problem=spec,
            evaluation=result.final_report,
            explainability=explain_report,
            run_id=result.initial.run_id,
        )

        features = [
            c
            for c in read_dataset(store, store.get(pipeline.modeling_dataset_id)).columns
            if c != spec.target_column
        ]
        stage(Stage.DEPLOY, StageStatus.ACTIVE)
        from mlteam.api.run_manager import _planned_experiment_ids

        build_deployment_bundle(
            store,
            model=result.best_model,
            profile=store.get(pipeline.train_profile_id),
            problem=spec,
            train_dataset_id=pipeline.modeling_dataset_id,
            feature_names=features,
            gate=AutoApproveGate(),
            run_id=result.initial.run_id,
            split_plan_id=pipeline.split_plan_id,
            outcomes=list_experiment_outcomes(store, result.initial.run_id),
            evaluation=result.final_report,
            explainability=explain_report,
            critique=result.final_critique,
            planned_experiment_ids=_planned_experiment_ids(store, pipeline),
        )
        stage(Stage.DEPLOY, StageStatus.DONE)

        stage(Stage.REPORT, StageStatus.ACTIVE)
        report = build_final_report(
            store,
            ReportInputs(
                problem=spec,
                model=result.best_model,
                evaluation=result.final_report,
                critique=result.final_critique,
                split_plan=store.get(pipeline.split_plan_id),
                goal_text="Identify customers likely to churn for proactive retention.",
                best_template_id=pipeline.best_template_id,
                best_experiment_id=result.best_experiment_id,
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
        note("Final report generated", "success", Stage.REPORT)

        from mlteam.api.run_manager import _record_ledger_outcome

        _record_ledger_outcome(store, spec=spec, pipeline=pipeline, best_model=result.best_model)

        record = registry.get(DEMO_UI_RUN_ID)
        if record is not None:
            record.result_run_id = result.initial.run_id
        registry.finish(
            DEMO_UI_RUN_ID,
            status=RunStatus.COMPLETED,
            final_report_id=report.meta.artifact_id,
        )
        _log.info("demo_run_seeded", report=report.meta.artifact_id)
        return DEMO_UI_RUN_ID
    except Exception as exc:  # a broken seed must not crash API startup
        _log.exception("demo_seed_failed")
        registry.finish(DEMO_UI_RUN_ID, status=RunStatus.FAILED, error=str(exc))
        return DEMO_UI_RUN_ID
