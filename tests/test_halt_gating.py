"""JUDGMENT_ROADMAP.md Task 0: a critic HALT verdict must block Gate 3
deployment by default, not merely get buried in a dossier a human has to
know to scroll to. Observed live (2026-07-08): halt at 16:32, Gate 3
offered and approved at 16:40, for a model whose every candidate had
negative R² -- nothing in the pipeline actually stopped a halted model
from reaching deployment sign-off.

These tests drive the REAL run_manager._execute with a monkeypatched
run_iterative_pipeline (the same pattern as
test_api.py::test_result_run_id_is_set_before_the_pipeline_runs_not_after)
returning a stub IterativeResult built from a genuinely trained
deterministic pipeline (run_tabular_pipeline) with only the critique
swapped to HALT -- so the model/evaluation/explainability plumbing
downstream of the halt check is exercised for real, not mocked away.
"""

from __future__ import annotations

import threading
import time

from mlteam.contracts import (
    CritiqueReport,
    TaskType,
    Verdict,
    new_artifact,
)
from mlteam.orchestration.agentic import AgenticResult
from mlteam.orchestration.iteration import IterativeResult
from mlteam.orchestration.pipeline import run_tabular_pipeline

RUN = "halt-gating-test"


def _halted_result(store, csv_path):
    """A genuinely trained PipelineResult (real model, real evaluation
    report) wrapped in an IterativeResult whose final_critique is HALT --
    everything except the critique verdict is real, produced work."""
    from mlteam.contracts import ProblemSpec

    pipeline_result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
    )
    best_key = next(
        k for k in pipeline_result.train_results if "dummy" not in k
    )
    best_model = pipeline_result.train_results[best_key].model

    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
        human_confirmed=True,
    )
    halt_critique = new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=RUN,
        evaluation_ref=pipeline_result.evaluation_report.ref(),
        verdict=Verdict.HALT,
    )
    # build_final_report's validate_run_membership (Task 2 of
    # REPORT_INTEGRITY_ROADMAP.md) checks every input artifact against the
    # store's run_artifacts membership table -- hand-built artifacts that
    # were never store.put() would correctly be rejected as not belonging
    # to this run.
    store.put(spec)
    store.put(halt_critique)
    agentic = AgenticResult(
        run_id=RUN,
        problem_spec=spec,
        pipeline=pipeline_result,
        critique=halt_critique,
        feature_plan_id=None,
        llm_cost_usd=0.0,
    )
    return IterativeResult(
        status="halted",
        stop_reason="critic halted",
        initial=agentic,
        iterations=(),
        final_report=pipeline_result.evaluation_report,
        final_critique=halt_critique,
        best_experiment_id=best_key,
        best_model=best_model,
    )


def _run_execute(
    tmp_path, monkeypatch, *, allow_deploy_on_halt: bool, registry=None
):
    import mlteam.api.run_manager as rm_module
    import mlteam.orchestration as orchestration_module
    from mlteam.api.registry import RunRegistry
    from mlteam.llm import FixtureTransport
    from synthetic import make_churn_frame

    reg = registry or RunRegistry()
    ui_run_id = "run-halt-gating"
    reg.create(ui_run_id, goal_text="predict churn", dataset_name="churn.csv")

    csv_path = tmp_path / "churn.csv"
    make_churn_frame(n_rows=200).write_csv(csv_path)
    store_root = tmp_path / "store"

    def stub_run_iterative_pipeline(store, _data_path, *_args, **_kwargs):
        return _halted_result(store, csv_path)

    monkeypatch.setattr(
        orchestration_module, "run_iterative_pipeline", stub_run_iterative_pipeline
    )

    rm_module._execute(
        ui_run_id=ui_run_id,
        data_path=csv_path,
        goal_text="predict churn",
        transport=FixtureTransport.from_texts("{}"),
        store_root=store_root,
        model="claude-sonnet-4-6",
        seed=42,
        n_folds=3,
        hpo_trials=1,
        max_iterations=1,
        max_llm_cost_usd=1.0,
        allow_deploy_on_halt=allow_deploy_on_halt,
        registry=reg,
    )
    return reg, ui_run_id, store_root


def test_halt_verdict_blocks_gate_3_by_default(tmp_path, monkeypatch):
    """No override: the deployment bundle must never be built, DEPLOY must
    show the blocking detail, and the final report must still be produced
    (the report is the evidence trail even for a halted run)."""
    from mlteam.api.events import Stage, StageStatus
    from mlteam.contracts import DeploymentBundle
    from mlteam.store import open_store

    reg, ui_run_id, store_root = _run_execute(
        tmp_path, monkeypatch, allow_deploy_on_halt=False
    )

    record = reg.get(ui_run_id)
    deploy_stage = next(s for s in record.stages if s.stage is Stage.DEPLOY)
    assert deploy_stage.status is StageStatus.SKIPPED
    assert "halt" in deploy_stage.detail.lower()

    store = open_store(store_root)
    bundles = [
        aid
        for aid in store.list_ids(DeploymentBundle.artifact_type())
        if store.is_in_run(aid, RUN, store.artifact_ids_for_run(RUN))
    ]
    assert bundles == [], "a halted run must never produce a DeploymentBundle without override"

    assert record.final_report_id is not None, "final report must still be built for a halted run"
    assert record.status.value == "completed"


def test_halt_verdict_with_override_offers_gate_3_and_stamps_the_warning(
    tmp_path, monkeypatch
):
    """With allow_deploy_on_halt=True, Gate 3 IS offered -- but its summary
    must carry the halt verdict and warning as TOP-LEVEL keys, not buried
    in the nested dossier, so a human approving it cannot miss what they
    are signing."""
    from mlteam.api.registry import RunRegistry

    reg = RunRegistry()
    thread = threading.Thread(
        target=_run_execute,
        kwargs=dict(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            allow_deploy_on_halt=True,
            registry=reg,
        ),
        daemon=True,
    )
    thread.start()

    ui_run_id = "run-halt-gating"
    deadline = time.monotonic() + 30
    pending_summary = None
    while time.monotonic() < deadline:
        for rid, gate_request in reg.pending_gates():
            if rid == ui_run_id:
                pending_summary = gate_request.summary
                break
        if pending_summary is not None:
            break
        time.sleep(0.2)
    assert pending_summary is not None, "Gate 3 was never offered despite the override"

    assert pending_summary["critic_verdict"] == "halt"
    assert "halt" in pending_summary["warning"].lower()

    assert reg.resolve_gate(ui_run_id, approved=True, notes="shipping anyway")
    thread.join(timeout=30)
    assert not thread.is_alive()

    record = reg.get(ui_run_id)
    assert record.status.value == "completed"
    assert record.final_report_id is not None
