"""Phase 10: the Critic sees the whole field, not just best-vs-dummy.

Before this, the Critic judged only the winner against the dummy floor — it
could not compare competitors, audit whether every planned experiment (incl.
screening) actually reached a terminal state, or put a number on how close
the winner and its nearest competitor really are. These tests prove the
deterministic field-check math (audit hole detection, pooled-std margin with
hand-computed numbers), that the critic's payload carries the full outcome
table + exclusions, and — the safety property that matters most — that an
incomplete experiment field is a HARD gate the critic cannot approve past.
"""

from __future__ import annotations

import json

from mlteam.agents.critic import compute_quantitative_gates, critique_evaluation
from mlteam.contracts import (
    BaselineDelta,
    BudgetEstimate,
    EvaluationReport,
    ExcludedTemplate,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentState,
    MetricSummary,
    Verdict,
    new_artifact,
)
from mlteam.evaluation.field_checks import (
    audit_terminal_states,
    compute_cost_census,
    compute_failure_census,
    compute_winner_margin,
)
from mlteam.llm import FixtureTransport, LLMClient

RUN = "test-critic-field"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


def _metric(name: str, mean: float, std: float) -> MetricSummary:
    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def _outcome(
    experiment_id: str,
    template_id: str,
    *,
    state: ExperimentState = ExperimentState.COMPLETED,
    roc_auc: tuple[float, float] | None = None,
    error: str | None = None,
    wall_time_seconds: float = 1.0,
) -> ExperimentOutcome:
    return new_artifact(
        ExperimentOutcome,
        producer_agent="test",
        run_id=RUN,
        experiment_id=experiment_id,
        template_id=template_id,
        state=state,
        cv_metrics=(_metric("roc_auc", *roc_auc),) if roc_auc is not None else (),
        wall_time_seconds=wall_time_seconds,
        error=error,
    )


# -- field_checks: pure math, hand-verified ----------------------------------


def test_audit_terminal_states_detects_holes():
    outcomes = (
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.5, 0.01)),
        _outcome("exp-lgbm", "lightgbm_default", roc_auc=(0.8, 0.01)),
    )
    audit = audit_terminal_states(("exp-dummy", "exp-lgbm", "exp-catboost"), outcomes)
    assert not audit.complete
    assert audit.missing_ids == ("exp-catboost",)


def test_audit_terminal_states_complete_when_all_present():
    outcomes = (_outcome("exp-dummy", "dummy_baseline", roc_auc=(0.5, 0.01)),)
    audit = audit_terminal_states(("exp-dummy",), outcomes)
    assert audit.complete
    assert audit.missing_ids == ()


def test_failure_census_counts_by_state():
    outcomes = (
        _outcome("a", "t1", roc_auc=(0.5, 0.01)),
        _outcome("b", "t2", state=ExperimentState.FAILED, error="boom"),
        _outcome("c", "t3", state=ExperimentState.FAILED, error="also boom"),
        _outcome("d", "t4", state=ExperimentState.SKIPPED),
    )
    census = compute_failure_census(outcomes)
    assert census.completed == 1
    assert census.failed == 2
    assert census.skipped == 1
    assert census.failure_reasons == ("boom", "also boom")


def test_winner_margin_hand_computed_numbers():
    """winner=0.90±0.02, runner-up=0.85±0.02, dummy=0.50±0.01 (roc_auc, higher
    is better). Pooled std for winner-vs-runner-up = sqrt((0.02^2+0.02^2)/2)
    = 0.02, so margin_in_stds = 0.05 / 0.02 = 2.5 exactly."""
    outcomes = (
        _outcome("exp-winner", "lightgbm_default", roc_auc=(0.90, 0.02)),
        _outcome("exp-runnerup", "catboost_default", roc_auc=(0.85, 0.02)),
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.50, 0.01)),
    )
    margin = compute_winner_margin(outcomes, "roc_auc", "exp-winner", "exp-dummy")
    assert margin.runner_up_template == "catboost_default"
    assert abs(margin.vs_runner_up_raw - 0.05) < 1e-9
    assert abs(margin.vs_runner_up_stds - 2.5) < 1e-9
    assert abs(margin.vs_dummy_raw - 0.40) < 1e-9


def test_winner_margin_near_tie_below_half_std():
    """winner=0.86±0.02, runner-up=0.85±0.02 -> raw=0.01, pooled_std=0.02,
    margin_in_stds = 0.5 exactly -- the tie-detection boundary."""
    outcomes = (
        _outcome("exp-winner", "lightgbm_default", roc_auc=(0.86, 0.02)),
        _outcome("exp-runnerup", "catboost_default", roc_auc=(0.85, 0.02)),
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.50, 0.01)),
    )
    margin = compute_winner_margin(outcomes, "roc_auc", "exp-winner", "exp-dummy")
    assert abs(margin.vs_runner_up_stds - 0.5) < 1e-9


def test_winner_margin_zero_pooled_std_is_none_not_a_crash():
    outcomes = (
        _outcome("exp-winner", "lightgbm_default", roc_auc=(0.90, 0.0)),
        _outcome("exp-runnerup", "catboost_default", roc_auc=(0.85, 0.0)),
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.50, 0.0)),
    )
    margin = compute_winner_margin(outcomes, "roc_auc", "exp-winner", "exp-dummy")
    assert margin.vs_runner_up_stds is None
    assert abs(margin.vs_runner_up_raw - 0.05) < 1e-9


def test_winner_margin_no_runner_up_when_only_dummy_present():
    outcomes = (
        _outcome("exp-winner", "lightgbm_default", roc_auc=(0.90, 0.02)),
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.50, 0.01)),
    )
    margin = compute_winner_margin(outcomes, "roc_auc", "exp-winner", "exp-dummy")
    assert margin.runner_up_template is None
    assert margin.vs_runner_up_raw is None
    assert margin.vs_dummy_raw is not None


def test_cost_census_flags_over_estimate():
    outcomes = (
        _outcome("a", "t1", roc_auc=(0.5, 0.01), wall_time_seconds=30.0),
        _outcome("b", "t2", roc_auc=(0.8, 0.01), wall_time_seconds=40.0),
    )
    specs = (
        ExperimentSpec(
            experiment_id="a", template_id="t1", budget=BudgetEstimate(wall_time_seconds=20.0)
        ),
        ExperimentSpec(
            experiment_id="b", template_id="t2", budget=BudgetEstimate(wall_time_seconds=20.0)
        ),
    )
    cost = compute_cost_census(outcomes, specs)
    assert cost.total_wall_time_seconds == 70.0
    assert cost.estimated_total_seconds == 40.0
    assert cost.over_estimate is True


# -- critic hard gate ---------------------------------------------------------


def test_incomplete_experiment_field_is_a_hard_gate():
    gates = compute_quantitative_gates(
        _minimal_evaluation_report(), "roc_auc", missing_outcome_ids=("exp-catboost",)
    )
    gate = next(g for g in gates if g.name == "complete_experiment_field")
    assert gate.hard is True
    assert gate.passed is False


def test_complete_field_passes_when_no_holes():
    gates = compute_quantitative_gates(_minimal_evaluation_report(), "roc_auc")
    gate = next(g for g in gates if g.name == "complete_experiment_field")
    assert gate.passed is True


def _minimal_evaluation_report() -> EvaluationReport:
    return new_artifact(
        EvaluationReport,
        producer_agent="test",
        run_id=RUN,
        model_ref={"artifact_id": "m", "artifact_type": "TrainedModel"},
        cv_metrics=(_metric("roc_auc", 0.9, 0.02),),
        holdout_untouched=True,
        # Beats the dummy floor significantly -- otherwise the PRE-EXISTING
        # beats_dummy_significantly hard gate fails regardless of what this
        # test is actually checking (the terminal-state-audit gate).
        baseline_deltas=(
            BaselineDelta(
                baseline_name="dummy_baseline",
                metric="roc_auc",
                baseline_value=0.5,
                model_value=0.9,
                delta=0.4,
                significant=True,
            ),
        ),
    )


def test_critic_payload_includes_outcome_table_and_exclusions(guarded_store):
    report = _minimal_evaluation_report()
    guarded_store.put(report)
    outcomes = (
        _outcome("exp-dummy", "dummy_baseline", roc_auc=(0.5, 0.01)),
        _outcome("exp-lgbm", "lightgbm_default", roc_auc=(0.9, 0.02)),
    )
    excluded = (ExcludedTemplate(template_id="svm_default", reason="too slow above 20k rows"),)
    client, transport = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="roc_auc",
        run_id=RUN,
        outcomes=outcomes,
        planned_experiment_ids=("exp-dummy", "exp-lgbm"),
        excluded_templates=excluded,
        winner_experiment_id="exp-lgbm",
        dummy_experiment_id="exp-dummy",
    )
    sent = json.loads(transport.requests[0].user_content)
    assert len(sent["outcome_table"]) == 2
    assert sent["outcome_table"][0]["experiment_id"] in ("exp-dummy", "exp-lgbm")
    assert sent["excluded_templates"] == [
        {"template_id": "svm_default", "reason": "too slow above 20k rows"}
    ]
    assert sent["winner_margin"] is not None
    assert sent["terminal_state_audit"]["complete"] is True


def test_critic_forces_halt_on_incomplete_field(guarded_store):
    report = _minimal_evaluation_report()
    guarded_store.put(report)
    client, transport = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique = critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="roc_auc",
        run_id=RUN,
        outcomes=(_outcome("exp-dummy", "dummy_baseline", roc_auc=(0.9, 0.02)),),
        planned_experiment_ids=("exp-dummy", "exp-lgbm-that-never-ran"),
    )
    assert critique.verdict == Verdict.HALT
    assert any(f.category == "gate_override" for f in critique.findings)


def test_critic_approves_normally_when_field_is_complete(guarded_store):
    report = _minimal_evaluation_report()
    guarded_store.put(report)
    client, transport = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique = critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="roc_auc",
        run_id=RUN,
        outcomes=(_outcome("exp-dummy", "dummy_baseline", roc_auc=(0.9, 0.02)),),
        planned_experiment_ids=("exp-dummy",),
    )
    assert critique.verdict == Verdict.APPROVE
