"""JUDGMENT_ROADMAP.md Task 3: capture the decision, raise the critic's bar.

The critic previously approved a model on "beats dummy significantly" alone
(the 2026-07-08 house-prices run: r^2=0.51, MAE ~32% of median price,
approved with no note that fitness for purpose was ever asked about).
ProblemSpec now carries decision_use/error_tolerance_note/human_baseline_note
(never invented — empty means the goal text genuinely didn't say), and the
critic computes a report-only relative_error_context gate plus a
deterministic (not LLM-trusted) fitness check: a stated-and-violated
tolerance forces the verdict away from approve; an absent tolerance with a
high relative error adds a high-severity finding without forcing anything.
"""

from __future__ import annotations

import json

from mlteam.agents.critic import compute_quantitative_gates, critique_evaluation
from mlteam.agents.requirement import understand_requirements
from mlteam.contracts import (
    BaselineDelta,
    ColumnProfile,
    DataProfile,
    EvaluationReport,
    MetricSummary,
    ProblemSpec,
    TaskType,
    Verdict,
    new_artifact,
)
from mlteam.llm import FixtureTransport, LLMClient

RUN = "test-fitness-for-purpose"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


def _metric(name: str, mean: float, std: float) -> MetricSummary:
    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def _profile_with_target_median(target_column: str, median: float) -> DataProfile:
    return new_artifact(
        DataProfile,
        producer_agent="test",
        run_id=RUN,
        dataset_ref={"artifact_id": "d", "artifact_type": "DatasetVersion"},
        columns=(
            ColumnProfile(
                name=target_column,
                dtype="Float64",
                cardinality=100,
                missing_pct=0.0,
                median=median,
            ),
        ),
    )


def _report_with_mae(mae: float) -> EvaluationReport:
    return new_artifact(
        EvaluationReport,
        producer_agent="test",
        run_id=RUN,
        model_ref={"artifact_id": "m", "artifact_type": "TrainedModel"},
        cv_metrics=(_metric("mae", mae, 0.01), _metric("rmse", mae * 1.3, 0.02)),
        holdout_untouched=True,
        # Beats the dummy floor significantly, so the pre-existing hard gate
        # never interferes with what these tests actually check.
        baseline_deltas=(
            BaselineDelta(
                baseline_name="dummy_baseline",
                metric="mae",
                baseline_value=mae * 2,
                model_value=mae,
                delta=mae,
                significant=True,
            ),
        ),
    )


def test_problem_spec_round_trips_fitness_fields():
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
        decision_use="set next month's inventory order",
        error_tolerance_note="within 10% for pricing",
        human_baseline_note="an analyst manually prices each listing",
    )
    assert spec.decision_use == "set next month's inventory order"
    assert spec.error_tolerance_note == "within 10% for pricing"
    assert spec.human_baseline_note == "an analyst manually prices each listing"


def test_problem_spec_fitness_fields_default_empty():
    """Fixtures/artifacts recorded before these fields existed still load,
    and the requirement agent's own default is empty, never invented."""
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
    )
    assert spec.decision_use == ""
    assert spec.error_tolerance_note == ""
    assert spec.human_baseline_note == ""


def test_requirement_extraction_populates_fitness_fields_when_stated(guarded_store):
    client, _ = _client(
        json.dumps(
            {
                "task_type": "regression",
                "target_column": "price",
                "primary_metric": "mae",
                "decision_use": "set the listing price shown to sellers",
                "error_tolerance_note": "within 10% for pricing",
                "human_baseline_note": "an agent manually appraises each listing",
            }
        )
    )
    spec = understand_requirements(
        client,
        guarded_store,
        goal_text="price listings within 10% for sellers, replacing manual appraisal",
        dataset_columns=["area_sqft", "price"],
        run_id=RUN,
    )
    assert spec.decision_use == "set the listing price shown to sellers"
    assert spec.error_tolerance_note == "within 10% for pricing"
    assert spec.human_baseline_note == "an agent manually appraises each listing"


def test_requirement_extraction_leaves_fitness_fields_empty_when_unstated(guarded_store):
    """The goal text says nothing about tolerance/decision/baseline — the
    agent must NOT invent plausible-sounding values for them."""
    client, _ = _client(
        json.dumps(
            {
                "task_type": "regression",
                "target_column": "price",
                "primary_metric": "mae",
            }
        )
    )
    spec = understand_requirements(
        client,
        guarded_store,
        goal_text="predict the sale price of a listing",
        dataset_columns=["area_sqft", "price"],
        run_id=RUN,
    )
    assert spec.decision_use == ""
    assert spec.error_tolerance_note == ""
    assert spec.human_baseline_note == ""


def test_relative_error_context_computes_documented_ratio():
    """MAE=25000, median target=100000 -> 25% relative error, hand-verified."""
    report = _report_with_mae(25_000.0)
    profile = _profile_with_target_median("price", 100_000.0)
    gates = compute_quantitative_gates(
        report, "mae", profile=profile, target_column="price"
    )
    gate = next(g for g in gates if g.name == "relative_error_context")
    assert gate.passed is True
    assert gate.hard is False
    assert "25.0%" in gate.detail


def test_relative_error_context_absent_without_profile():
    report = _report_with_mae(25_000.0)
    gates = compute_quantitative_gates(report, "mae")
    assert not any(g.name == "relative_error_context" for g in gates)


def test_critic_forces_non_approve_when_stated_tolerance_violated(guarded_store):
    """Stated tolerance 10%, actual relative error 25% -- a deterministic
    post-check, not the LLM's own judgment, must override an approve."""
    report = _report_with_mae(25_000.0)
    guarded_store.put(report)
    profile = _profile_with_target_median("price", 100_000.0)
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
        error_tolerance_note="within 10% for pricing",
    )
    client, _ = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique = critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="mae",
        run_id=RUN,
        profile=profile,
        problem=problem,
    )
    assert critique.verdict is not Verdict.APPROVE
    assert any(f.category == "fitness_for_purpose" for f in critique.findings)


def test_critic_flags_unknown_fitness_without_forcing_verdict(guarded_store):
    """No stated tolerance, but relative error (30%) exceeds the generous
    unstated-tolerance threshold (25%) -- a high-severity finding must
    appear, but the approve verdict may stand (beats-dummy remains a valid
    floor)."""
    report = _report_with_mae(30_000.0)
    guarded_store.put(report)
    profile = _profile_with_target_median("price", 100_000.0)
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
    )
    client, _ = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique = critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="mae",
        run_id=RUN,
        profile=profile,
        problem=problem,
    )
    assert critique.verdict is Verdict.APPROVE
    assert any(f.category == "fitness_for_purpose" for f in critique.findings)


def test_critic_no_fitness_finding_when_error_is_small(guarded_store):
    """Relative error well under the threshold and no stated tolerance --
    no fitness finding should be manufactured."""
    report = _report_with_mae(2_000.0)
    guarded_store.put(report)
    profile = _profile_with_target_median("price", 100_000.0)
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
    )
    client, _ = _client(
        json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []})
    )
    critique = critique_evaluation(
        client,
        guarded_store,
        report=report,
        primary_metric="mae",
        run_id=RUN,
        profile=profile,
        problem=problem,
    )
    assert critique.verdict is Verdict.APPROVE
    assert not any(f.category == "fitness_for_purpose" for f in critique.findings)
