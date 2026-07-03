"""Agent tests over recorded fixtures — no live API anywhere."""

from __future__ import annotations

import json

import pytest

from mlteam.agents import (
    critique_evaluation,
    ingest_file,
    plan_experiments,
    profile_dataset,
    propose_cleaning_ops_llm,
    propose_features,
    understand_requirements,
)
from mlteam.contracts import (
    ArtifactRef,
    BaselineDelta,
    CleaningOpType,
    EvaluationReport,
    LeakageCheck,
    MetricSummary,
    TaskType,
    Verdict,
    new_artifact,
)
from mlteam.llm import FixtureTransport, LLMClient

RUN = "test-llm-agents"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


@pytest.fixture()
def churn_profile(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    return profile


# -- Requirement Understanding -------------------------------------------------


def test_requirement_agent_builds_problem_spec(guarded_store):
    client, _ = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
                "assumptions": ["churned=1 means left"],
            }
        )
    )
    spec = understand_requirements(
        client,
        guarded_store,
        goal_text="predict which customers will churn",
        dataset_columns=["tenure_months", "churned"],
        run_id=RUN,
    )
    assert spec.task_type is TaskType.BINARY_CLASSIFICATION
    assert spec.human_confirmed is False
    assert guarded_store.exists(spec.meta.artifact_id)


def test_requirement_agent_retries_on_bad_target(guarded_store):
    client, transport = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "nonexistent",
                "primary_metric": "roc_auc",
            }
        ),
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
            }
        ),
    )
    spec = understand_requirements(
        client,
        guarded_store,
        goal_text="predict churn",
        dataset_columns=["churned"],
        run_id=RUN,
    )
    assert spec.target_column == "churned"
    assert "not in the dataset" in transport.requests[1].user_content


def test_requirement_agent_rejects_metric_task_mismatch(guarded_store):
    client, transport = _client(
        json.dumps(
            {"task_type": "regression", "target_column": "price", "primary_metric": "roc_auc"}
        ),
        json.dumps({"task_type": "regression", "target_column": "price", "primary_metric": "rmse"}),
    )
    spec = understand_requirements(
        client,
        guarded_store,
        goal_text="predict house prices",
        dataset_columns=["price"],
        run_id=RUN,
    )
    assert spec.primary_metric == "rmse"


# -- Cleaning proposer ----------------------------------------------------------


def test_cleaning_proposer_returns_whitelisted_ops(churn_profile):
    client, _ = _client(
        json.dumps(
            {
                "ops": [
                    {
                        "op_type": "drop_column",
                        "column": "row_id",
                        "justification": "near-unique identifier",
                        "affected_columns": ["row_id"],
                    }
                ]
            }
        )
    )
    ops = propose_cleaning_ops_llm(client, churn_profile, target_column="churned")
    assert len(ops) == 1
    assert ops[0].op_type is CleaningOpType.DROP_COLUMN


def test_cleaning_proposer_rejects_learned_ops(churn_profile):
    client, transport = _client(
        json.dumps(
            {
                "ops": [
                    {
                        "op_type": "impute_median",
                        "columns": ["monthly_charges"],
                        "justification": "fill missing charges",
                    }
                ]
            }
        ),
        json.dumps({"ops": []}),
    )
    ops = propose_cleaning_ops_llm(client, churn_profile, target_column="churned")
    assert ops == ()
    assert "not executable" in transport.requests[1].user_content


def test_cleaning_proposer_protects_target(churn_profile):
    client, transport = _client(
        json.dumps(
            {
                "ops": [
                    {
                        "op_type": "drop_column",
                        "column": "churned",
                        "justification": "constant-ish",
                    }
                ]
            }
        ),
        json.dumps({"ops": []}),
    )
    propose_cleaning_ops_llm(client, churn_profile, target_column="churned")
    assert "target column" in transport.requests[1].user_content


# -- Feature proposer -------------------------------------------------------------


def _spec_artifact(store):
    from mlteam.contracts import ProblemSpec

    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )
    store.put(spec)
    return spec


def test_feature_proposer_builds_plan(guarded_store, churn_profile):
    spec = _spec_artifact(guarded_store)
    client, _ = _client(
        json.dumps(
            {
                "proposals": [
                    {
                        "name": "calls_per_month",
                        "op_type": "ratio",
                        "input_columns": ["num_support_calls", "tenure_months"],
                        "rationale": "support intensity normalized by tenure",
                    }
                ]
            }
        )
    )
    plan = propose_features(client, guarded_store, profile=churn_profile, problem=spec, run_id=RUN)
    assert plan.proposals[0].name == "calls_per_month"
    assert guarded_store.exists(plan.meta.artifact_id)


def test_feature_proposer_rejects_target_leak(guarded_store, churn_profile):
    spec = _spec_artifact(guarded_store)
    client, transport = _client(
        json.dumps(
            {
                "proposals": [
                    {
                        "name": "leaky",
                        "op_type": "ratio",
                        "input_columns": ["churned", "tenure_months"],
                        "rationale": "oops",
                    }
                ]
            }
        ),
        json.dumps({"proposals": []}),
    )
    propose_features(client, guarded_store, profile=churn_profile, problem=spec, run_id=RUN)
    assert "target leakage" in transport.requests[1].user_content


# -- Experiment planner --------------------------------------------------------------


def test_planner_enforces_baselines_first(guarded_store, churn_profile):
    spec = _spec_artifact(guarded_store)
    bad = json.dumps(
        {
            "experiments": [
                {"experiment_id": "e1", "template_id": "lightgbm_default", "is_baseline": False}
            ],
            "rationale": "skip baselines",
        }
    )
    good = json.dumps(
        {
            "experiments": [
                {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
            ],
            "rationale": "baselines first",
        }
    )
    client, transport = _client(bad, good)
    plan = plan_experiments(
        client,
        guarded_store,
        profile=churn_profile,
        problem=spec,
        n_rows=200,
        run_id=RUN,
    )
    assert plan.experiments[0].template_id == "dummy_baseline"
    assert plan.human_approved is False
    assert all(e.budget.wall_time_seconds > 0 for e in plan.experiments)
    assert "must start with dummy_baseline" in transport.requests[1].user_content


def test_planner_rejects_unknown_template(guarded_store, churn_profile):
    spec = _spec_artifact(guarded_store)
    client, transport = _client(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                    {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
                    {"experiment_id": "x", "template_id": "gpt_magic", "is_baseline": False},
                ],
                "rationale": "?",
            }
        ),
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                    {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
                ],
                "rationale": "registry only",
            }
        ),
    )
    plan = plan_experiments(
        client, guarded_store, profile=churn_profile, problem=spec, n_rows=200, run_id=RUN
    )
    assert len(plan.experiments) == 2
    assert "not in the registry" in transport.requests[1].user_content


# -- Critic ------------------------------------------------------------------------


def _evaluation_report(store, *, significant: bool, leakage_ok: bool = True):
    report = new_artifact(
        EvaluationReport,
        producer_agent="evaluator",
        run_id=RUN,
        model_ref=ArtifactRef(artifact_id="m" * 64, artifact_type="TrainedModel"),
        cv_metrics=(
            MetricSummary(name="roc_auc", fold_values=(0.7, 0.72, 0.71), mean=0.71, std=0.008),
        ),
        baseline_deltas=(
            BaselineDelta(
                baseline_name="dummy_baseline",
                metric="roc_auc",
                baseline_value=0.5,
                model_value=0.71,
                delta=0.21,
                significant=significant,
            ),
        ),
        leakage_checks=(
            LeakageCheck(name="near_perfect_features", passed=leakage_ok, details="-"),
        ),
        holdout_untouched=True,
    )
    store.put(report)
    return report


def test_critic_approve_stands_when_gates_pass(guarded_store):
    report = _evaluation_report(guarded_store, significant=True)
    client, _ = _client(json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []}))
    critique = critique_evaluation(
        client, guarded_store, report=report, primary_metric="roc_auc", run_id=RUN
    )
    assert critique.verdict is Verdict.APPROVE


def test_critic_cannot_approve_past_failed_dummy_gate(guarded_store):
    report = _evaluation_report(guarded_store, significant=False)
    client, _ = _client(json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []}))
    critique = critique_evaluation(
        client, guarded_store, report=report, primary_metric="roc_auc", run_id=RUN
    )
    assert critique.verdict is Verdict.ITERATE
    assert critique.iteration_actions  # actions synthesized from failed gates
    assert any(f.category == "gate_override" for f in critique.findings)


def test_critic_leakage_failure_forces_halt(guarded_store):
    report = _evaluation_report(guarded_store, significant=True, leakage_ok=False)
    client, _ = _client(json.dumps({"findings": [], "verdict": "approve", "iteration_actions": []}))
    critique = critique_evaluation(
        client, guarded_store, report=report, primary_metric="roc_auc", run_id=RUN
    )
    assert critique.verdict is Verdict.HALT


def test_critic_overcautious_halt_stands_but_is_audited(guarded_store):
    """LLM halts when gates only require iterate: the halt stands (never
    downgrade), but the report must carry an audit finding and actions."""
    report = _evaluation_report(guarded_store, significant=False)  # dummy gate fails
    client, _ = _client(json.dumps({"findings": [], "verdict": "halt", "iteration_actions": []}))
    critique = critique_evaluation(
        client, guarded_store, report=report, primary_metric="roc_auc", run_id=RUN
    )
    assert critique.verdict is Verdict.HALT
    assert any(f.category == "gate_note" for f in critique.findings)
    assert critique.iteration_actions  # never an unexplained dead end
