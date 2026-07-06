"""Phase 3: DecisionRecord — every gate answers WHY, not just WHAT.

Covers the draft-to-DecisionRecord mapping for both the requirement agent
(target choice) and the experiment planner (template selection + the plan's
negative space), and that Gate 1/2 payloads surface that reasoning. Also
proves fixtures recorded before these fields existed still replay clean —
the new draft fields are optional-with-defaults specifically for this.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlteam.agents import ingest_file, plan_experiments, profile_dataset, understand_requirements
from mlteam.contracts import ProblemSpec, TaskType, new_artifact
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import GateDecision, GateName, run_agentic_pipeline

RUN = "test-decision-records"
AGENTIC_FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "agentic_churn_run.json"


class _RecordingGate:
    """Captures every gate summary it's asked to approve, then auto-approves."""

    def __init__(self) -> None:
        self.summaries: dict[GateName, dict] = {}

    def request(self, gate: GateName, summary: dict) -> GateDecision:
        self.summaries[gate] = summary
        return GateDecision(approved=True, approver="test")


def _client(*texts: str) -> LLMClient:
    return LLMClient(FixtureTransport.from_texts(*texts), model="claude-sonnet-4-6")


def test_requirement_draft_maps_to_decision_record(guarded_store):
    client = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
                "assumptions": ["churned=1 means left"],
                "why_target": "column name and goal both reference churn directly",
                "target_alternatives": ["tenure_months"],
                "target_evidence": ["column literally named 'churned'"],
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
    assert spec.reasoning is not None
    assert "churned" in spec.reasoning.decision
    assert spec.reasoning.why == "column name and goal both reference churn directly"
    assert spec.reasoning.alternatives_considered == ("tenure_months",)
    assert spec.reasoning.evidence == ("column literally named 'churned'",)
    assert spec.reasoning.assumptions == ("churned=1 means left",)


def test_requirement_reasoning_defaults_safely_without_new_fields(guarded_store):
    """A fixture recorded before why_target/target_alternatives/target_evidence
    existed must still replay: the draft fields are optional-with-defaults."""
    client = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
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
    assert spec.reasoning is not None
    assert spec.reasoning.why == ""
    assert spec.reasoning.alternatives_considered == ()
    assert spec.reasoning.evidence == ()


def _spec_artifact(store):
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


def test_planner_draft_maps_excluded_templates_and_reasoning(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)
    client = _client(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                    {
                        "experiment_id": "b2",
                        "template_id": "lightgbm_default",
                        "is_baseline": True,
                    },
                ],
                "rationale": "small dataset, tree-based baselines are enough to start",
                "excluded_templates": [
                    {
                        "template_id": "svm_default",
                        "reason": "quadratic cost not worth it at this size",
                    },
                    {"template_id": "mlp_default", "reason": "too little data for a neural net"},
                ],
            }
        )
    )
    plan = plan_experiments(
        client, guarded_store, profile=profile, problem=spec, n_rows=200, run_id=RUN
    )
    assert len(plan.excluded_templates) == 2
    excluded_ids = {e.template_id for e in plan.excluded_templates}
    assert excluded_ids == {"svm_default", "mlp_default"}
    assert plan.reasoning is not None
    assert plan.reasoning.why == "small dataset, tree-based baselines are enough to start"
    assert len(plan.reasoning.alternatives_considered) == 2
    assert any("svm_default" in a for a in plan.reasoning.alternatives_considered)


def test_planner_reasoning_defaults_safely_without_excluded_templates(guarded_store, churn_csv):
    """A fixture recorded before excluded_templates existed must still replay."""
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)
    client = _client(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                    {
                        "experiment_id": "b2",
                        "template_id": "lightgbm_default",
                        "is_baseline": True,
                    },
                ],
                "rationale": "baselines only",
            }
        )
    )
    plan = plan_experiments(
        client, guarded_store, profile=profile, problem=spec, n_rows=200, run_id=RUN
    )
    assert plan.excluded_templates == ()
    assert plan.reasoning is not None
    assert plan.reasoning.alternatives_considered == ()


def test_gate_payloads_carry_reasoning_end_to_end(tmp_path):
    """Full agentic pipeline over a fixture recorded BEFORE reasoning fields
    existed: Gate 1 and Gate 2 summaries must still include the new keys
    (with empty inner content), proving old fixtures replay through the real
    orchestrator, not just the isolated agent calls above."""
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(AGENTIC_FIXTURE), model="claude-sonnet-4-6")
    gate = _RecordingGate()

    run_agentic_pipeline(
        store,
        csv_path,
        goal_text="Identify customers likely to churn so retention can call them first.",
        client=client,
        gate=gate,
        seed=42,
        n_folds=3,
    )

    spec_summary = gate.summaries[GateName.PROBLEM_SPEC]
    assert "reasoning" in spec_summary

    plan_summary = gate.summaries[GateName.TRAINING_PLAN]
    assert "excluded_templates" in plan_summary
    assert plan_summary["excluded_templates"] == []
