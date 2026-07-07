"""End-to-end agentic pipeline over the recorded fixture run — no live API."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.contracts import Verdict
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import (
    AutoApproveGate,
    GateDecision,
    GateName,
    GateRejectedError,
    run_agentic_pipeline,
)

FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "agentic_churn_run.json"


class RejectingGate:
    def request(self, gate: GateName, summary: dict) -> GateDecision:
        return GateDecision(approved=False, notes="not today")


@pytest.fixture(scope="module")
def agentic_run(tmp_path_factory):
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = tmp_path_factory.mktemp("agentic")
    csv_path = tmp / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(FIXTURE), model="claude-sonnet-4-6")
    result = run_agentic_pipeline(
        store,
        csv_path,
        goal_text="Identify customers likely to churn so retention can call them first.",
        client=client,
        gate=AutoApproveGate(),
        seed=42,
        n_folds=3,
    )
    return store, result


def test_problem_spec_confirmed_after_gate1(agentic_run):
    _, result = agentic_run
    assert result.problem_spec.human_confirmed is True
    assert result.problem_spec.target_column == "churned"


def test_llm_cleaning_ops_were_executed(agentic_run):
    store, result = agentic_run
    manifest = store.get(result.pipeline.cleaning_manifest_id)
    op_types = {op.op_type.value for op in manifest.ops}
    assert "drop_column" in op_types and "drop_duplicate_rows" in op_types
    assert all(op.justification for op in manifest.ops)


def test_feature_plan_stored(agentic_run):
    store, result = agentic_run
    assert result.feature_plan_id is not None
    plan = store.get(result.feature_plan_id)
    assert len(plan.proposals) == 3


def test_llm_experiment_plan_trained_extra_candidate(agentic_run):
    _, result = agentic_run
    # Fixture plan: dummy + lgbm default + tuned lgbm candidate (same template,
    # distinct experiment ids — results must not collide). Screening probes
    # (screen-dummy/screen-linear/screen-lightgbm) are ALSO present: audit
    # Task 8 merges them into train_results so they compete for selection
    # instead of being informative-only evidence the planner reads but the
    # selector ignores.
    assert result.pipeline.experiment_plan_id
    assert set(result.pipeline.train_results) == {
        "exp-dummy",
        "exp-lgbm-default",
        "exp-lgbm-tuned",
        "screen-dummy",
        "screen-linear",
        "screen-lightgbm",
    }
    # On this fixture, screen-linear (roc_auc 0.772) genuinely outscores
    # every planned candidate including exp-lgbm-default (0.769) -- this is
    # the HONEST winner, not a regression: before Task 8, a real fixture
    # figure was inaccessible for selection purely because it was trained
    # a few lines earlier in the pipeline than the planned candidates.
    assert result.pipeline.best_template_id == "linear_default"


def test_critique_produced_and_holdout_untouched(agentic_run):
    store, result = agentic_run
    assert result.critique.verdict in (Verdict.APPROVE, Verdict.ITERATE)
    assert result.pipeline.evaluation_report.holdout_untouched is True
    assert store.holdout_untouched()


def test_llm_cost_tracked(agentic_run):
    _, result = agentic_run
    assert result.llm_cost_usd > 0


def test_gate_rejection_stops_pipeline(tmp_path):
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(FIXTURE), model="claude-sonnet-4-6")
    with pytest.raises(GateRejectedError, match="gate1_problem_spec"):
        run_agentic_pipeline(
            store,
            csv_path,
            goal_text="predict churn",
            client=client,
            gate=RejectingGate(),
            n_folds=3,
        )
