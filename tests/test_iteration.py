"""Iteration loop tests over recorded fixtures — no live API."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.contracts import Verdict
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import AutoApproveGate, RunBudget, run_iterative_pipeline

FIXTURES = Path(__file__).parent / "fixtures" / "llm"


def tiny_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
    }


def _run(tmp_path, fixture_name: str, budget: RunBudget):
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(
        FixtureTransport.from_file(FIXTURES / fixture_name), model="claude-sonnet-4-6"
    )
    result = run_iterative_pipeline(
        store,
        csv_path,
        goal_text="reduce churn by calling at-risk customers",
        client=client,
        gate=AutoApproveGate(),
        budget=budget,
        seed=42,
        n_folds=3,
        hpo_trials=2,
        hpo_space_override=tiny_space,
    )
    return store, result


@pytest.fixture(scope="module")
def approved_run(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("iter-ok"),
        "iterative_churn_approved.json",
        RunBudget(max_iterations=3),
    )


def test_loop_stops_on_approval(approved_run):
    _, result = approved_run
    assert result.status == "approved"
    assert result.final_critique.verdict is Verdict.APPROVE
    assert len(result.iterations) == 1


def test_iteration_trained_revision(approved_run):
    _, result = approved_run
    record = result.iterations[0]
    # Audit Task 8: screening results now compete for selection, and on this
    # fixture screen-linear (linear_default) genuinely outscores every
    # planned lightgbm candidate. HPO is correctly SKIPPED this iteration --
    # linear_default has no vetted Optuna search space (SEARCH_SPACES only
    # covers tree-ensemble templates), so there is nothing for HPO to tune
    # on the current best. This is honest behavior, not a regression: before
    # Task 8, HPO always fired here only because a lightgbm candidate was
    # unconditionally "best" among the (screening-blind) planned set.
    assert not any(eid.startswith("hpo-") for eid in record.new_experiment_ids)
    assert "exp-lgbm-rev1" in record.new_experiment_ids
    assert record.verdict is Verdict.APPROVE


def test_screening_winner_is_selectable_even_when_not_hpo_tunable(approved_run):
    """The whole point of audit Task 8: a screening probe (here,
    screen-linear) can be the FINAL winner across the whole run, even
    though it was never in the LLM's plan and even though it can't be
    HPO-tuned -- selection is honest about which trained candidate is
    actually best, not restricted to the planner's candidate list."""
    _, result = approved_run
    assert result.best_experiment_id == "screen-linear"
    assert result.best_model.template_id == "linear_default"


def test_budget_snapshot_recorded(approved_run):
    _, result = approved_run
    snap = result.budget_snapshot
    assert snap["iterations_used"] == 1
    assert snap["llm_cost_usd"] > 0


def test_holdout_untouched_through_iterations(approved_run):
    store, result = approved_run
    assert result.final_report.holdout_untouched is True
    assert store.holdout_untouched() is True


def test_budget_exhaustion_stops_loop(tmp_path):
    _, result = _run(
        tmp_path,
        "iterative_churn_exhaust.json",
        RunBudget(max_iterations=1),
    )
    assert result.status == "budget_exhausted"
    assert "iteration budget" in result.stop_reason
    assert len(result.iterations) == 1
    assert result.final_critique.verdict is Verdict.ITERATE
