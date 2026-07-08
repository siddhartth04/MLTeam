"""Iteration loop tests over recorded fixtures — no live API."""

from __future__ import annotations

import json
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


# -- JUDGMENT_ROADMAP.md Task 4: evidence-cited revisions + stop verb -------
# revise_experiment_plan unit-tested directly (same pattern as
# test_critic_field_evidence.py/test_decision_records.py) -- faster and more
# targeted than a full run_iterative_pipeline fixture for scenarios that are
# really about revise_experiment_plan's own grounding/stop behavior.


def _revision_client(*texts: str):
    from mlteam.llm import FixtureTransport, LLMClient

    return LLMClient(FixtureTransport.from_texts(*texts), model="claude-sonnet-4-6")


def _problem_spec():
    from mlteam.contracts import ProblemSpec, TaskType, new_artifact

    return new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id="test-revision",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )


def test_revision_drops_experiment_citing_nonexistent_id():
    """A cited experiment_id that isn't in this run's results must be
    dropped with a warning -- the run continues with whatever remains,
    not a hard failure of the whole revision."""
    from mlteam.agents.experiment_planner import revise_experiment_plan

    client = _revision_client(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "exp-ghost",
                        "template_id": "lightgbm_default",
                        "config": {},
                        "cited_experiment_ids": ["exp-that-never-ran"],
                        "rationale": "addresses the capacity finding",
                    },
                    {
                        "experiment_id": "exp-real",
                        "template_id": "lightgbm_default",
                        "config": {"n_estimators": 50},
                        "cited_experiment_ids": ["exp-lgbm"],
                        "rationale": "builds on exp-lgbm per the capacity finding",
                    },
                ],
                "rationale": "two candidates addressing capacity",
            }
        )
    )
    result = revise_experiment_plan(
        client,
        problem=_problem_spec(),
        critique_actions=("increase model capacity",),
        prior_results=[],
        prior_experiment_ids={"exp-dummy", "exp-lgbm"},
        n_rows=500,
    )
    ids = {s.experiment_id for s in result.specs}
    assert ids == {"exp-real"}
    assert result.stop is False


def test_revision_drops_experiment_with_no_citation_at_all():
    """No cited_experiment_ids AND no cited_finding_categories -- ungrounded,
    dropped regardless of how plausible the free-text rationale reads."""
    from mlteam.agents.experiment_planner import revise_experiment_plan

    client = _revision_client(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "exp-unfounded",
                        "template_id": "lightgbm_default",
                        "config": {},
                        "rationale": "this sounds like a good idea",
                    }
                ],
                "rationale": "one candidate",
            }
        )
    )
    result = revise_experiment_plan(
        client,
        problem=_problem_spec(),
        critique_actions=("increase model capacity",),
        prior_results=[],
        prior_experiment_ids={"exp-dummy", "exp-lgbm"},
        n_rows=500,
    )
    assert result.specs == ()


def test_revision_stop_true_ends_the_loop_with_reason():
    """The explicit stop verb: zero experiments, stop=true, a concrete
    reason -- the caller (run_iterative_pipeline) must break on this."""
    from mlteam.agents.experiment_planner import revise_experiment_plan

    client = _revision_client(
        json.dumps(
            {
                "experiments": [],
                "rationale": "no viable candidates remain",
                "stop": True,
                "stop_reason": "last two iterations moved roc_auc by <0.001; further tuning "
                "unlikely to help",
            }
        )
    )
    result = revise_experiment_plan(
        client,
        problem=_problem_spec(),
        critique_actions=("increase model capacity",),
        prior_results=[],
        prior_experiment_ids={"exp-dummy", "exp-lgbm"},
        n_rows=500,
    )
    assert result.specs == ()
    assert result.stop is True
    assert "further tuning unlikely to help" in result.stop_reason


def test_revision_target_transform_carries_through_to_training(guarded_store):
    """A revision that sets target_transform="log1p" on a new spec must
    produce a TrainedModel that actually carries it -- ties Task 2's
    wrap_target_transform plumbing to the iteration loop, not just to
    revise_experiment_plan's own return value."""
    import numpy as np
    import polars as pl

    from mlteam.agents.data_io import store_dataframe
    from mlteam.agents.experiment_planner import revise_experiment_plan
    from mlteam.agents.trainer import compute_env_hash, train_experiment
    from mlteam.contracts import DatasetRole, ProblemSpec, TaskType, new_artifact

    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(1, 10, n)
    price = np.expm1(2.0 + 0.3 * x + rng.normal(0, 0.2, n))
    df = pl.DataFrame({"feature": x, "price": price})
    dataset = store_dataframe(
        guarded_store,
        df,
        role=DatasetRole.TRAIN,
        producer_agent="test",
        run_id="test-revision-transform",
        target_column="price",
    )

    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id="test-revision-transform",
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="mae",
    )
    client = _revision_client(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "exp-log1p-rev",
                        "template_id": "lightgbm_default",
                        "config": {"n_estimators": 30},
                        "target_transform": "log1p",
                        "cited_experiment_ids": ["exp-lgbm"],
                        "rationale": "profile flagged target_transform_recommended",
                    }
                ],
                "rationale": "apply the recommended log1p transform",
            }
        )
    )
    result = revise_experiment_plan(
        client,
        problem=problem,
        critique_actions=("apply target transform",),
        prior_results=[],
        prior_experiment_ids={"exp-lgbm"},
        n_rows=n,
    )
    assert len(result.specs) == 1
    assert result.specs[0].target_transform == "log1p"

    train_result = train_experiment(
        guarded_store,
        dataset.meta.artifact_id,
        result.specs[0],
        task_type=TaskType.REGRESSION,
        run_id="test-revision-transform",
        seed=42,
        n_folds=3,
        env_hash=compute_env_hash(),
    )
    assert train_result.model.target_transform == "log1p"
