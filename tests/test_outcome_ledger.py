"""Phase 7: outcome ledger — cross-run experience as a PRIOR, never a pruner.

The planner used to start cold every run. These tests prove: the ledger
round-trips write/read correctly and buckets by data scale; the planner's
payload includes matching priors when a ledger exists; and — the safety
property that matters most — the baselines-first validator still rejects
a plan missing the mandatory baselines even when priors are present, so a
prior can never be used to skip evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlteam.agents import ingest_file, plan_experiments, profile_dataset
from mlteam.contracts import ProblemSpec, TaskType, new_artifact
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.store.ledger import matching_priors, read_ledger, record_outcome

RUN = "test-ledger"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


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


def test_ledger_write_read_roundtrip(tmp_path: Path):
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=500,
        n_features=10,
        categorical_share=0.3,
        winner_template="lightgbm_default",
        margin_vs_second=0.05,
        primary_metric="roc_auc",
    )
    entries = read_ledger(tmp_path)
    assert len(entries) == 1
    assert entries[0]["winner_template"] == "lightgbm_default"
    assert entries[0]["task_type"] == "binary_classification"
    assert entries[0]["categorical_share"] == 0.3
    assert entries[0]["margin_vs_second"] == 0.05


def test_ledger_is_append_only_across_multiple_writes(tmp_path: Path):
    for i in range(3):
        record_outcome(
            tmp_path,
            task_type="regression",
            n_rows=1000,
            n_features=5,
            categorical_share=0.0,
            winner_template=f"template-{i}",
            margin_vs_second=None,
            primary_metric="rmse",
        )
    entries = read_ledger(tmp_path)
    assert len(entries) == 3
    assert [e["winner_template"] for e in entries] == [
        "template-0",
        "template-1",
        "template-2",
    ]


def test_matching_priors_filters_by_task_type_and_scale(tmp_path: Path):
    # Same task, similar scale -> should match.
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=500,
        n_features=8,
        categorical_share=0.2,
        winner_template="catboost_default",
        margin_vs_second=0.1,
        primary_metric="roc_auc",
    )
    # Different task type -> must NOT match.
    record_outcome(
        tmp_path,
        task_type="regression",
        n_rows=500,
        n_features=8,
        categorical_share=0.2,
        winner_template="linear_default",
        margin_vs_second=0.02,
        primary_metric="rmse",
    )
    # Same task, wildly different scale (bucket differs by > 1) -> must NOT match.
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=5_000_000,
        n_features=8,
        categorical_share=0.2,
        winner_template="xgboost_default",
        margin_vs_second=0.15,
        primary_metric="roc_auc",
    )
    priors = matching_priors(tmp_path, task_type="binary_classification", n_rows=600)
    assert len(priors) == 1
    assert priors[0]["winner_template"] == "catboost_default"


def test_matching_priors_caps_at_max_entries_most_recent_first(tmp_path: Path):
    for i in range(8):
        record_outcome(
            tmp_path,
            task_type="binary_classification",
            n_rows=500,
            n_features=8,
            categorical_share=0.2,
            winner_template=f"template-{i}",
            margin_vs_second=0.01 * i,
            primary_metric="roc_auc",
        )
    priors = matching_priors(tmp_path, task_type="binary_classification", n_rows=500)
    assert len(priors) == 5
    # Most recent first.
    assert priors[0]["winner_template"] == "template-7"
    assert priors[-1]["winner_template"] == "template-3"


def test_matching_priors_empty_when_no_ledger_exists(tmp_path: Path):
    assert matching_priors(tmp_path, task_type="binary_classification", n_rows=500) == []


def test_matching_priors_filters_by_primary_metric(tmp_path: Path):
    """margin_vs_second is only comparable within the same metric's scale
    (audit L3) — an roc_auc margin and a pr_auc margin must not be mixed
    into the same priors list even when task_type and scale both match."""
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=500,
        n_features=8,
        categorical_share=0.2,
        winner_template="catboost_default",
        margin_vs_second=0.1,
        primary_metric="roc_auc",
    )
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=500,
        n_features=8,
        categorical_share=0.2,
        winner_template="xgboost_default",
        margin_vs_second=0.2,
        primary_metric="pr_auc",
    )
    priors = matching_priors(
        tmp_path, task_type="binary_classification", n_rows=500, primary_metric="roc_auc"
    )
    assert len(priors) == 1
    assert priors[0]["winner_template"] == "catboost_default"

    priors_pr_auc = matching_priors(
        tmp_path, task_type="binary_classification", n_rows=500, primary_metric="pr_auc"
    )
    assert len(priors_pr_auc) == 1
    assert priors_pr_auc[0]["winner_template"] == "xgboost_default"

    # Omitting primary_metric keeps the old (pre-L3) behavior of matching both.
    priors_unfiltered = matching_priors(tmp_path, task_type="binary_classification", n_rows=500)
    assert len(priors_unfiltered) == 2


def test_planner_payload_includes_matching_priors(guarded_store, churn_csv, tmp_path: Path):
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=200,
        n_features=6,
        categorical_share=0.3,
        winner_template="catboost_default",
        margin_vs_second=0.08,
        primary_metric="roc_auc",
    )
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)
    client, transport = _client(
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
                "rationale": "priors favor catboost but starting with baselines",
            }
        )
    )
    plan_experiments(
        client,
        guarded_store,
        profile=profile,
        problem=spec,
        n_rows=200,
        run_id=RUN,
        store_root=tmp_path,
    )
    sent = json.loads(transport.requests[0].user_content)
    assert "prior_outcomes" in sent
    assert len(sent["prior_outcomes"]) == 1
    assert sent["prior_outcomes"][0]["winner_template"] == "catboost_default"


def test_planner_payload_empty_priors_without_store_root(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)
    client, transport = _client(
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
                "rationale": "no priors available",
            }
        )
    )
    plan_experiments(
        client, guarded_store, profile=profile, problem=spec, n_rows=200, run_id=RUN
    )
    sent = json.loads(transport.requests[0].user_content)
    assert sent["prior_outcomes"] == []


def test_validator_rejects_baseline_free_plan_even_with_priors(
    guarded_store, churn_csv, tmp_path: Path
):
    """The safety property: priors may reorder candidates, but the plan
    validator still enforces baselines-first regardless of what the LLM
    (possibly swayed by a strong prior) tries to submit."""
    record_outcome(
        tmp_path,
        task_type="binary_classification",
        n_rows=200,
        n_features=6,
        categorical_share=0.3,
        winner_template="catboost_default",
        margin_vs_second=0.30,  # a very strong prior — must still not skip baselines
        primary_metric="roc_auc",
    )
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)

    # First attempt skips baselines entirely (as if swayed by the strong prior);
    # the post-validator must reject it and force a retry.
    bad = json.dumps(
        {
            "experiments": [
                {
                    "experiment_id": "e1",
                    "template_id": "catboost_default",
                    "is_baseline": False,
                }
            ],
            "rationale": "prior strongly favors catboost, skipping baselines",
        }
    )
    good = json.dumps(
        {
            "experiments": [
                {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                {"experiment_id": "b2", "template_id": "lightgbm_default", "is_baseline": True},
                {"experiment_id": "e1", "template_id": "catboost_default", "is_baseline": False},
            ],
            "rationale": "baselines first, then the prior-favored candidate",
        }
    )
    client, transport = _client(bad, good)
    plan = plan_experiments(
        client,
        guarded_store,
        profile=profile,
        problem=spec,
        n_rows=200,
        run_id=RUN,
        store_root=tmp_path,
    )
    assert plan.experiments[0].template_id == "dummy_baseline"
    assert plan.experiments[1].template_id == "lightgbm_default"
    assert len(transport.requests) == 2  # the bad draft was rejected and retried once

    # The contract-level validator (independent of the agent's post_validate
    # closure) must also reject a baseline-free plan directly.
    from pydantic import ValidationError

    from mlteam.contracts import BudgetEstimate, ExperimentPlan, ExperimentSpec

    with pytest.raises(ValidationError, match="baseline"):
        ExperimentPlan(
            meta=plan.meta,
            experiments=(
                ExperimentSpec(
                    experiment_id="e1",
                    template_id="catboost_default",
                    budget=BudgetEstimate(wall_time_seconds=10),
                    is_baseline=False,
                ),
            ),
            rationale="no baselines",
        )
