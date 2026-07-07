"""Phase 9: adaptive two-round planning (screen -> contend).

The planner used to spend all its intelligence BEFORE any experiment ran --
it planned once, blind to results. These tests prove: the cheap screening
trio (dummy/linear/lightgbm) trains and persists real ExperimentOutcomes
before the plan is drafted; the planner's payload carries those results;
an identical template+config+dataset+seed+env_hash spec reuses the
screening run's model instead of retraining it; and the Gate 2 summary
surfaces the screening table so a human sees the evidence the plan was
conditioned on.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mlteam.agents import ingest_file, plan_experiments, profile_dataset
from mlteam.contracts import (
    BudgetEstimate,
    ExperimentSpec,
    ExperimentState,
    ProblemSpec,
    TaskType,
    new_artifact,
)
from mlteam.evaluation.results import list_experiment_outcomes
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration.agentic import _run_screening
from mlteam.orchestration.pipeline import _train_all
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-adaptive-planning"


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


def _spec_artifact(store, task_type=TaskType.BINARY_CLASSIFICATION, primary_metric="roc_auc"):
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=task_type,
        target_column="churned",
        primary_metric=primary_metric,
    )
    store.put(spec)
    return spec


def test_screening_persists_real_outcomes(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    summaries, train_results = _run_screening(
        guarded_store,
        dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        n_rows=dataset.n_rows,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
        mlflow_enabled=False,
    )
    assert len(summaries) == 3
    templates = {r["template"] for r in summaries}
    assert templates == {"dummy_baseline", "linear_default", "lightgbm_default"}
    for r in summaries:
        assert r["state"] == "completed"
        assert r["primary_metric_mean"] is not None
        assert r["wall_time_seconds"] >= 0

    # Audit Task 8: _run_screening also returns the actual TrainResults,
    # keyed by experiment_id, so screening models can compete for selection
    # alongside planned candidates instead of being informative-only.
    assert set(train_results) == {"screen-dummy", "screen-linear", "screen-lightgbm"}
    assert all(r.model.template_id in templates for r in train_results.values())

    outcomes = list_experiment_outcomes(guarded_store, RUN)
    screen_outcomes = [o for o in outcomes if o.experiment_id.startswith("screen-")]
    assert len(screen_outcomes) == 3
    assert all(o.state is ExperimentState.COMPLETED for o in screen_outcomes)
    assert all(o.model_ref is not None for o in screen_outcomes)


def test_train_all_reuses_identical_spec_without_retraining(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    common = dict(
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="fixed-env",
        mlflow_enabled=False,
    )
    screen_spec = ExperimentSpec(
        experiment_id="screen-dummy",
        template_id="dummy_baseline",
        budget=BudgetEstimate(wall_time_seconds=2),
        is_baseline=True,
    )
    _train_all(guarded_store, dataset.meta.artifact_id, (screen_spec,), **common)

    real_spec = ExperimentSpec(
        experiment_id="exp-dummy",  # different id, SAME template + config
        template_id="dummy_baseline",
        budget=BudgetEstimate(wall_time_seconds=2),
        is_baseline=True,
    )
    with patch(
        "mlteam.orchestration.pipeline.train_experiment",
        side_effect=AssertionError("should have reused, not retrained"),
    ):
        results, skipped = _train_all(
            guarded_store, dataset.meta.artifact_id, (real_spec,), **common
        )
    assert skipped == ()
    assert "exp-dummy" in results

    outcomes = {o.experiment_id: o for o in list_experiment_outcomes(guarded_store, RUN)}
    assert outcomes["exp-dummy"].wall_time_seconds == 0.0
    # Both outcomes point at the exact same trained model artifact.
    reused_id = outcomes["exp-dummy"].model_ref.artifact_id
    screened_id = outcomes["screen-dummy"].model_ref.artifact_id
    assert reused_id == screened_id


def test_train_all_retrains_when_config_differs(guarded_store, churn_csv):
    """A different config must NOT be treated as reusable -- it is a
    genuinely different model even with the same template."""
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    common = dict(
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="fixed-env",
        mlflow_enabled=False,
    )
    screen_spec = ExperimentSpec(
        experiment_id="screen-lightgbm",
        template_id="lightgbm_default",
        budget=BudgetEstimate(wall_time_seconds=5),
        is_baseline=True,
    )
    _train_all(guarded_store, dataset.meta.artifact_id, (screen_spec,), **common)

    tuned_spec = ExperimentSpec(
        experiment_id="exp-lightgbm-tuned",
        template_id="lightgbm_default",
        config={"n_estimators": 50},  # differs from screening's default config
        budget=BudgetEstimate(wall_time_seconds=5),
        is_baseline=True,
    )
    results, skipped = _train_all(
        guarded_store, dataset.meta.artifact_id, (tuned_spec,), **common
    )
    assert skipped == ()
    outcomes = {o.experiment_id: o for o in list_experiment_outcomes(guarded_store, RUN)}
    # A real retrain happened -- different config, so NOT the same model.
    assert (
        outcomes["exp-lightgbm-tuned"].model_ref.artifact_id
        != outcomes["screen-lightgbm"].model_ref.artifact_id
    )
    assert outcomes["exp-lightgbm-tuned"].wall_time_seconds > 0.0


def test_planner_payload_includes_screening_results(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = _spec_artifact(guarded_store)
    screening = [
        {
            "template": "dummy_baseline",
            "primary_metric_mean": 0.5,
            "primary_metric_std": 0.01,
            "wall_time_seconds": 0.3,
            "state": "completed",
        },
        {
            "template": "lightgbm_default",
            "primary_metric_mean": 0.82,
            "primary_metric_std": 0.02,
            "wall_time_seconds": 1.1,
            "state": "completed",
        },
    ]
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
                "rationale": "screening shows strong nonlinear signal",
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
        screening_results=screening,
    )
    sent = json.loads(transport.requests[0].user_content)
    assert sent["screening_results"] == screening


def test_planner_payload_empty_screening_by_default(guarded_store, churn_csv):
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
                "rationale": "no screening available",
            }
        )
    )
    plan_experiments(client, guarded_store, profile=profile, problem=spec, n_rows=200, run_id=RUN)
    sent = json.loads(transport.requests[0].user_content)
    assert sent["screening_results"] == []


def test_gate2_payload_includes_screening_table(tmp_path: Path):
    """Full agentic pipeline over the recorded fixture: Gate 2's summary must
    carry the screening table so a human sees the evidence the plan was
    conditioned on."""
    from mlteam.orchestration import GateDecision, GateName, run_agentic_pipeline
    from synthetic import make_churn_frame

    fixture = Path(__file__).parent / "fixtures" / "llm" / "agentic_churn_run.json"
    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(fixture), model="claude-sonnet-4-6")

    class _RecordingGate:
        def __init__(self) -> None:
            self.summaries: dict = {}

        def request(self, gate, summary: dict) -> GateDecision:
            self.summaries[gate] = summary
            return GateDecision(approved=True, approver="test")

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
    plan_summary = gate.summaries[GateName.TRAINING_PLAN]
    assert "screening_results" in plan_summary
    assert len(plan_summary["screening_results"]) == 3
    templates = {r["template"] for r in plan_summary["screening_results"]}
    assert templates == {"dummy_baseline", "linear_default", "lightgbm_default"}
