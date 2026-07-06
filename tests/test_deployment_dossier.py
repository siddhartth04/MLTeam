"""Phase 5: the deployment dossier — Gate 3 answers "why trust this?"

Before this existed, Gate 3 showed only artifact hashes. All the evidence a
human needs (how every candidate scored, by how much the winner led, which
features drove it, what the critic flagged) already existed as artifacts;
nobody assembled it into the room where the deployment decision gets made.
"""

from __future__ import annotations

from mlteam.agents.deployer import build_deployment_bundle, build_deployment_dossier
from mlteam.contracts import (
    CritiqueReport,
    ExperimentOutcome,
    ExperimentState,
    ExplainabilityReport,
    Finding,
    FindingSeverity,
    ImportanceMethod,
    MetricSummary,
    ProblemSpec,
    TaskType,
    TrainedModel,
    Verdict,
    new_artifact,
)
from mlteam.contracts.explainability import FeatureImportance

RUN = "test-dossier"


def _metric(name: str, mean: float, std: float = 0.01) -> MetricSummary:
    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def _model(experiment_id: str, template_id: str) -> TrainedModel:
    return new_artifact(
        TrainedModel,
        producer_agent="test",
        run_id=RUN,
        experiment_id=experiment_id,
        template_id=template_id,
        pipeline_blob="a" * 64,
        config={},
        seed=42,
        env_hash="test-env",
        train_data_ref={"artifact_id": "x", "artifact_type": "DatasetVersion"},
    )


def _outcome(
    experiment_id: str,
    template_id: str,
    *,
    state: ExperimentState = ExperimentState.COMPLETED,
    roc_auc: float | None = None,
    error: str | None = None,
) -> ExperimentOutcome:
    return new_artifact(
        ExperimentOutcome,
        producer_agent="test",
        run_id=RUN,
        experiment_id=experiment_id,
        template_id=template_id,
        state=state,
        cv_metrics=(_metric("roc_auc", roc_auc),) if roc_auc is not None else (),
        error=error,
    )


def _spec() -> ProblemSpec:
    return new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )


def test_dossier_ranks_completed_outcomes_and_computes_margins():
    outcomes = (
        _outcome("exp-dummy", "dummy_baseline", roc_auc=0.50),
        _outcome("exp-winner", "lightgbm_default", roc_auc=0.90),
        _outcome("exp-runnerup", "random_forest_default", roc_auc=0.85),
        _outcome("exp-broken", "svm_default", state=ExperimentState.FAILED, error="boom"),
    )
    winner = _model("exp-winner", "lightgbm_default")
    dossier = build_deployment_dossier(
        model=winner,
        problem=_spec(),
        train_dataset_id="ds-1",
        split_plan_id="split-1",
        outcomes=outcomes,
    )
    assert len(dossier["comparison"]) == 4
    winner_row = next(r for r in dossier["comparison"] if r["is_winner"])
    assert winner_row["experiment_id"] == "exp-winner"
    failed_row = next(r for r in dossier["comparison"] if r["state"] == "failed")
    assert failed_row["mean"] is None

    margin = dossier["winner_margin"]
    assert margin["vs_runner_up"]["template_id"] == "random_forest_default"
    assert abs(margin["vs_runner_up"]["improvement"] - 0.05) < 1e-9
    assert abs(margin["vs_dummy"] - 0.40) < 1e-9

    assert dossier["lineage"]["model_id"] == winner.meta.artifact_id
    assert dossier["lineage"]["dataset_id"] == "ds-1"
    assert dossier["lineage"]["split_plan_id"] == "split-1"
    assert "rollback" not in str(dossier).lower()  # no fabricated rollback claim


def test_dossier_top_features_truncated_and_sorted():
    importances = tuple(
        FeatureImportance(feature=f"f{i}", importance=float(i)) for i in range(8)
    )
    explainability = new_artifact(
        ExplainabilityReport,
        producer_agent="test",
        run_id=RUN,
        model_ref={"artifact_id": "m", "artifact_type": "TrainedModel"},
        method=ImportanceMethod.PERMUTATION,
        importances=importances,
    )
    dossier = build_deployment_dossier(
        model=_model("exp-winner", "lightgbm_default"),
        problem=_spec(),
        train_dataset_id="ds-1",
        split_plan_id=None,
        explainability=explainability,
    )
    assert len(dossier["top_features"]) == 5
    assert dossier["top_features"][0]["feature"] == "f7"  # highest importance first
    assert dossier["top_features"][-1]["feature"] == "f3"


def test_dossier_includes_critique_findings():
    critique = new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=RUN,
        evaluation_ref={"artifact_id": "e", "artifact_type": "EvaluationReport"},
        verdict=Verdict.APPROVE,
        findings=(
            Finding(
                severity=FindingSeverity.LOW,
                category="calibration",
                evidence="slightly overconfident on the minority class",
                recommendation="monitor post-deploy",
            ),
        ),
    )
    dossier = build_deployment_dossier(
        model=_model("exp-winner", "lightgbm_default"),
        problem=_spec(),
        train_dataset_id="ds-1",
        split_plan_id=None,
        critique=critique,
    )
    assert dossier["critique"]["verdict"] == "approve"
    assert len(dossier["critique"]["findings"]) == 1
    assert dossier["critique"]["findings"][0]["severity"] == "low"


def test_dossier_handles_no_optional_inputs_gracefully():
    dossier = build_deployment_dossier(
        model=_model("exp-winner", "lightgbm_default"),
        problem=_spec(),
        train_dataset_id="ds-1",
        split_plan_id=None,
    )
    assert dossier["comparison"] == []
    assert dossier["winner_margin"] is None
    assert dossier["top_features"] == []
    assert dossier["critique"] is None


class _RecordingGate:
    def __init__(self) -> None:
        self.summary: dict | None = None

    def request(self, gate, summary: dict):
        from mlteam.orchestration.gates import GateDecision

        self.summary = summary
        return GateDecision(approved=True, approver="test")


def test_build_deployment_bundle_gate_payload_includes_dossier(guarded_store, churn_csv):
    from mlteam.agents import ingest_file, profile_dataset

    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    model = _model("exp-winner", "lightgbm_default")
    guarded_store.put(model)
    outcomes = (
        _outcome("exp-dummy", "dummy_baseline", roc_auc=0.5),
        _outcome("exp-winner", "lightgbm_default", roc_auc=0.9),
    )
    gate = _RecordingGate()

    build_deployment_bundle(
        guarded_store,
        model=model,
        profile=profile,
        problem=_spec(),
        train_dataset_id=dataset.meta.artifact_id,
        feature_names=["tenure_months"],
        gate=gate,
        run_id=RUN,
        outcomes=outcomes,
    )

    assert gate.summary is not None
    assert "dossier" in gate.summary
    assert len(gate.summary["dossier"]["comparison"]) == 2
