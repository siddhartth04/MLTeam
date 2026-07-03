"""Contract tests: round-trips, content addressing, immutability, validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlteam.contracts import (
    ARTIFACT_REGISTRY,
    ArtifactRef,
    BudgetEstimate,
    CleaningManifest,
    CleaningOp,
    CleaningOpType,
    CritiqueReport,
    DatasetRole,
    DatasetVersion,
    ExperimentPlan,
    ExperimentSpec,
    ProblemSpec,
    SplitPlan,
    SplitStrategy,
    TaskType,
    Verdict,
    new_artifact,
)

RUN = "run-contracts"
REF = ArtifactRef(artifact_id="a" * 64, artifact_type="DatasetVersion")


def _spec(**overrides):
    payload = {
        "task_type": TaskType.BINARY_CLASSIFICATION,
        "target_column": "churned",
        "primary_metric": "roc_auc",
    }
    payload.update(overrides)
    return new_artifact(ProblemSpec, producer_agent="req-agent", run_id=RUN, **payload)


def test_round_trip_preserves_artifact():
    spec = _spec()
    reloaded = ProblemSpec.model_validate_json(spec.model_dump_json())
    assert reloaded == spec


def test_registry_covers_all_artifact_types():
    assert len(ARTIFACT_REGISTRY) == 14
    for name, cls in ARTIFACT_REGISTRY.items():
        assert cls.artifact_type() == name


def test_content_addressing_is_deterministic():
    a = _spec()
    b = _spec()
    assert a.meta.artifact_id == b.meta.artifact_id


def test_content_addressing_ignores_run_identity():
    a = new_artifact(
        ProblemSpec,
        producer_agent="req-agent",
        run_id="run-A",
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="rmse",
    )
    b = new_artifact(
        ProblemSpec,
        producer_agent="other-agent",
        run_id="run-B",
        task_type=TaskType.REGRESSION,
        target_column="price",
        primary_metric="rmse",
    )
    assert a.meta.artifact_id == b.meta.artifact_id


def test_different_payload_changes_id():
    assert _spec().meta.artifact_id != _spec(primary_metric="f1").meta.artifact_id


def test_config_hash_changes_id():
    a = _spec()
    b = new_artifact(
        ProblemSpec,
        producer_agent="req-agent",
        run_id=RUN,
        config_hash="different-config",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )
    assert a.meta.artifact_id != b.meta.artifact_id


def test_artifacts_are_immutable():
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.primary_metric = "accuracy"


def test_experiment_plan_requires_baseline():
    non_baseline = ExperimentSpec(
        experiment_id="e1",
        template_id="lgbm_default",
        budget=BudgetEstimate(wall_time_seconds=60),
    )
    with pytest.raises(ValidationError, match="baseline"):
        new_artifact(
            ExperimentPlan,
            producer_agent="planner",
            run_id=RUN,
            experiments=(non_baseline,),
            rationale="no baseline here",
        )


def test_experiment_plan_rejects_baseline_after_model():
    baseline = ExperimentSpec(
        experiment_id="b1",
        template_id="dummy",
        budget=BudgetEstimate(wall_time_seconds=5),
        is_baseline=True,
    )
    model = ExperimentSpec(
        experiment_id="e1",
        template_id="xgb",
        budget=BudgetEstimate(wall_time_seconds=120),
    )
    with pytest.raises(ValidationError, match="precede"):
        new_artifact(
            ExperimentPlan,
            producer_agent="planner",
            run_id=RUN,
            experiments=(baseline, model, baseline.model_copy(update={"experiment_id": "b2"})),
            rationale="baseline out of order",
        )


def test_experiment_plan_total_budget():
    plan = new_artifact(
        ExperimentPlan,
        producer_agent="planner",
        run_id=RUN,
        experiments=(
            ExperimentSpec(
                experiment_id="b1",
                template_id="dummy",
                budget=BudgetEstimate(wall_time_seconds=5, cost_usd=0.01),
                is_baseline=True,
            ),
            ExperimentSpec(
                experiment_id="e1",
                template_id="xgb",
                budget=BudgetEstimate(wall_time_seconds=120, cost_usd=0.05),
            ),
        ),
        rationale="baseline then xgb",
    )
    assert plan.total_budget.wall_time_seconds == 125
    assert plan.total_budget.cost_usd == pytest.approx(0.06)


def test_critique_iterate_requires_actions():
    with pytest.raises(ValidationError, match="action"):
        new_artifact(
            CritiqueReport,
            producer_agent="critic",
            run_id=RUN,
            evaluation_ref=REF,
            verdict=Verdict.ITERATE,
        )


def test_split_plan_temporal_requires_time_column():
    with pytest.raises(ValidationError, match="time_column"):
        new_artifact(
            SplitPlan,
            producer_agent="splitter",
            run_id=RUN,
            strategy=SplitStrategy.TEMPORAL,
            rationale="time-ordered data",
            n_folds=5,
            seed=42,
            holdout_fraction=0.2,
            train_ref=REF,
            holdout_ref=REF,
            fold_assignment_hash="abc",
        )


def test_cleaning_op_requires_justification():
    with pytest.raises(ValidationError):
        CleaningOp(op_type=CleaningOpType.DROP_COLUMN, justification="")


def test_cleaning_manifest_round_trip():
    manifest = new_artifact(
        CleaningManifest,
        producer_agent="cleaner",
        run_id=RUN,
        input_dataset_ref=REF,
        ops=(
            CleaningOp(
                op_type=CleaningOpType.IMPUTE_MEDIAN,
                params={"columns": ["age"]},
                justification="age has 4% missing, roughly symmetric distribution",
                affected_columns=("age",),
                affected_rows=40,
            ),
        ),
    )
    reloaded = CleaningManifest.model_validate_json(manifest.model_dump_json())
    assert reloaded == manifest
    assert reloaded.ops[0].op_type is CleaningOpType.IMPUTE_MEDIAN


def test_nan_payload_is_rejected():
    from mlteam.contracts import ColumnProfile, DataProfile

    nan_column = ColumnProfile(
        name="x", dtype="f64", cardinality=3, missing_pct=0.0, mean=float("nan")
    )
    with pytest.raises(ValueError, match="NaN"):
        new_artifact(
            DataProfile,
            producer_agent="profiler",
            run_id=RUN,
            dataset_ref=REF,
            columns=(nan_column,),
        )


def test_mlflow_run_id_excluded_from_identity_hash():
    from mlteam.contracts import TrainedModel

    def build(mlflow_run_id):
        return new_artifact(
            TrainedModel,
            producer_agent="trainer",
            run_id=RUN,
            experiment_id="exp-1",
            template_id="lightgbm_default",
            pipeline_blob="b" * 64,
            seed=42,
            env_hash="e" * 64,
            train_data_ref=REF,
            mlflow_run_id=mlflow_run_id,
        )

    a = build("mlflow-run-aaa")
    b = build("mlflow-run-bbb")
    assert a.meta.artifact_id == b.meta.artifact_id  # identity ignores tracking id
    assert a.payload_dict()["mlflow_run_id"] == "mlflow-run-aaa"  # but it IS stored


def test_dataset_role_enum_serializes_as_string():
    ds = new_artifact(
        DatasetVersion,
        producer_agent="ingest",
        run_id=RUN,
        blob_hash="f" * 64,
        n_rows=10,
        n_cols=2,
        role=DatasetRole.HOLDOUT,
    )
    assert ds.payload_dict()["role"] == "holdout"
