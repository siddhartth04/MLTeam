"""Task 2 of REPORT_INTEGRITY_ROADMAP.md: ``ReportInputs`` must fail loudly
when its pieces don't actually describe one coherent model/evaluation, or
when a metric illegal for the run's task type has snuck in.

Deliberately does NOT test ``Artifact.meta.run_id`` equality across inputs
as a coherence signal -- that field is first-writer-wins provenance under
content addressing (see mlteam/contracts/base.py and
mlteam/store/lineage.py's ``run_artifacts`` membership table), so a run
that legitimately reuses/dedupes an earlier run's artifact would trip a
naive equality check on entirely normal behavior. ``validate_run_membership``
is tested separately against the store's actual membership table, which is
the dedup-safe way to ask "does this artifact belong to this run".
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mlteam.contracts import (
    BaselineDelta,
    CalibrationSummary,
    CritiqueReport,
    DecisionThreshold,
    EvaluationReport,
    ExplainabilityReport,
    ImportanceMethod,
    MetricSummary,
    ProblemSpec,
    SplitStrategy,
    TaskType,
    TrainedModel,
    Verdict,
    new_artifact,
)
from mlteam.contracts.split import SplitPlan
from mlteam.reporting.builder import ReportInputs
from mlteam.reporting.contract import (
    ReportContractError,
    validate_report_inputs,
    validate_run_membership,
)

RUN = "contract-test-run"


def _metric(name: str, mean: float = 0.8, std: float = 0.02) -> MetricSummary:
    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def _model(run_id: str = RUN, experiment_id: str = "exp-1") -> TrainedModel:
    # experiment_id feeds artifact-id hashing, so a caller wanting a
    # genuinely DIFFERENT artifact (not a content-addressed dedup of the
    # same model) must vary it -- unlike run_id, which is deliberately
    # excluded from hashing (see mlteam/contracts/base.py).
    return new_artifact(
        TrainedModel,
        producer_agent="test",
        run_id=run_id,
        experiment_id=experiment_id,
        template_id="lightgbm_default",
        pipeline_blob="0" * 64,
        seed=42,
        env_hash="0" * 16,
        train_data_ref={"artifact_id": "train-data", "artifact_type": "DatasetVersion"},
    )


def _evaluation(
    model: TrainedModel,
    *,
    primary_metric_name: str = "roc_auc",
    calibration: CalibrationSummary | None = None,
    decision_threshold: DecisionThreshold | None = None,
    run_id: str = RUN,
) -> EvaluationReport:
    if calibration is None:
        calibration = CalibrationSummary()
    return new_artifact(
        EvaluationReport,
        producer_agent="test",
        run_id=run_id,
        model_ref=model.ref(),
        cv_metrics=(_metric(primary_metric_name),),
        baseline_deltas=(
            BaselineDelta(
                baseline_name="dummy_baseline",
                metric=primary_metric_name,
                baseline_value=0.5,
                model_value=0.8,
                delta=0.3,
            ),
        ),
        calibration=calibration,
        decision_threshold=decision_threshold,
        holdout_untouched=True,
    )


def _critique(evaluation: EvaluationReport, run_id: str = RUN) -> CritiqueReport:
    return new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=run_id,
        evaluation_ref=evaluation.ref(),
        verdict=Verdict.APPROVE,
    )


def _split_plan(run_id: str = RUN) -> SplitPlan:
    return new_artifact(
        SplitPlan,
        producer_agent="test",
        run_id=run_id,
        strategy=SplitStrategy.STRATIFIED,
        rationale="test",
        n_folds=5,
        seed=42,
        holdout_fraction=0.2,
        train_ref={"artifact_id": "train", "artifact_type": "DatasetVersion"},
        holdout_ref={"artifact_id": "holdout", "artifact_type": "DatasetVersion"},
        fold_assignment_hash="0" * 16,
    )


def _problem(
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
    primary_metric: str = "roc_auc",
    run_id: str = RUN,
) -> ProblemSpec:
    return new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=run_id,
        task_type=task_type,
        target_column="target",
        primary_metric=primary_metric,
        human_confirmed=True,
    )


def _valid_classification_inputs(run_id: str = RUN) -> ReportInputs:
    model = _model(run_id=run_id)
    evaluation = _evaluation(model, run_id=run_id)
    return ReportInputs(
        problem=_problem(run_id=run_id),
        model=model,
        evaluation=evaluation,
        critique=_critique(evaluation, run_id=run_id),
        split_plan=_split_plan(run_id=run_id),
        goal_text="predict churn",
        best_template_id="lightgbm_default",
        best_experiment_id="exp-1",
    )


def _valid_regression_inputs(run_id: str = RUN) -> ReportInputs:
    model = _model(run_id=run_id)
    evaluation = _evaluation(model, primary_metric_name="rmse", run_id=run_id)
    return ReportInputs(
        problem=_problem(task_type=TaskType.REGRESSION, primary_metric="rmse", run_id=run_id),
        model=model,
        evaluation=evaluation,
        critique=_critique(evaluation, run_id=run_id),
        split_plan=_split_plan(run_id=run_id),
        goal_text="predict price",
        best_template_id="lightgbm_default",
        best_experiment_id="exp-1",
    )


# -- validate_report_inputs: the valid case must pass -----------------------


def test_valid_classification_inputs_pass():
    validate_report_inputs(_valid_classification_inputs())  # must not raise


def test_valid_regression_inputs_pass():
    validate_report_inputs(_valid_regression_inputs())  # must not raise


# -- model_ref coherence -----------------------------------------------------


def test_evaluation_pointing_at_a_different_model_raises():
    inputs = _valid_classification_inputs()
    wrong_model = _model(experiment_id="exp-2")  # a distinct TrainedModel, distinct id
    broken_eval = inputs.evaluation.model_copy(update={"model_ref": wrong_model.ref()})
    broken = replace(inputs, evaluation=broken_eval)
    with pytest.raises(ReportContractError, match="evaluation.model_ref"):
        validate_report_inputs(broken)


def test_explainability_pointing_at_a_different_model_raises():
    inputs = _valid_classification_inputs()
    wrong_ref = {"artifact_id": "some-other-model", "artifact_type": "TrainedModel"}
    explainability = new_artifact(
        ExplainabilityReport,
        producer_agent="test",
        run_id=RUN,
        model_ref=wrong_ref,
        method=ImportanceMethod.PERMUTATION,
        importances=(),
    )
    broken = replace(inputs, explainability=explainability)
    with pytest.raises(ReportContractError, match="explainability.model_ref"):
        validate_report_inputs(broken)


# -- task-illegal metrics -----------------------------------------------------


def test_classification_metric_on_a_regression_spec_raises():
    inputs = _valid_regression_inputs()
    # roc_auc is legal for binary_classification, not regression.
    illegal = inputs.evaluation.model_copy(
        update={"cv_metrics": (_metric("roc_auc"),)}
    )
    broken = replace(inputs, evaluation=illegal)
    with pytest.raises(ReportContractError, match="roc_auc.*not legal for task type regression"):
        validate_report_inputs(broken)


def test_regression_metric_on_a_classification_spec_raises():
    inputs = _valid_classification_inputs()
    illegal = inputs.evaluation.model_copy(update={"cv_metrics": (_metric("rmse"),)})
    broken = replace(inputs, evaluation=illegal)
    with pytest.raises(ReportContractError, match="rmse"):
        validate_report_inputs(broken)


def test_illegal_metric_in_baseline_deltas_raises():
    inputs = _valid_regression_inputs()
    illegal_delta = BaselineDelta(
        baseline_name="dummy_baseline",
        metric="accuracy",  # illegal for regression
        baseline_value=0.5,
        model_value=0.8,
        delta=0.3,
    )
    broken_eval = inputs.evaluation.model_copy(update={"baseline_deltas": (illegal_delta,)})
    broken = replace(inputs, evaluation=broken_eval)
    with pytest.raises(ReportContractError, match="baseline_deltas.*accuracy"):
        validate_report_inputs(broken)


def test_illegal_primary_metric_raises():
    inputs = _valid_regression_inputs()
    broken_problem = inputs.problem.model_copy(update={"primary_metric": "roc_auc"})
    broken = replace(inputs, problem=broken_problem)
    with pytest.raises(ReportContractError, match="primary_metric"):
        validate_report_inputs(broken)


# -- classification-only sections on a regression report --------------------


def test_calibration_present_for_regression_raises():
    inputs = _valid_regression_inputs()
    calibrated = inputs.evaluation.model_copy(
        update={
            "calibration": CalibrationSummary(
                expected_calibration_error=0.02, brier_score=0.1, method="isotonic"
            )
        }
    )
    broken = replace(inputs, evaluation=calibrated)
    with pytest.raises(ReportContractError, match="calibration"):
        validate_report_inputs(broken)


def test_decision_threshold_present_for_regression_raises():
    inputs = _valid_regression_inputs()
    threshold = DecisionThreshold(
        threshold=0.5,
        expected_cost=10.0,
        cost_at_default_threshold=12.0,
        false_positive_cost=1.0,
        false_negative_cost=5.0,
        false_positives=1,
        false_negatives=1,
        true_positives=1,
        true_negatives=1,
    )
    thresholded = inputs.evaluation.model_copy(update={"decision_threshold": threshold})
    broken = replace(inputs, evaluation=thresholded)
    with pytest.raises(ReportContractError, match="decision_threshold"):
        validate_report_inputs(broken)


# -- multiple simultaneous violations are all reported -----------------------


def test_multiple_violations_are_all_named_in_one_error():
    inputs = _valid_regression_inputs()
    illegal_eval = inputs.evaluation.model_copy(
        update={
            "cv_metrics": (_metric("roc_auc"),),
            "calibration": CalibrationSummary(expected_calibration_error=0.1),
        }
    )
    broken_problem = inputs.problem.model_copy(update={"primary_metric": "f1_macro"})
    broken = replace(inputs, evaluation=illegal_eval, problem=broken_problem)
    with pytest.raises(ReportContractError) as excinfo:
        validate_report_inputs(broken)
    # All three independent violations surface in one error, not just the first.
    assert len(excinfo.value.violations) >= 3


# -- validate_run_membership (needs a store) ---------------------------------


def test_run_membership_passes_for_artifacts_actually_stored_under_the_run(guarded_store):
    store = guarded_store
    inputs = _valid_classification_inputs(run_id=RUN)
    for artifact in (inputs.model, inputs.evaluation, inputs.critique, inputs.split_plan):
        store.put(artifact)
    validate_run_membership(store, inputs, run_id=RUN)  # must not raise


def test_run_membership_raises_for_an_artifact_from_a_different_run(guarded_store):
    """The actual cross-run mixup bug: an evaluation genuinely produced by a
    DIFFERENT run's pipeline gets handed to this run's report assembly.
    Distinct experiment_ids keep the two models' artifact ids from
    colliding, so "belongs to other-run only" is unambiguous rather than
    relying on content-addressed dedup coincidence."""
    store = guarded_store
    other_run_model = _model(run_id="other-run", experiment_id="other-run-exp")
    other_run_eval = _evaluation(other_run_model, run_id="other-run")
    store.put(other_run_model)
    store.put(other_run_eval)  # only ever stored under "other-run"

    this_run_model = _model(run_id=RUN, experiment_id="this-run-exp")
    store.put(this_run_model)

    inputs = ReportInputs(
        problem=_problem(run_id=RUN),
        model=this_run_model,
        evaluation=other_run_eval,  # belongs to "other-run", not RUN
        critique=_critique(other_run_eval, run_id=RUN),
        split_plan=_split_plan(run_id=RUN),
        goal_text="predict churn",
        best_template_id="lightgbm_default",
        best_experiment_id="this-run-exp",
    )
    store.put(inputs.critique)
    store.put(inputs.split_plan)

    with pytest.raises(ReportContractError, match="evaluation.*not a member of run"):
        validate_run_membership(store, inputs, run_id=RUN)


def test_run_membership_allows_legitimately_deduped_artifact(guarded_store):
    """Content addressing lets two runs share the identical TrainedModel when
    they train the same config over the same data -- the membership table
    (not meta.run_id) is what proves it belongs to BOTH runs' reports."""
    store = guarded_store
    shared_model = _model(run_id="first-run")  # meta.run_id = "first-run" (first writer)
    store.put(shared_model)  # records run_artifacts membership for "first-run"

    # A second run reuses/dedupes the identical model artifact (same content
    # -> same id) by storing it again under a different run_id: the
    # membership table gains a SECOND row for the same artifact_id.
    reused = new_artifact(
        TrainedModel,
        producer_agent="test",
        run_id="second-run",
        experiment_id="exp-1",
        template_id="lightgbm_default",
        pipeline_blob="0" * 64,
        seed=42,
        env_hash="0" * 16,
        train_data_ref={"artifact_id": "train-data", "artifact_type": "DatasetVersion"},
    )
    assert reused.meta.artifact_id == shared_model.meta.artifact_id  # confirms the dedup premise
    store.put(reused)

    evaluation = _evaluation(shared_model, run_id="second-run")
    store.put(evaluation)
    critique = _critique(evaluation, run_id="second-run")
    store.put(critique)
    split_plan = _split_plan(run_id="second-run")
    store.put(split_plan)

    inputs = ReportInputs(
        problem=_problem(run_id="second-run"),
        model=shared_model,  # meta.run_id says "first-run" -- but IS a member of second-run too
        evaluation=evaluation,
        critique=critique,
        split_plan=split_plan,
        goal_text="predict churn",
        best_template_id="lightgbm_default",
        best_experiment_id="exp-1",
    )
    # Must NOT raise: the model belongs to second-run via the membership
    # table even though its own meta.run_id names first-run.
    validate_run_membership(store, inputs, run_id="second-run")
