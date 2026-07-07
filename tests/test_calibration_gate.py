"""Audit Task 7 (H8): the calibration_acceptable gate must not be audit
theater. Before this fix, EvaluationReport.calibration was never populated
by any producer, so the gate always read ece=None, always passed, and
always appeared in the gate list -- implying calibration had been checked
when it structurally never could fail. The fix: the gate is only ADDED to
the list when calibration was actually measured; when not measured, it is
simply absent (not a vacuous pass)."""

from __future__ import annotations

from mlteam.agents.critic import compute_quantitative_gates
from mlteam.contracts import BaselineDelta, CalibrationSummary, EvaluationReport, new_artifact

RUN = "test-calibration-gate"


def _metric(name: str, mean: float, std: float):
    from mlteam.contracts import MetricSummary

    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def _report(calibration: CalibrationSummary) -> EvaluationReport:
    return new_artifact(
        EvaluationReport,
        producer_agent="test",
        run_id=RUN,
        model_ref={"artifact_id": "m", "artifact_type": "TrainedModel"},
        cv_metrics=(_metric("roc_auc", 0.9, 0.02),),
        holdout_untouched=True,
        calibration=calibration,
        baseline_deltas=(
            BaselineDelta(
                baseline_name="dummy_baseline",
                metric="roc_auc",
                baseline_value=0.5,
                model_value=0.9,
                delta=0.4,
                significant=True,
            ),
        ),
    )


def test_calibration_gate_absent_when_not_measured():
    gates = compute_quantitative_gates(_report(CalibrationSummary()), "roc_auc")
    names = {g.name for g in gates}
    assert "calibration_acceptable" not in names


def test_calibration_gate_present_and_passes_when_well_calibrated():
    calibration = CalibrationSummary(
        expected_calibration_error=0.02, brier_score=0.05, method="out_of_fold_binned_ece_10bins"
    )
    gates = compute_quantitative_gates(_report(calibration), "roc_auc")
    gate = next(g for g in gates if g.name == "calibration_acceptable")
    assert gate.passed is True
    assert gate.hard is False


def test_calibration_gate_present_and_fails_when_poorly_calibrated():
    calibration = CalibrationSummary(
        expected_calibration_error=0.35, brier_score=0.20, method="out_of_fold_binned_ece_10bins"
    )
    gates = compute_quantitative_gates(_report(calibration), "roc_auc")
    gate = next(g for g in gates if g.name == "calibration_acceptable")
    assert gate.passed is False
    assert gate.hard is False  # soft gate -- informs the critic, never force-halts
