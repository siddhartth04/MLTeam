"""Print the complete, ground-truth trace of one run: every stage that ran,
every experiment tried with its real metrics, the winner, the critic's
verdict, and every artifact produced -- reconstructed entirely from the
artifact store, not from the dashboard's in-memory registry.

Why this exists: the dashboard shows live progress but no single view
answers "what actually happened in this run, end to end" once it's done --
that requires cross-referencing the problem spec, the triage report, every
experiment outcome, the evaluation report, the critique, and the final
report by hand. This script does that walk once, sourced from the same
store the report generator itself reads, so what it prints is exactly what
happened -- never a guess or a summary of what SHOULD have happened.

Usage:
    uv run python scripts/run_trace.py <pipeline-run-id> [--store-root PATH]

``<pipeline-run-id>`` is the mlteam pipeline run id (RunRecord.result_run_id
from GET /api/runs/{ui_run_id} -- NOT the dashboard's ui_run_id). It is
``None`` until the run finishes; this script only works on completed (or
far-enough-along) runs for exactly the reason Task 1 of
REPORT_INTEGRITY_ROADMAP.md exists: querying by an unresolved id must not
silently fall back to scanning the whole store.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mlteam.contracts import (
    CleaningManifest,
    CritiqueReport,
    DeploymentBundle,
    EvaluationReport,
    ExperimentOutcome,
    ExperimentPlan,
    ExplainabilityReport,
    FeaturePlan,
    FinalReport,
    ProblemSpec,
    TriageReport,
)
from mlteam.store import GuardedArtifactStore, open_store

_DEFAULT_STORE_ROOT = Path("./artifact_store")

_RULE = "=" * 78


def _section(title: str) -> None:
    print()
    print(_RULE)
    print(title)
    print(_RULE)


def _one_of_type(store: GuardedArtifactStore, run_id: str, artifact_type: str) -> Any | None:
    """The single artifact of ``artifact_type`` belonging to ``run_id``, or
    None. Several artifact types (ProblemSpec, TriageReport, CleaningManifest,
    FeaturePlan, ExperimentPlan) can legitimately have more than one
    revision across an iterative run -- this returns the LATEST by
    ``created_at`` so the trace reflects the final state, matching how
    views.py's problem_spec_view/critique_view pick a "current" revision."""
    scoped = store.artifact_ids_for_run(run_id)
    latest = None
    latest_at = ""
    for aid in store.list_ids(artifact_type):
        record = store._inner.lineage.get_record(aid)
        if record is None or not store.is_in_run(aid, run_id, scoped):
            continue
        if record.created_at >= latest_at:
            latest_at = record.created_at
            latest = aid
    return store.get(latest) if latest is not None else None


def _all_of_type(store: GuardedArtifactStore, run_id: str, artifact_type: str) -> list[Any]:
    scoped = store.artifact_ids_for_run(run_id)
    out = []
    for aid in store.list_ids(artifact_type):
        record = store._inner.lineage.get_record(aid)
        if record is None or not store.is_in_run(aid, run_id, scoped):
            continue
        out.append((record.created_at, store.get(aid)))
    out.sort(key=lambda pair: pair[0])
    return [artifact for _, artifact in out]


def trace_run(store: GuardedArtifactStore, run_id: str) -> None:
    problem: ProblemSpec | None = _one_of_type(store, run_id, ProblemSpec.artifact_type())
    if problem is None:
        print(f"No ProblemSpec found for run '{run_id}' -- is this a valid, run_id, "
              "and has the run progressed past Gate 1?", file=sys.stderr)
        raise SystemExit(1)

    _section("PROBLEM SPECIFICATION")
    print(f"Run id:          {run_id}")
    print(f"Task type:       {problem.task_type.value}")
    print(f"Target column:   {problem.target_column}")
    print(f"Primary metric:  {problem.primary_metric}")
    if problem.secondary_metrics:
        print(f"Secondary:       {', '.join(problem.secondary_metrics)}")
    if problem.prediction_horizon:
        print(f"Horizon:         {problem.prediction_horizon}")
    if problem.business_constraints:
        print("Business constraints:")
        for c in problem.business_constraints:
            print(f"  - {c}")

    triage: TriageReport | None = _one_of_type(store, run_id, TriageReport.artifact_type())
    if triage is not None:
        _section("DATA TRIAGE")
        print(f"Rows before triage: {triage.rows_before}")
        print(f"Rows after triage:  {triage.rows_after}")
        if triage.rows_before:
            dropped_pct = 100 * (triage.rows_before - triage.rows_after) / triage.rows_before
            dropped_n = triage.rows_before - triage.rows_after
            print(f"Rows dropped:       {dropped_n} ({dropped_pct:.1f}%)")
        if triage.issues:
            print("\nIssues detected:")
            for issue in triage.issues:
                print(f"  [{issue.code.value}] {issue.column}: {issue.detail}")
        if triage.remedies:
            print("\nRemedies applied:")
            for remedy in triage.remedies:
                print(f"  [{remedy.action.value}] {remedy.column} "
                      f"({remedy.affected_rows} rows) -- {remedy.justification}")

    cleaning_manifests = _all_of_type(store, run_id, CleaningManifest.artifact_type())
    if cleaning_manifests:
        _section("DATA CLEANING")
        for manifest in cleaning_manifests:
            for op in manifest.ops:
                cols = ", ".join(op.affected_columns) if op.affected_columns else "-"
                print(f"  [{op.op_type.value}] columns={cols} rows={op.affected_rows} "
                      f"-- {op.justification}")

    feature_plans = _all_of_type(store, run_id, FeaturePlan.artifact_type())
    if feature_plans:
        _section("FEATURE ENGINEERING")
        for plan in feature_plans:
            for proposal in plan.proposals:
                inputs = ", ".join(proposal.input_columns)
                print(f"  [{proposal.op_type.value}] {proposal.name} <- ({inputs}) "
                      f"-- {proposal.rationale}")

    plans = _all_of_type(store, run_id, ExperimentPlan.artifact_type())
    if plans:
        _section("EXPERIMENT PLAN")
        for plan in plans:
            for exp in plan.experiments:
                baseline_tag = " [baseline]" if exp.is_baseline else ""
                print(f"  {exp.experiment_id} -> {exp.template_id}{baseline_tag}")

    outcomes: list[ExperimentOutcome] = _all_of_type(
        store, run_id, ExperimentOutcome.artifact_type()
    )
    if outcomes:
        _section("EXPERIMENT RESULTS (every attempt, whatever its fate)")
        for outcome in outcomes:
            print(
                f"\n  {outcome.experiment_id}  [{outcome.template_id}]  "
                f"state={outcome.state.value}"
            )
            if outcome.wall_time_seconds:
                print(f"    wall time: {outcome.wall_time_seconds:.1f}s")
            if outcome.error:
                print(f"    ERROR: {outcome.error}")
            for metric in outcome.cv_metrics:
                print(f"    {metric.name}: mean={metric.mean:.6f} std={metric.std:.6f} "
                      f"folds={[round(v, 4) for v in metric.fold_values]}")

    evaluation: EvaluationReport | None = _one_of_type(
        store, run_id, EvaluationReport.artifact_type()
    )
    if evaluation is not None:
        _section("EVALUATION REPORT (winning model)")
        print(f"Winning model artifact: {evaluation.model_ref.artifact_id[:16]}...")
        print(f"Holdout untouched during development: {evaluation.holdout_untouched}")
        print("\nCross-validation metrics:")
        for metric in evaluation.cv_metrics:
            print(f"  {metric.name}: mean={metric.mean:.6f} std={metric.std:.6f}")
        if evaluation.baseline_deltas:
            print("\nBaseline comparisons:")
            for delta in evaluation.baseline_deltas:
                sig = "significant" if delta.significant else "not significant"
                p = f" (p={delta.p_value:.4f})" if delta.p_value is not None else ""
                print(f"  vs {delta.baseline_name} on {delta.metric}: "
                      f"{delta.baseline_value:.4f} -> {delta.model_value:.4f} "
                      f"(delta={delta.delta:+.4f}, {delta.significance_method}, {sig}{p})")
        if evaluation.leakage_checks:
            print("\nLeakage checks:")
            for check in evaluation.leakage_checks:
                status = "PASS" if check.passed else "FAIL"
                print(f"  [{status}] {check.name}: {check.details}")
        if evaluation.calibration.expected_calibration_error is not None:
            print(f"\nCalibration: ECE={evaluation.calibration.expected_calibration_error:.4f} "
                  f"Brier={evaluation.calibration.brier_score:.4f} "
                  f"method={evaluation.calibration.method}")
        if evaluation.decision_threshold is not None:
            dt = evaluation.decision_threshold
            print(
                f"\nCost-optimal decision threshold: {dt.threshold:.3f} "
                f"(expected cost {dt.expected_cost:.2f} vs "
                f"{dt.cost_at_default_threshold:.2f} at 0.5)"
            )

    critique: CritiqueReport | None = _one_of_type(store, run_id, CritiqueReport.artifact_type())
    if critique is not None:
        _section("CRITIC VERDICT")
        print(f"Verdict: {critique.verdict.value}")
        if critique.findings:
            print("\nFindings:")
            for finding in critique.findings:
                print(f"  [{finding.severity.value}] {finding.category}: {finding.evidence}")
                print(f"    -> {finding.recommendation}")
        if critique.iteration_actions:
            print("\nRequested iteration actions:")
            for action in critique.iteration_actions:
                print(f"  - {action}")

    explainability: ExplainabilityReport | None = _one_of_type(
        store, run_id, ExplainabilityReport.artifact_type()
    )
    if explainability is not None:
        _section("EXPLAINABILITY")
        print(
            f"Method: {explainability.method.value}  "
            f"SHAP available: {explainability.shap_available}"
        )
        print("\nTop feature importances:")
        for fi in explainability.importances[:10]:
            print(f"  {fi.feature}: {fi.importance:.6f} (std={fi.std:.6f})")
        if explainability.residual_slices:
            print("\nWeakest residual slices (regression error analysis):")
            for rs in sorted(explainability.residual_slices, key=lambda s: -s.mae)[:5]:
                print(f"  {rs.feature}={rs.value}: mae={rs.mae:.4f} (n={rs.n_rows})")

    bundle: DeploymentBundle | None = _one_of_type(store, run_id, DeploymentBundle.artifact_type())
    if bundle is not None:
        _section("DEPLOYMENT BUNDLE")
        print(f"Signed off: {bundle.human_signed_off}")
        print(f"Has Dockerfile: {bundle.dockerfile_blob is not None}")
        print(f"Has drift baseline: {bundle.drift_baseline_blob is not None}")

    final_report: FinalReport | None = _one_of_type(store, run_id, FinalReport.artifact_type())
    if final_report is not None:
        _section("FINAL REPORT")
        print(f"Report artifact id: {final_report.meta.artifact_id}")
        print("Headline holdout metrics:")
        for name, value in final_report.headline_metrics.items():
            print(f"  {name}: {value:.6f}")

    _section("DONE")
    print(f"Trace complete for run '{run_id}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="The mlteam pipeline run id (RunRecord.result_run_id)")
    parser.add_argument(
        "--store-root",
        type=Path,
        default=_DEFAULT_STORE_ROOT,
        help=f"Artifact store root directory (default: {_DEFAULT_STORE_ROOT})",
    )
    args = parser.parse_args()

    store = open_store(args.store_root)
    trace_run(store, args.run_id)


if __name__ == "__main__":
    main()
