"""Fail-loud coherence checks over ``ReportInputs`` before a report is built.

Nothing upstream of ``build_final_report`` previously verified that a
report's inputs actually belonged together: that the evaluation's
``model_ref`` really points at the model being reported on, or that a
regression run's evaluation never carries a classification-only metric
name. A caller-assembly bug (mixing up which run's artifacts to pass) would
silently render as a plausible-looking report with the wrong content — the
exact failure class investigated in the 2026-07-07 report-pipeline audit.

Deliberately NOT checked here: ``Artifact.meta.run_id`` equality across
inputs. Artifact ids are content-addressed and ``new_artifact`` excludes
run_id from hashing on purpose (mlteam/contracts/base.py), so a run that
legitimately reuses/dedupes an earlier run's ``TrainedModel`` produces an
artifact whose own ``meta.run_id`` is first-writer-wins provenance for a
DIFFERENT run (see ``ArtifactLineageIndex.record``'s docstring and the
``run_artifacts`` membership table in mlteam/store/lineage.py). Comparing
``meta.run_id`` across ``ReportInputs`` would raise on that entirely normal
case. The real question -- "do these artifacts belong to the run I am
reporting on?" -- is answered by ``GuardedArtifactStore.is_in_run`` against
the membership table, via ``validate_run_membership`` below, which callers
that hold a store should run in addition to ``validate_report_inputs``.
"""

from __future__ import annotations

from mlteam.contracts import TaskType, is_classification
from mlteam.evaluation.metrics import scoring_for
from mlteam.reporting.builder import ReportInputs
from mlteam.store.guard import GuardedArtifactStore

# Single source of truth for "which metric names are legal for this task
# type" -- derived from scoring_for rather than hand-copied, so a new metric
# added to metrics.py is automatically legal here without a second edit.
ALLOWED_METRICS: dict[TaskType, frozenset[str]] = {
    task_type: frozenset(scoring_for(task_type)) for task_type in TaskType
}

# Every metric name that is legal for ANY task type -- used to compute the
# forbidden set for a specific task type (Task 3's vocabulary linter).
_ALL_METRICS: frozenset[str] = frozenset().union(*ALLOWED_METRICS.values())


class ReportContractError(ValueError):
    """A ``ReportInputs`` (or its rendered output) violates a coherence
    invariant a report must never violate. Carries every violation found,
    not just the first, so a caller fixing an assembly bug sees the whole
    picture in one failure instead of one exception per retry."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            f"{len(violations)} report contract violation(s): " + "; ".join(violations)
        )


def _model_ref_violations(inputs: ReportInputs) -> list[str]:
    violations: list[str] = []
    model_id = inputs.model.meta.artifact_id
    if inputs.evaluation.model_ref.artifact_id != model_id:
        violations.append(
            "evaluation.model_ref "
            f"({inputs.evaluation.model_ref.artifact_id}) does not point at "
            f"the reported model ({model_id})"
        )
    explainability = inputs.explainability
    if explainability is not None and explainability.model_ref.artifact_id != model_id:
        violations.append(
            "explainability.model_ref "
            f"({inputs.explainability.model_ref.artifact_id}) does not point "
            f"at the reported model ({model_id})"
        )
    if inputs.insights is not None and inputs.insights.model_ref.artifact_id != model_id:
        violations.append(
            "insights.model_ref "
            f"({inputs.insights.model_ref.artifact_id}) does not point at "
            f"the reported model ({model_id})"
        )
    return violations


def _metric_legality_violations(inputs: ReportInputs) -> list[str]:
    task_type = inputs.problem.task_type
    allowed = ALLOWED_METRICS[task_type]
    violations: list[str] = []

    if inputs.problem.primary_metric not in allowed:
        violations.append(
            f"primary_metric '{inputs.problem.primary_metric}' is not legal for "
            f"task type {task_type.value} (allowed: {sorted(allowed)})"
        )

    for m in inputs.evaluation.cv_metrics:
        if m.name not in allowed:
            violations.append(
                f"cv_metrics contains '{m.name}', not legal for task type "
                f"{task_type.value} (allowed: {sorted(allowed)})"
            )

    for d in inputs.evaluation.baseline_deltas:
        if d.metric not in allowed:
            violations.append(
                f"baseline_deltas contains metric '{d.metric}', not legal for "
                f"task type {task_type.value} (allowed: {sorted(allowed)})"
            )

    return violations


def _classification_section_violations(inputs: ReportInputs) -> list[str]:
    """Calibration and decision-threshold are meaningful only for a model
    that outputs class probabilities -- a regression/forecasting run must
    never carry either, since their presence in the report would read as
    "this model was calibration-checked" for a model type calibration does
    not apply to."""
    violations: list[str] = []
    if not is_classification(inputs.problem.task_type):
        calibration = inputs.evaluation.calibration
        if calibration.expected_calibration_error is not None:
            violations.append(
                f"task type {inputs.problem.task_type.value} carries a "
                "calibration.expected_calibration_error, but calibration only "
                "applies to classification tasks"
            )
        if inputs.evaluation.decision_threshold is not None:
            violations.append(
                f"task type {inputs.problem.task_type.value} carries a "
                "decision_threshold, but decision thresholds only apply to "
                "classification tasks"
            )
    return violations


def validate_report_inputs(inputs: ReportInputs) -> None:
    """Raise ``ReportContractError`` if ``inputs`` cannot possibly describe
    one coherent model/evaluation. Store-independent: does not check run
    membership (see ``validate_run_membership`` for that, which needs a
    store handle). Call this first, unconditionally, from
    ``build_final_report``."""
    violations = [
        *_model_ref_violations(inputs),
        *_metric_legality_violations(inputs),
        *_classification_section_violations(inputs),
    ]
    if violations:
        raise ReportContractError(violations)


def validate_run_membership(
    store: GuardedArtifactStore, inputs: ReportInputs, *, run_id: str
) -> None:
    """Raise ``ReportContractError`` if any artifact in ``inputs`` does not
    belong to ``run_id`` per the store's ``run_artifacts`` membership table.

    This is the actual "these inputs got mixed up across runs" check --
    intentionally separate from ``validate_report_inputs`` because it needs
    a store handle ``ReportInputs`` itself does not carry, and because
    ``Artifact.meta.run_id`` (first-writer-wins provenance) is NOT a valid
    substitute: a legitimately deduped artifact belongs to this run via the
    membership table while its own ``meta.run_id`` still names whichever run
    produced it first.
    """
    scoped = store.artifact_ids_for_run(run_id)
    candidates = [
        ("model", inputs.model.meta.artifact_id),
        ("evaluation", inputs.evaluation.meta.artifact_id),
        ("critique", inputs.critique.meta.artifact_id),
        ("split_plan", inputs.split_plan.meta.artifact_id),
    ]
    if inputs.explainability is not None:
        candidates.append(("explainability", inputs.explainability.meta.artifact_id))
    if inputs.insights is not None:
        candidates.append(("insights", inputs.insights.meta.artifact_id))
    if inputs.feature_plan is not None:
        candidates.append(("feature_plan", inputs.feature_plan.meta.artifact_id))
    if inputs.cleaning_manifest is not None:
        candidates.append(("cleaning_manifest", inputs.cleaning_manifest.meta.artifact_id))

    violations = [
        f"{field_name} artifact ({artifact_id}) is not a member of run '{run_id}'"
        for field_name, artifact_id in candidates
        if not store.is_in_run(artifact_id, run_id, scoped)
    ]
    if violations:
        raise ReportContractError(violations)


def lint_rendered_metric_names(
    metric_names: list[str], *, task_type: TaskType, primary_metric: str
) -> None:
    """Check the METRIC NAMES that actually got rendered into a report's
    tables/cards against the task type's legal set.

    Defense-in-depth for Task 2's ``validate_report_inputs``: catches a
    future template change that renders a metric name from somewhere Task 2
    doesn't inspect (e.g. a new template section reading a differently-named
    field). ``metric_names`` must be the literal list of metric-name strings
    the template actually iterated over (headline card labels, CV/holdout
    table rows, baseline-delta rows) -- never raw page HTML.

    An earlier version of this function scanned the full rendered HTML for
    forbidden tokens as whole words. That approach shipped a false positive
    to production the first time it ran against a real regression report:
    the LLM-narrated insight text legitimately used the English word
    "accuracy" ("prioritizing prediction accuracy for negotiation"), which
    is indistinguishable from the classification metric under a word-
    boundary regex, and the run failed at the final report step after a
    successful two-hour pipeline. Free-text narration (insight findings/
    caveats, critic evidence/recommendations) is written in plain English by
    an LLM and WILL contain ordinary English words that collide with metric
    names ("accuracy", conceivably "recall", "support") without meaning the
    metric -- there is no regex that can tell prose from data. Checking the
    actual metric-name values a template rendered, rather than pattern-
    matching the whole page, closes that hole by construction: prose is
    simply never in the input list.
    """
    allowed = ALLOWED_METRICS[task_type]
    forbidden = _ALL_METRICS - allowed

    violations = [
        f"report for task type {task_type.value} rendered forbidden metric "
        f"'{name}' (legal only for other task types)"
        for name in metric_names
        if name in forbidden
    ]
    if metric_names and primary_metric not in metric_names:
        violations.append(
            f"report does not render its own primary_metric '{primary_metric}' anywhere"
        )
    if violations:
        raise ReportContractError(violations)
