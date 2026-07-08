"""Critic agent: structured verdict over the EvaluationReport — nothing else.

The Critic reads ONLY the structured EvaluationReport summary. Quantitative
gates are computed deterministically in code; the LLM interprets and writes
findings, but it cannot approve past a failed hard gate — code enforces that,
logging an override finding when it must.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from mlteam.contracts import (
    CritiqueReport,
    DataProfile,
    EvaluationReport,
    ExcludedTemplate,
    ExperimentOutcome,
    ExperimentSpec,
    Finding,
    FindingSeverity,
    ProblemSpec,
    Verdict,
    new_artifact,
)
from mlteam.evaluation.field_checks import (
    audit_terminal_states,
    compute_cost_census,
    compute_failure_census,
    compute_winner_margin,
)
from mlteam.llm.client import LLMClient
from mlteam.llm.summaries import evaluation_summary
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are the Critic of an autonomous ML engineering team. You \
receive a structured evaluation report and deterministic quantitative gate \
results. You never see raw data — judge only the evidence provided.

Verdict rules (the orchestrator enforces these; violating them gets overridden):
- If any leakage check failed or the holdout was touched: verdict MUST be halt.
- If the model does not beat the dummy baseline significantly on the primary \
metric: verdict MUST be iterate_with_actions, with concrete actions. Each \
baseline_delta carries significance_method: "nadeau_bengio_paired_t" means \
`significant` came from a real corrected paired t-test over per-fold deltas \
(p_value is populated) — cite the p-value in your finding. \
"heuristic_2sigma" means fold pairing was unavailable and a documented \
pooled-std heuristic was used instead — say so explicitly rather than \
implying a real test ran.
- If the terminal-state audit reports incomplete (a planned experiment has no \
outcome): verdict MUST be halt — this is an orchestration bug, not a model- \
quality question, and must be fixed before anything is trusted.
- approve ONLY when every hard gate passes.
- iterate_with_actions requires specific, actionable steps (e.g. "add class \
weights", "try target encoding for high-cardinality column X") — not vague \
advice.

You also receive the FULL experiment field (every outcome this run, not just \
the winner), the plan's excluded_templates (families deliberately not tried, \
with reasons), and a winner_margin already computed in units of pooled fold \
standard deviation:
- Compare the winner against every completed competitor, not just the dummy. \
If a competitor is close, say so.
- Challenge exploration coverage: given the profile and task, was excluding a \
listed family actually justified? If a plausible family was skipped without a \
good reason (or wasn't listed in excluded_templates at all), emit an \
iterate_with_actions action naming the specific template to try.
- When winner_margin's vs_runner_up_stds is below 0.5 (or null because fold \
variance was ~0 on both sides), the winner and its closest competitor are \
statistically indistinguishable. Your finding text MUST address this directly: \
either recommend the simpler/cheaper of the two, or give a concrete reason to \
prefer the more complex winner anyway (e.g. materially better calibration or \
slice performance) — do not stay silent on a tie.
- Every critique must explicitly state, in a finding or the rationale evidence: \
"evidence sufficient for deployment: yes/no, because ...".

When decision_threshold is present (a binary classification run with a stated \
cost asymmetry), it is the cost-optimal cutoff computed from out-of-fold \
probabilities — NOT 0.5. Cite expected_cost vs cost_at_default_threshold; if \
the gap is large the report should be understood at THAT threshold's confusion \
counts, not the rank metrics alone. When it is null, do not assume a threshold \
was chosen — say so if a deployment decision would need one.

decision_context (decision_use/error_tolerance_note/human_baseline_note) tells \
you what the model is FOR and what today's alternative is — read it before \
judging "beats dummy" as sufficient. Any empty field there means the business \
never said, not that there is no constraint: name that gap in a finding rather \
than assuming a comfortable default. A relative_error_context gate, when \
present, states MAE as a fraction of the median target value — cite it plainly \
("typical error is X% of the median value") so the business, not just the \
metric, can judge fitness. Note: a stated-and-violated error tolerance or a \
high relative error without a stated tolerance is enforced deterministically \
by the orchestrator (not by your verdict alone) — write the finding regardless, \
so the reasoning is visible even when the override does the enforcing.

For findings, cite the numeric evidence from the report. Severity: critical for \
leakage/holdout issues, high for failing the dummy gate, medium/low for quality \
concerns (calibration, fold variance, slice gaps, an unresolved tie, or \
unjustified excluded coverage).
Return only JSON matching the schema."""


_REGRESSION_METRICS = frozenset({"rmse", "mae", "r2"})
# Generous on purpose (JUDGMENT_ROADMAP.md Task 3): this only fires the
# "fitness unknown" finding when NO tolerance was stated at all, so it must
# be loose enough not to nag on ordinary, healthy models — it is a floor
# for "the business has not told us if this is fit for purpose", not a
# quality bar in its own right.
_RELATIVE_ERROR_UNSTATED_TOLERANCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class QuantitativeGate:
    name: str
    passed: bool
    detail: str
    hard: bool  # hard gates constrain the verdict


class FindingDraft(BaseModel):
    severity: FindingSeverity
    category: str
    evidence: str
    recommendation: str


class CritiqueDraft(BaseModel):
    findings: list[FindingDraft] = Field(default_factory=list)
    verdict: Verdict
    iteration_actions: list[str] = Field(default_factory=list)


def _relative_error_ratio(
    report: EvaluationReport, profile: DataProfile | None, target_column: str | None
) -> float | None:
    """MAE of the winner / median(|target|) from the profile. None when the
    ingredients aren't there (non-regression run, no MAE reported, or no
    profile/target given) -- shared by the report-only gate below and the
    deterministic fitness check, so both agree on one number."""
    if profile is None or target_column is None:
        return None
    mae = next((m for m in report.cv_metrics if m.name == "mae"), None)
    if mae is None:
        return None
    target_col = next((c for c in profile.columns if c.name == target_column), None)
    if target_col is None or target_col.median is None or target_col.median == 0:
        return None
    return mae.mean / abs(target_col.median)


def _relative_error_context(
    report: EvaluationReport, profile: DataProfile | None, target_column: str | None
) -> QuantitativeGate | None:
    """Report-only (never hard, never fails the model): states the model's
    error in real-world terms instead of an abstract metric number."""
    ratio = _relative_error_ratio(report, profile, target_column)
    if ratio is None:
        return None
    mae = next(m for m in report.cv_metrics if m.name == "mae")
    return QuantitativeGate(
        name="relative_error_context",
        passed=True,  # informational only -- never blocks a verdict
        detail=f"typical error is {ratio:.1%} of the median target value (MAE={mae.mean:.4g})",
        hard=False,
    )


_PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _stated_tolerance_fraction(note: str) -> float | None:
    """Best-effort extraction of a percentage tolerance from free text
    (e.g. "within 10% for pricing" -> 0.10). None when no percentage is
    found -- the note is still shown to the LLM verbatim regardless; this
    only feeds the deterministic violation check below."""
    match = _PERCENT_PATTERN.search(note)
    return float(match.group(1)) / 100.0 if match else None


def _fitness_findings(
    problem: ProblemSpec | None, ratio: float | None
) -> tuple[tuple[Finding, ...], bool]:
    """Deterministic fitness-for-purpose check (JUDGMENT_ROADMAP.md Task 3)
    -- prompts steer, code enforces, so this never depends on the LLM
    noticing the gap on its own. Returns (extra findings,
    tolerance_violated); ``tolerance_violated`` forces the verdict away
    from approve in ``_enforce_verdict``."""
    if problem is None or ratio is None:
        return (), False
    stated = _stated_tolerance_fraction(problem.error_tolerance_note)
    if stated is not None:
        if ratio > stated:
            finding = Finding(
                severity=FindingSeverity.HIGH,
                category="fitness_for_purpose",
                evidence=(
                    f"stated tolerance '{problem.error_tolerance_note}' implies "
                    f"{stated:.1%}, but relative error is {ratio:.1%}"
                ),
                recommendation="do not approve until error is within the stated tolerance",
            )
            return (finding,), True
        return (), False
    if ratio > _RELATIVE_ERROR_UNSTATED_TOLERANCE_THRESHOLD:
        finding = Finding(
            severity=FindingSeverity.HIGH,
            category="fitness_for_purpose",
            evidence=f"relative error is {ratio:.1%} of the median target value",
            recommendation=(
                "fitness for purpose is unknown -- ask the business for an error "
                "tolerance before treating 'beats dummy' as sufficient"
            ),
        )
        return (finding,), False
    return (), False


def compute_quantitative_gates(
    report: EvaluationReport,
    primary_metric: str,
    *,
    missing_outcome_ids: tuple[str, ...] = (),
    profile: DataProfile | None = None,
    target_column: str | None = None,
) -> tuple[QuantitativeGate, ...]:
    """Deterministic gate results — the checklist the Critic must respect.

    ``profile``/``target_column``, when given, add a report-only
    ``relative_error_context`` gate for regression (see
    ``_relative_error_context``) — never pass/fail, purely informational.

    ``missing_outcome_ids`` comes from the terminal-state audit: any planned
    experiment (including screening) with no persisted ExperimentOutcome. A
    non-empty tuple here means something crashed hard enough to skip recording
    an outcome — an orchestration bug, not a model-quality question — so it is
    a HARD gate: the critic cannot approve past it.
    """
    gates: list[QuantitativeGate] = []

    gates.append(
        QuantitativeGate(
            name="complete_experiment_field",
            passed=not missing_outcome_ids,
            detail=(
                "every planned experiment has a persisted outcome"
                if not missing_outcome_ids
                else f"missing outcomes for: {list(missing_outcome_ids)}"
            ),
            hard=True,
        )
    )

    leakage_ok = all(c.passed for c in report.leakage_checks)
    failed = [c.name for c in report.leakage_checks if not c.passed]
    gates.append(
        QuantitativeGate(
            name="leakage_checks_pass",
            passed=leakage_ok,
            detail="all leakage checks passed" if leakage_ok else f"failed: {failed}",
            hard=True,
        )
    )
    gates.append(
        QuantitativeGate(
            name="holdout_untouched",
            passed=report.holdout_untouched,
            detail=f"holdout_untouched={report.holdout_untouched}",
            hard=True,
        )
    )
    primary_delta = next((d for d in report.baseline_deltas if d.metric == primary_metric), None)
    if primary_delta is None:
        gates.append(
            QuantitativeGate(
                name="beats_dummy_significantly",
                passed=False,
                detail=f"no baseline delta recorded for primary metric '{primary_metric}'",
                hard=True,
            )
        )
    else:
        gates.append(
            QuantitativeGate(
                name="beats_dummy_significantly",
                passed=bool(primary_delta.significant),
                detail=(
                    f"{primary_metric}: dummy={primary_delta.baseline_value:.4f} -> "
                    f"model={primary_delta.model_value:.4f}, "
                    f"significant={primary_delta.significant}"
                ),
                hard=True,
            )
        )
    ece = report.calibration.expected_calibration_error
    if ece is not None:
        # Only add this gate when calibration was actually measured (out-of-
        # fold ECE, evaluation/calibration.py). A gate that is ALWAYS
        # unmeasured and therefore ALWAYS passes is audit theater -- worse
        # than no gate, since its presence implies calibration was checked.
        gates.append(
            QuantitativeGate(
                name="calibration_acceptable",
                passed=ece <= 0.10,
                detail=f"ECE={ece:.4f} (threshold 0.10)",
                hard=False,
            )
        )
    relative_error = _relative_error_context(report, profile, target_column)
    if relative_error is not None:
        gates.append(relative_error)
    return tuple(gates)


def _enforce_verdict(
    draft: CritiqueDraft,
    gates: tuple[QuantitativeGate, ...],
    *,
    fitness_findings: tuple[Finding, ...] = (),
    tolerance_violated: bool = False,
) -> tuple[Verdict, tuple[Finding, ...], tuple[str, ...]]:
    """Code, not the LLM, has the final word on the verdict."""
    findings = [
        Finding(
            severity=f.severity,
            category=f.category,
            evidence=f.evidence,
            recommendation=f.recommendation,
        )
        for f in draft.findings
    ]
    findings.extend(fitness_findings)
    actions = list(draft.iteration_actions)
    verdict = draft.verdict

    safety_failed = any(
        not g.passed
        and g.name in ("leakage_checks_pass", "holdout_untouched", "complete_experiment_field")
        for g in gates
    )
    dummy_failed = any(not g.passed and g.name == "beats_dummy_significantly" for g in gates)

    required = (
        Verdict.HALT
        if safety_failed
        else Verdict.ITERATE
        if dummy_failed or tolerance_violated
        else None
    )
    failed_hard = "; ".join(g.detail for g in gates if not g.passed and g.hard)
    if required is not None and verdict is not required:
        if verdict is Verdict.HALT:
            # Escalating beyond what gates require is allowed (never downgrade
            # a halt), but it must be audited — a halt may not be unexplained.
            _log.warning("critic_escalated_to_halt", gates_required=required.value)
            findings.append(
                Finding(
                    severity=FindingSeverity.HIGH,
                    category="gate_note",
                    evidence=failed_hard or "no hard gate failed",
                    recommendation=(
                        f"Critic chose halt although gates only required "
                        f"{required.value}; human review advised"
                    ),
                )
            )
        else:
            _log.warning(
                "critic_verdict_overridden",
                llm_verdict=verdict.value,
                enforced=required.value,
            )
            findings.append(
                Finding(
                    severity=FindingSeverity.CRITICAL,
                    category="gate_override",
                    evidence=failed_hard or "stated error tolerance violated (no hard gate failed)",
                    recommendation=(
                        "verdict overridden by deterministic quantitative gates — "
                        "the Critic may not approve past a failed hard gate or a "
                        "violated, explicitly stated error tolerance"
                    ),
                )
            )
            verdict = required
    if verdict in (Verdict.ITERATE, Verdict.HALT) and not actions:
        actions = [f"address failed gate: {g.name} ({g.detail})" for g in gates if not g.passed]
        if tolerance_violated:
            actions.append("reduce error until it is within the stated business tolerance")
    return verdict, tuple(findings), tuple(actions)


def _outcome_table(outcomes: tuple[ExperimentOutcome, ...]) -> list[dict[str, object]]:
    return [
        {
            "experiment_id": o.experiment_id,
            "template_id": o.template_id,
            "state": o.state.value,
            "metrics": {
                m.name: {"mean": round(m.mean, 6), "std": round(m.std, 6)} for m in o.cv_metrics
            },
            "wall_time_seconds": round(o.wall_time_seconds, 2),
            "error": o.error,
        }
        for o in outcomes
    ]


def critique_evaluation(
    client: LLMClient,
    store: ArtifactStore,
    *,
    report: EvaluationReport,
    primary_metric: str,
    run_id: str,
    outcomes: tuple[ExperimentOutcome, ...] = (),
    planned_experiment_ids: tuple[str, ...] = (),
    specs: tuple[ExperimentSpec, ...] = (),
    excluded_templates: tuple[ExcludedTemplate, ...] = (),
    winner_experiment_id: str | None = None,
    dummy_experiment_id: str | None = None,
    profile: DataProfile | None = None,
    problem: ProblemSpec | None = None,
) -> CritiqueReport:
    """Run the Critic and store its (gate-enforced) CritiqueReport.

    ``outcomes``/``planned_experiment_ids``/``specs``/``excluded_templates``/
    ``winner_experiment_id``/``dummy_experiment_id`` are all optional so a
    bare best-vs-dummy call still works (e.g. mid-iteration re-critiques that
    don't have a fresh excluded_templates list); when given, the critic sees
    the full field instead of just the winner and the dummy floor.

    ``profile``/``problem``, when given (JUDGMENT_ROADMAP.md Task 3), add
    the report-only ``relative_error_context`` gate and the deterministic
    fitness-for-purpose check (``_fitness_findings``) — a stated-and-
    violated error tolerance forces the verdict away from approve; an
    absent tolerance with a high relative error adds a high-severity
    "fitness unknown" finding without forcing anything (beats-dummy remains
    a valid floor on its own).
    """
    audit = audit_terminal_states(planned_experiment_ids, outcomes)
    target_column = problem.target_column if problem is not None else None
    gates = compute_quantitative_gates(
        report,
        primary_metric,
        missing_outcome_ids=audit.missing_ids,
        profile=profile,
        target_column=target_column,
    )
    relative_error_ratio = _relative_error_ratio(report, profile, target_column)
    fitness_findings, tolerance_violated = _fitness_findings(problem, relative_error_ratio)

    failures = compute_failure_census(outcomes)
    margin = (
        compute_winner_margin(outcomes, primary_metric, winner_experiment_id, dummy_experiment_id)
        if winner_experiment_id and dummy_experiment_id
        else None
    )
    cost = compute_cost_census(outcomes, specs) if specs else None

    result = client.structured(
        agent="critic",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "primary_metric": primary_metric,
            "quantitative_gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail, "hard": g.hard}
                for g in gates
            ],
            "evaluation_report": evaluation_summary(report),
            "outcome_table": _outcome_table(outcomes),
            "terminal_state_audit": {
                "complete": audit.complete,
                "missing_ids": list(audit.missing_ids),
            },
            "failure_census": {
                "completed": failures.completed,
                "failed": failures.failed,
                "skipped": failures.skipped,
                "failure_reasons": list(failures.failure_reasons),
            },
            "winner_margin": (
                {
                    "runner_up_template": margin.runner_up_template,
                    "vs_runner_up_raw": margin.vs_runner_up_raw,
                    "vs_runner_up_stds": margin.vs_runner_up_stds,
                    "vs_dummy_raw": margin.vs_dummy_raw,
                    "vs_dummy_stds": margin.vs_dummy_stds,
                }
                if margin is not None
                else None
            ),
            "cost_census": (
                {
                    "total_wall_time_seconds": cost.total_wall_time_seconds,
                    "estimated_total_seconds": cost.estimated_total_seconds,
                    "over_estimate": cost.over_estimate,
                }
                if cost is not None
                else None
            ),
            "excluded_templates": [
                {"template_id": e.template_id, "reason": e.reason} for e in excluded_templates
            ],
            "decision_context": (
                {
                    "decision_use": problem.decision_use,
                    "error_tolerance_note": problem.error_tolerance_note,
                    "human_baseline_note": problem.human_baseline_note,
                }
                if problem is not None
                else None
            ),
        },
        schema=CritiqueDraft,
    )
    verdict, findings, actions = _enforce_verdict(
        result.output,
        gates,
        fitness_findings=fitness_findings,
        tolerance_violated=tolerance_violated,
    )
    critique = new_artifact(
        CritiqueReport,
        producer_agent="critic",
        run_id=run_id,
        parent_ids=(report.meta.artifact_id,),
        evaluation_ref=report.ref(),
        findings=findings,
        verdict=verdict,
        iteration_actions=actions,
    )
    store.put(critique)
    return critique
