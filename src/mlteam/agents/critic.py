"""Critic agent: structured verdict over the EvaluationReport — nothing else.

The Critic reads ONLY the structured EvaluationReport summary. Quantitative
gates are computed deterministically in code; the LLM interprets and writes
findings, but it cannot approve past a failed hard gate — code enforces that,
logging an override finding when it must.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from mlteam.contracts import (
    CritiqueReport,
    EvaluationReport,
    ExcludedTemplate,
    ExperimentOutcome,
    ExperimentSpec,
    Finding,
    FindingSeverity,
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

For findings, cite the numeric evidence from the report. Severity: critical for \
leakage/holdout issues, high for failing the dummy gate, medium/low for quality \
concerns (calibration, fold variance, slice gaps, an unresolved tie, or \
unjustified excluded coverage).
Return only JSON matching the schema."""


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


def compute_quantitative_gates(
    report: EvaluationReport,
    primary_metric: str,
    *,
    missing_outcome_ids: tuple[str, ...] = (),
) -> tuple[QuantitativeGate, ...]:
    """Deterministic gate results — the checklist the Critic must respect.

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
    gates.append(
        QuantitativeGate(
            name="calibration_acceptable",
            passed=ece is None or ece <= 0.10,
            detail="not measured" if ece is None else f"ECE={ece:.4f} (threshold 0.10)",
            hard=False,
        )
    )
    return tuple(gates)


def _enforce_verdict(
    draft: CritiqueDraft, gates: tuple[QuantitativeGate, ...]
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
    actions = list(draft.iteration_actions)
    verdict = draft.verdict

    safety_failed = any(
        not g.passed
        and g.name in ("leakage_checks_pass", "holdout_untouched", "complete_experiment_field")
        for g in gates
    )
    dummy_failed = any(not g.passed and g.name == "beats_dummy_significantly" for g in gates)

    required = Verdict.HALT if safety_failed else Verdict.ITERATE if dummy_failed else None
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
                    evidence=failed_hard,
                    recommendation=(
                        "verdict overridden by deterministic quantitative gates — "
                        "the Critic may not approve past a failed hard gate"
                    ),
                )
            )
            verdict = required
    if verdict in (Verdict.ITERATE, Verdict.HALT) and not actions:
        actions = [f"address failed gate: {g.name} ({g.detail})" for g in gates if not g.passed]
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
) -> CritiqueReport:
    """Run the Critic and store its (gate-enforced) CritiqueReport.

    ``outcomes``/``planned_experiment_ids``/``specs``/``excluded_templates``/
    ``winner_experiment_id``/``dummy_experiment_id`` are all optional so a
    bare best-vs-dummy call still works (e.g. mid-iteration re-critiques that
    don't have a fresh excluded_templates list); when given, the critic sees
    the full field instead of just the winner and the dummy floor.
    """
    audit = audit_terminal_states(planned_experiment_ids, outcomes)
    gates = compute_quantitative_gates(
        report, primary_metric, missing_outcome_ids=audit.missing_ids
    )

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
        },
        schema=CritiqueDraft,
    )
    verdict, findings, actions = _enforce_verdict(result.output, gates)
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
