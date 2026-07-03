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
    Finding,
    FindingSeverity,
    Verdict,
    new_artifact,
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
metric: verdict MUST be iterate_with_actions, with concrete actions.
- approve ONLY when every hard gate passes.
- iterate_with_actions requires specific, actionable steps (e.g. "add class \
weights", "try target encoding for high-cardinality column X") — not vague \
advice.
For findings, cite the numeric evidence from the report. Severity: critical for \
leakage/holdout issues, high for failing the dummy gate, medium/low for quality \
concerns (calibration, fold variance, slice gaps).
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
    report: EvaluationReport, primary_metric: str
) -> tuple[QuantitativeGate, ...]:
    """Deterministic gate results — the checklist the Critic must respect."""
    gates: list[QuantitativeGate] = []

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
        not g.passed and g.name in ("leakage_checks_pass", "holdout_untouched") for g in gates
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


def critique_evaluation(
    client: LLMClient,
    store: ArtifactStore,
    *,
    report: EvaluationReport,
    primary_metric: str,
    run_id: str,
) -> CritiqueReport:
    """Run the Critic and store its (gate-enforced) CritiqueReport."""
    gates = compute_quantitative_gates(report, primary_metric)
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
