"""Human approval gates.

Three gates (problem spec, training plan + cost, pre-deployment sign-off).
Backends: Prefect pause (production flows, surfaced in the dashboard from
Phase 6), console prompt (CLI runs), auto-approve (tests/examples — LOUDLY
logged, never a silent default).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog

_log = structlog.get_logger(__name__)


class GateName(StrEnum):
    PROBLEM_SPEC = "gate1_problem_spec"
    TRAINING_PLAN = "gate2_training_plan"
    DEPLOYMENT = "gate3_deployment"


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    notes: str = ""
    approver: str = ""


class GateRejectedError(RuntimeError):
    """Raised by the pipeline when a human gate is not approved."""

    def __init__(self, gate: GateName, decision: GateDecision) -> None:
        super().__init__(f"{gate.value} rejected: {decision.notes or 'no reason given'}")
        self.gate = gate
        self.decision = decision


@runtime_checkable
class ApprovalGate(Protocol):
    def request(self, gate: GateName, summary: dict[str, Any]) -> GateDecision: ...


class AutoApproveGate:
    """Approves everything. For tests and unattended examples ONLY."""

    def request(self, gate: GateName, summary: dict[str, Any]) -> GateDecision:
        _log.warning(
            "gate_auto_approved",
            gate=gate.value,
            hint="AutoApproveGate is for tests/examples; use a human gate in production",
        )
        return GateDecision(approved=True, approver="auto-approve")


class ConsoleGate:
    """Interactive y/n prompt for CLI runs."""

    def request(self, gate: GateName, summary: dict[str, Any]) -> GateDecision:
        import json

        print(f"\n=== HUMAN GATE: {gate.value} ===")
        print(json.dumps(summary, indent=2, default=str))
        answer = input("Approve? [y/N]: ").strip().lower()
        approved = answer in ("y", "yes")
        notes = "" if approved else input("Reason for rejection (optional): ").strip()
        return GateDecision(approved=approved, notes=notes, approver="console")


class PrefectPauseGate:
    """Pauses the enclosing Prefect flow run until a human resumes it with a
    decision. Must be called from inside a flow run; the pause is surfaced in
    the Prefect UI (and the Streamlit dashboard from Phase 6)."""

    def __init__(self, timeout_seconds: int = 3600) -> None:
        self._timeout = timeout_seconds

    def request(self, gate: GateName, summary: dict[str, Any]) -> GateDecision:
        from prefect.flow_runs import pause_flow_run
        from prefect.input import RunInput

        class GateApprovalInput(RunInput):
            approved: bool = False
            notes: str = ""

        _log.info("gate_pausing_flow", gate=gate.value, summary=summary)
        received = pause_flow_run(
            wait_for_input=GateApprovalInput.with_initial_data(
                description=f"Human gate: {gate.value}\n\n```json\n{summary}\n```"
            ),
            timeout=self._timeout,
        )
        return GateDecision(approved=received.approved, notes=received.notes, approver="prefect-ui")


def require_approval(gate_backend: ApprovalGate, gate: GateName, summary: dict[str, Any]) -> None:
    """Request a decision and raise GateRejectedError unless approved."""
    decision = gate_backend.request(gate, summary)
    if not decision.approved:
        raise GateRejectedError(gate, decision)
    _log.info("gate_approved", gate=gate.value, approver=decision.approver)
