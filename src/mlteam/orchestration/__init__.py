"""Orchestration: the deterministic pipeline core + Prefect flow wrappers.

``run_tabular_pipeline`` is the plain-Python core (fast, test-friendly);
``tabular_baseline_flow`` wraps the same agent functions as cached Prefect
tasks. Prefect is imported lazily via the flows module so importing this
package stays cheap.
"""

from mlteam.orchestration.agentic import AgenticResult, run_agentic_pipeline
from mlteam.orchestration.gates import (
    ApprovalGate,
    AutoApproveGate,
    ConsoleGate,
    GateDecision,
    GateName,
    GateRejectedError,
    PrefectPauseGate,
    require_approval,
)
from mlteam.orchestration.pipeline import (
    PipelineResult,
    default_experiment_specs,
    new_run_id,
    run_tabular_pipeline,
)

__all__ = [
    "AgenticResult",
    "ApprovalGate",
    "AutoApproveGate",
    "ConsoleGate",
    "GateDecision",
    "GateName",
    "GateRejectedError",
    "PipelineResult",
    "PrefectPauseGate",
    "default_experiment_specs",
    "new_run_id",
    "require_approval",
    "run_agentic_pipeline",
    "run_tabular_pipeline",
]
