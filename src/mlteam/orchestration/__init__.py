"""Orchestration: the deterministic pipeline core + Prefect flow wrappers.

``run_tabular_pipeline`` is the plain-Python core (fast, test-friendly);
``tabular_baseline_flow`` wraps the same agent functions as cached Prefect
tasks. Prefect is imported lazily via the flows module so importing this
package stays cheap.
"""

from mlteam.orchestration.pipeline import (
    PipelineResult,
    default_experiment_specs,
    new_run_id,
    run_tabular_pipeline,
)

__all__ = [
    "PipelineResult",
    "default_experiment_specs",
    "new_run_id",
    "run_tabular_pipeline",
]
