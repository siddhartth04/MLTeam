"""Shared result containers passed between training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlteam.contracts import (
    ExperimentOutcome,
    ExperimentState,
    MetricSummary,
    TrainedModel,
    new_artifact,
)
from mlteam.store.interface import ArtifactStore


@dataclass(frozen=True)
class TrainResult:
    """A trained model artifact plus its CV metric summaries."""

    model: TrainedModel
    metrics: dict[str, MetricSummary]


def store_experiment_outcome(
    store: ArtifactStore,
    *,
    experiment_id: str,
    template_id: str,
    state: ExperimentState,
    run_id: str,
    config: dict[str, Any] | None = None,
    result: TrainResult | None = None,
    wall_time_seconds: float = 0.0,
    error: str | None = None,
) -> ExperimentOutcome:
    """Persist the terminal-state record for one experiment attempt.

    Called for every specced experiment — completed, failed, or skipped —
    so the leaderboard can distinguish "scored worse" from "crashed" from
    "never ran" instead of rendering all three as a blank row.
    """
    outcome = new_artifact(
        ExperimentOutcome,
        producer_agent="trainer",
        run_id=run_id,
        parent_ids=(result.model.meta.artifact_id,) if result is not None else (),
        experiment_id=experiment_id,
        template_id=template_id,
        state=state,
        config=dict(config or (result.model.config if result is not None else {})),
        cv_metrics=tuple(result.metrics.values()) if result is not None else (),
        wall_time_seconds=wall_time_seconds,
        error=error,
        model_ref=result.model.ref() if result is not None else None,
    )
    store.put(outcome)
    return outcome
