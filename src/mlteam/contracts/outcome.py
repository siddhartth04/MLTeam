"""ExperimentOutcome: the terminal-state record of one training attempt.

Every experiment a plan schedules — baseline, planned candidate, HPO retrain,
plan-revision addition — reaches exactly one of three terminal states, and
that state is PERSISTED, not just logged. Before this contract existed, only
the winning model's evaluation was ever stored; every other experiment's
metrics lived in an in-memory dict that vanished when the process moved on,
and a training failure was recorded only as a log line. A dashboard built on
top of that could not tell "this model scored worse" apart from "this model
crashed" apart from "this model never ran" — all three rendered as a blank
leaderboard row. ExperimentOutcome makes the three states first-class and
queryable per run, so the leaderboard and the critic can see the whole field.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from mlteam.contracts.base import Artifact, ArtifactRef
from mlteam.contracts.evaluation import MetricSummary


class ExperimentState(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExperimentOutcome(Artifact):
    """Terminal-state record for one experiment attempt within a run."""

    experiment_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    state: ExperimentState
    config: dict[str, Any] = Field(default_factory=dict)
    cv_metrics: tuple[MetricSummary, ...] = ()
    wall_time_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    model_ref: ArtifactRef | None = None
