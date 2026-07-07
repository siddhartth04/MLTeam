"""Run lifecycle: the agent stages the dashboard visualizes, and run events.

Stages map 1:1 to the agent-flow nodes in the UI. A run publishes StageUpdate
and LogEvent objects as it progresses; the SSE endpoint streams them and the
registry keeps the latest snapshot so a late subscriber sees current state.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from mlteam.contracts.base import utcnow


class Stage(StrEnum):
    """One node in the agent-flow diagram (order = canonical pipeline order)."""

    REQUIREMENT = "requirement"
    GATE_SPEC = "gate_spec"
    INGEST = "ingest"
    PROFILE = "profile"
    SPLIT = "split"
    CLEAN = "clean"
    FEATURES = "features"
    PLAN = "plan"
    GATE_PLAN = "gate_plan"
    TRAIN = "train"
    EVALUATE = "evaluate"
    CRITIC = "critic"
    ITERATE = "iterate"
    EXPLAIN = "explain"
    DEPLOY = "deploy"
    GATE_DEPLOY = "gate_deploy"
    REPORT = "report"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)

STAGE_LABELS: dict[Stage, str] = {
    Stage.REQUIREMENT: "Requirement Understanding",
    Stage.GATE_SPEC: "Gate 1 · Problem Spec",
    Stage.INGEST: "Dataset Ingestion",
    Stage.PROFILE: "Profiling",
    Stage.SPLIT: "Split Strategist",
    Stage.CLEAN: "Data Cleaning",
    Stage.FEATURES: "Feature Engineering",
    Stage.PLAN: "Experiment Planner",
    Stage.GATE_PLAN: "Gate 2 · Training Plan",
    Stage.TRAIN: "Training",
    Stage.EVALUATE: "Evaluation",
    Stage.CRITIC: "Critic",
    Stage.ITERATE: "Iteration Loop",
    Stage.EXPLAIN: "Explainability",
    Stage.DEPLOY: "Deployment",
    Stage.GATE_DEPLOY: "Gate 3 · Sign-off",
    Stage.REPORT: "Report Generator",
}

GATE_STAGES: frozenset[Stage] = frozenset({Stage.GATE_SPEC, Stage.GATE_PLAN, Stage.GATE_DEPLOY})


class StageStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    FAILED = "failed"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageState(BaseModel):
    stage: Stage
    label: str
    status: StageStatus = StageStatus.PENDING
    is_gate: bool = False
    detail: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None


class LogEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utcnow)
    level: Literal["info", "success", "warning", "error", "gate"] = "info"
    stage: Stage | None = None
    message: str
    extra: dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """One SSE payload: either a stage-state change or a log line or a heartbeat."""

    type: Literal["snapshot", "stage", "log", "status", "heartbeat", "done"]
    data: dict[str, Any] = Field(default_factory=dict)


def initial_stage_states() -> list[StageState]:
    return [
        StageState(stage=s, label=STAGE_LABELS[s], is_gate=s in GATE_STAGES) for s in STAGE_ORDER
    ]
