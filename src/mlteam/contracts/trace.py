"""RunTrace: the per-run audit log of every decision, call, and cost."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, utcnow


class TraceEventType(StrEnum):
    AGENT_DECISION = "agent_decision"
    TOOL_CALL = "tool_call"
    ARTIFACT_PRODUCED = "artifact_produced"
    LLM_CALL = "llm_call"
    HUMAN_GATE = "human_gate"
    GUARD_ACCESS_ATTEMPT = "guard_access_attempt"
    ERROR = "error"


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=utcnow)
    event_type: TraceEventType
    agent: str = Field(min_length=1)
    message: str = ""
    artifact_id: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunTrace(Artifact):
    events: tuple[TraceEvent, ...] = ()
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
