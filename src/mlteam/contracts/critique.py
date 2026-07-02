"""CritiqueReport: the Critic's structured verdict on an EvaluationReport."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlteam.contracts.base import Artifact, ArtifactRef


class FindingSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Verdict(StrEnum):
    APPROVE = "approve"
    ITERATE = "iterate_with_actions"
    HALT = "halt"


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: FindingSeverity
    category: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class CritiqueReport(Artifact):
    evaluation_ref: ArtifactRef
    findings: tuple[Finding, ...] = ()
    verdict: Verdict
    iteration_actions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _iterate_requires_actions(self) -> CritiqueReport:
        if self.verdict is Verdict.ITERATE and not self.iteration_actions:
            raise ValueError("verdict iterate_with_actions requires at least one action")
        return self
