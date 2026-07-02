"""ProblemSpec: the confirmed definition of what we are solving (Human Gate 1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class CostAsymmetry(BaseModel):
    """Relative business cost of the two error directions (classification)."""

    model_config = ConfigDict(frozen=True)

    false_positive_cost: float = Field(default=1.0, gt=0)
    false_negative_cost: float = Field(default=1.0, gt=0)
    rationale: str = ""


class ProblemSpec(Artifact):
    """Structured problem definition produced by the Requirement Understanding
    agent and confirmed by a human before any pipeline work starts."""

    task_type: TaskType
    target_column: str = Field(min_length=1)
    prediction_horizon: str | None = None
    primary_metric: str = Field(min_length=1)
    secondary_metrics: tuple[str, ...] = ()
    business_constraints: tuple[str, ...] = ()
    cost_asymmetry: CostAsymmetry = CostAsymmetry()
    assumptions: tuple[str, ...] = ()
    human_confirmed: bool = False
