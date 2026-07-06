"""ExperimentPlan: what to train, in what order, at what estimated cost.

Human Gate 2 approves this plan (with its budget) before training starts.
A validator enforces the 'cheap baselines always first' rule structurally.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlteam.contracts.base import Artifact
from mlteam.contracts.decision import DecisionRecord


class BudgetEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_time_seconds: float = Field(ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1, description="Id in the vetted template registry")
    config: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetEstimate
    is_baseline: bool = False


class ExcludedTemplate(BaseModel):
    """A template family the planner deliberately did NOT schedule, and why.

    Makes the plan's negative space auditable — a reviewer (human or the
    critic) can ask "was skipping SVM here actually justified?" instead of
    just seeing the templates that were chosen.
    """

    model_config = ConfigDict(frozen=True)

    template_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExperimentPlan(Artifact):
    experiments: tuple[ExperimentSpec, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    human_approved: bool = False
    excluded_templates: tuple[ExcludedTemplate, ...] = ()
    # WHY this template set was chosen for THIS data profile — optional so
    # artifacts stored before this field existed still load unchanged.
    reasoning: DecisionRecord | None = None

    @model_validator(mode="after")
    def _baselines_first(self) -> ExperimentPlan:
        if not any(e.is_baseline for e in self.experiments):
            raise ValueError("ExperimentPlan must include at least one baseline experiment")
        seen_non_baseline = False
        for exp in self.experiments:
            if not exp.is_baseline:
                seen_non_baseline = True
            elif seen_non_baseline:
                raise ValueError("baseline experiments must precede all non-baseline experiments")
        return self

    @property
    def total_budget(self) -> BudgetEstimate:
        return BudgetEstimate(
            wall_time_seconds=sum(e.budget.wall_time_seconds for e in self.experiments),
            cost_usd=sum(e.budget.cost_usd for e in self.experiments),
            tokens=sum(e.budget.tokens for e in self.experiments),
        )
