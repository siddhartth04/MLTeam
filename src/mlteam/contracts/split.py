"""SplitPlan: how data is split, including the frozen, guard-protected holdout."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from mlteam.contracts.base import Artifact, ArtifactRef


class SplitStrategy(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    TEMPORAL = "temporal"
    GROUP = "group"


class SplitPlan(Artifact):
    """Frozen split decision. ``holdout_ref`` points at a DatasetVersion with
    role=HOLDOUT which the store guard makes unreadable until final reporting.
    """

    strategy: SplitStrategy
    rationale: str = Field(min_length=1)
    n_folds: int = Field(ge=2, le=20)
    seed: int
    holdout_fraction: float = Field(gt=0.0, lt=0.5)
    time_column: str | None = None
    group_column: str | None = None
    train_ref: ArtifactRef
    holdout_ref: ArtifactRef
    fold_assignment_hash: str = Field(
        min_length=1, description="Hash of the exact fold membership, for reproducibility checks"
    )

    @model_validator(mode="after")
    def _strategy_columns_consistent(self) -> SplitPlan:
        if self.strategy is SplitStrategy.TEMPORAL and not self.time_column:
            raise ValueError("temporal strategy requires time_column")
        if self.strategy is SplitStrategy.GROUP and not self.group_column:
            raise ValueError("group strategy requires group_column")
        return self
