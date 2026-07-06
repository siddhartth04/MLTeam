"""ProblemSpec: the confirmed definition of what we are solving (Human Gate 1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact
from mlteam.contracts.decision import DecisionRecord


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    # Forecasting is direct supervised regression over lag/rolling/calendar
    # features with strictly chronological splitting — never shuffled CV.
    FORECASTING = "forecasting"
    # Labeled anomaly detection (fraud, failure/predictive-maintenance events):
    # binary labels with a rare anomaly class. Dedicated unsupervised detectors
    # (fit on normal rows, scored against labels) compete with the ordinary
    # classifiers; PR-AUC is the headline metric.
    ANOMALY_DETECTION = "anomaly_detection"


def is_classification(task_type: TaskType) -> bool:
    """True for classification-like tasks (incl. labeled anomaly detection);
    regression and forecasting take the regressor branch downstream."""
    return task_type in (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
    )


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
    # Required context for FORECASTING: the datetime column that orders rows.
    # None lets the splitter auto-detect a single Date/Datetime column.
    time_column: str | None = None
    primary_metric: str = Field(min_length=1)
    secondary_metrics: tuple[str, ...] = ()
    business_constraints: tuple[str, ...] = ()
    cost_asymmetry: CostAsymmetry = CostAsymmetry()
    assumptions: tuple[str, ...] = ()
    human_confirmed: bool = False
    # WHY this target/task_type was chosen over alternatives — optional so
    # artifacts stored before this field existed still load unchanged.
    reasoning: DecisionRecord | None = None
