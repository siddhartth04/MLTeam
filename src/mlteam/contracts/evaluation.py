"""EvaluationReport: deterministic evaluation output the Critic reads.

All numbers here come from deterministic code — never from an LLM. The Critic
agent consumes this structured report ONLY (no raw data access).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class MetricSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    fold_values: tuple[float, ...] = Field(min_length=2)
    mean: float
    std: float = Field(ge=0.0)


class BaselineDelta(BaseModel):
    """Model performance relative to a sanity baseline (dummy / default LGBM)."""

    model_config = ConfigDict(frozen=True)

    baseline_name: str
    metric: str
    baseline_value: float
    model_value: float
    delta: float
    significant: bool | None = Field(
        default=None, description="Whether the improvement passes the significance check"
    )
    p_value: float | None = Field(
        default=None,
        description=(
            "One-sided p-value from the paired Nadeau-Bengio corrected "
            "resampled t-test over per-fold deltas, when the model and "
            "baseline were scored on identical folds. None when fold "
            "pairing was unavailable (fold counts differ, or fewer than 2 "
            "folds) -- ``significant`` then falls back to a documented "
            "heuristic instead of a real test."
        ),
    )
    significance_method: str = Field(
        default="none",
        description="'nadeau_bengio_paired_t' when p_value was computed, else 'heuristic_2sigma'",
    )


class SliceMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    slice_name: str
    n_rows: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class CalibrationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_calibration_error: float | None = None
    brier_score: float | None = None
    method: str | None = None


class LeakageCheck(BaseModel):
    """One deterministic leakage detector result (e.g. near-perfect-MI feature,
    holdout-untouched assertion, fold-variance sanity)."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    details: str = ""


class LabelIssueSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_flagged: int = Field(ge=0)
    flagged_fraction: float = Field(ge=0.0, le=1.0)
    method: str = "cleanlab"


class EvaluationReport(Artifact):
    model_ref: ArtifactRef
    cv_metrics: tuple[MetricSummary, ...] = Field(min_length=1)
    baseline_deltas: tuple[BaselineDelta, ...] = ()
    slice_metrics: tuple[SliceMetric, ...] = ()
    calibration: CalibrationSummary = CalibrationSummary()
    leakage_checks: tuple[LeakageCheck, ...] = ()
    label_issues: LabelIssueSummary | None = None
    holdout_untouched: bool = Field(
        description="Asserted from the store guard's access log, not convention"
    )
