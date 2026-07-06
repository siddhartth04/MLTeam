"""DataProfile: deterministic structured profiling output (computed in Polars)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class WarningCode(StrEnum):
    CONSTANT_COLUMN = "constant_column"
    SUSPECTED_ID = "suspected_id"
    HIGH_CARDINALITY = "high_cardinality"
    HIGH_MISSINGNESS = "high_missingness"
    SUSPECTED_LEAK = "suspected_leak"
    DATETIME_DETECTED = "datetime_detected"
    TARGET_IMBALANCE = "target_imbalance"
    DUPLICATE_ROWS = "duplicate_rows"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ProfileWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: WarningCode
    severity: WarningSeverity
    column: str | None = None
    message: str = Field(min_length=1)


class ColumnProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    cardinality: int = Field(ge=0)
    missing_pct: float = Field(ge=0.0, le=100.0)
    mean: float | None = None
    std: float | None = None
    min_value: str | None = None
    max_value: str | None = None
    top_values: tuple[tuple[str, int], ...] = ()


class CorrelationPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    column_a: str
    column_b: str
    pearson: float = Field(ge=-1.0, le=1.0)


class TargetRelationship(BaseModel):
    """Per-feature association with the target; feeds leakage suspicion.

    ``mutual_information`` is always a bounded [0, 1) NORMALIZED score for
    BOTH classification (MI / target entropy) and regression (the Gaussian-
    equivalent correlation transform sqrt(1 - exp(-2*MI)), Joe 1989) — the
    same 0.9/0.99 thresholds mean the same thing regardless of task type.
    ``spearman`` catches monotone-but-nonlinear leaks (e.g. a rank or
    probability transform of the target) that Pearson's linear correlation
    can miss entirely.
    """

    model_config = ConfigDict(frozen=True)

    column: str
    mutual_information: float | None = None
    correlation: float | None = None
    spearman: float | None = None


class DataProfile(Artifact):
    dataset_ref: ArtifactRef
    columns: tuple[ColumnProfile, ...]
    correlations: tuple[CorrelationPair, ...] = ()
    target_relationships: tuple[TargetRelationship, ...] = ()
    warnings: tuple[ProfileWarning, ...] = ()
