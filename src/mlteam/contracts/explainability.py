"""ExplainabilityReport: global feature-importance + effect artifacts.

Permutation importance, SHAP summary, and partial-dependence figures are
computed on the training data (the holdout stays frozen). Plotly figures are
stored as JSON blobs referenced by hash; this contract keeps their structured
provenance and the ranked importances for the report generator to consume.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class ImportanceMethod(StrEnum):
    PERMUTATION = "permutation"
    SHAP = "shap"


class FeatureImportance(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    importance: float
    std: float = Field(default=0.0, ge=0.0)


class FigureRef(BaseModel):
    """A stored Plotly figure (JSON blob) plus what it depicts."""

    model_config = ConfigDict(frozen=True)

    # "permutation_importance" | "shap_summary" | "partial_dependence"
    # | "residual_hist" | "residual_vs_predicted"
    kind: str
    title: str
    blob_hash: str = Field(min_length=1)
    feature: str | None = None


class ResidualSlice(BaseModel):
    """Per-group error for one value of a categorical feature (regression).

    Surfaces WHERE a regression model is weakest — a group with a much higher
    MAE than the rest is exactly what a data scientist would investigate
    before trusting the model on that segment.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    value: str
    mae: float = Field(ge=0.0)
    n_rows: int = Field(ge=0)


class ExplainabilityReport(Artifact):
    model_ref: ArtifactRef
    method: ImportanceMethod
    importances: tuple[FeatureImportance, ...]
    shap_available: bool = False
    figures: tuple[FigureRef, ...] = ()
    # Regression-only error analysis; empty for classification or when no
    # categorical feature was available to slice on. Optional/defaulted so
    # artifacts stored before this field existed still load.
    residual_slices: tuple[ResidualSlice, ...] = ()
