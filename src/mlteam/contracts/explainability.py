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

    kind: str  # "permutation_importance" | "shap_summary" | "partial_dependence"
    title: str
    blob_hash: str = Field(min_length=1)
    feature: str | None = None


class ExplainabilityReport(Artifact):
    model_ref: ArtifactRef
    method: ImportanceMethod
    importances: tuple[FeatureImportance, ...]
    shap_available: bool = False
    figures: tuple[FigureRef, ...] = ()
