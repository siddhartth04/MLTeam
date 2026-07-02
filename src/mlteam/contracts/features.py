"""FeaturePlan / FeatureSet: whitelisted, in-pipeline feature engineering.

Feature ops execute INSIDE sklearn Pipelines (fit within CV folds only), so
proposals here are specs, never precomputed columns on full data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class FeatureOpType(StrEnum):
    RATIO = "ratio"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    LOG_TRANSFORM = "log_transform"
    BINNING = "binning"
    DATETIME_DECOMPOSE = "datetime_decompose"
    ONE_HOT_ENCODE = "one_hot_encode"
    ORDINAL_ENCODE = "ordinal_encode"
    TARGET_ENCODE = "target_encode"
    FREQUENCY_ENCODE = "frequency_encode"
    GROUP_AGGREGATE = "group_aggregate"
    TEXT_LENGTH = "text_length"


class FeatureProposal(BaseModel):
    """One proposed feature with the domain rationale for creating it."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    op_type: FeatureOpType
    input_columns: tuple[str, ...] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)


class FeaturePlan(Artifact):
    dataset_ref: ArtifactRef
    proposals: tuple[FeatureProposal, ...]


class FeatureSet(Artifact):
    """The executed outcome of a FeaturePlan: which features made it in."""

    plan_ref: ArtifactRef
    dataset_ref: ArtifactRef
    feature_names: tuple[str, ...]
    dropped_proposals: tuple[str, ...] = Field(
        default=(), description="Proposal names rejected during execution, e.g. failed validation"
    )
    pipeline_spec_blob: str | None = Field(
        default=None, description="Blob hash of the serialized transformer spec"
    )
