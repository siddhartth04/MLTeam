"""CleaningManifest: ordered, declarative, justified cleaning operations.

Source data is never mutated; a deterministic executor applies these ops to
produce a NEW DatasetVersion. Op types form the whitelist the LLM proposer
must choose from — anything outside it is rejected at validation time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class CleaningOpType(StrEnum):
    DROP_COLUMN = "drop_column"
    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    DROP_HIGH_MISSING_ROWS = "drop_high_missing_rows"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    IMPUTE_CONSTANT = "impute_constant"
    CLIP_OUTLIERS = "clip_outliers"
    CAST_DTYPE = "cast_dtype"
    PARSE_DATETIME = "parse_datetime"
    STRIP_WHITESPACE = "strip_whitespace"
    STANDARDIZE_MISSING_TOKENS = "standardize_missing_tokens"


class CleaningOp(BaseModel):
    """One declarative, reversible cleaning step with its justification."""

    model_config = ConfigDict(frozen=True)

    op_type: CleaningOpType
    params: dict[str, Any] = Field(default_factory=dict)
    justification: str = Field(min_length=1)
    affected_columns: tuple[str, ...] = ()
    affected_rows: int = Field(default=0, ge=0)


class CleaningManifest(Artifact):
    input_dataset_ref: ArtifactRef
    output_dataset_ref: ArtifactRef | None = Field(
        default=None, description="Set after the executor materializes the cleaned dataset"
    )
    ops: tuple[CleaningOp, ...]
