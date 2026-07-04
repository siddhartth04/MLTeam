"""TriageReport: the data-readiness decisions made before the pipeline runs.

Real-world datasets routinely violate the pipeline's trainability contract —
unlabeled rows, a regression target stored as text ("₹45,00,000", "45.5 Lac"),
numeric feature columns polluted with bucket labels ("> 10"). A senior ML
engineer doesn't crash on these and doesn't silently mangle them either: they
apply a small set of defensible fixes, record exactly what was changed and
why, and halt with a clear explanation only when the data is truly hopeless.

This artifact is that record. Every issue found and every remedy applied is
listed with affected-row counts, so Gate 2 reviewers and the dashboard see
precisely how the modeling dataset differs from what the user uploaded.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlteam.contracts.base import Artifact, ArtifactRef


class TriageIssueCode(StrEnum):
    TARGET_MISSING_ROWS = "target_missing_rows"
    TARGET_NOT_NUMERIC = "target_not_numeric"
    NUMERIC_STRING_COLUMN = "numeric_string_column"
    TOO_FEW_LABELED_ROWS = "too_few_labeled_rows"


class TriageAction(StrEnum):
    DROP_UNLABELED_ROWS = "drop_unlabeled_rows"
    COERCE_TARGET_NUMERIC = "coerce_target_numeric"
    COERCE_COLUMN_NUMERIC = "coerce_column_numeric"


class TriageIssue(BaseModel):
    """One detected trainability problem."""

    model_config = ConfigDict(frozen=True)

    code: TriageIssueCode
    column: str
    detail: str
    affected_rows: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)


class TriageRemedy(BaseModel):
    """One applied fix, with its justification and blast radius."""

    model_config = ConfigDict(frozen=True)

    action: TriageAction
    column: str
    affected_rows: int = Field(ge=0)
    justification: str = Field(min_length=1)


class TriageReport(Artifact):
    """The full triage outcome for one uploaded dataset."""

    dataset_ref: ArtifactRef
    triaged_dataset_ref: ArtifactRef  # equals dataset_ref when nothing changed
    issues: tuple[TriageIssue, ...] = ()
    remedies: tuple[TriageRemedy, ...] = ()
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
