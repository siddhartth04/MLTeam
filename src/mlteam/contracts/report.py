"""FinalReport: business + technical HTML reports with embedded Plotly figures."""

from __future__ import annotations

from pydantic import Field

from mlteam.contracts.base import Artifact


class FinalReport(Artifact):
    business_report_blob: str = Field(min_length=1)
    technical_report_blob: str = Field(min_length=1)
    figure_blobs: tuple[str, ...] = Field(
        default=(), description="Blob hashes of JSON-serialized Plotly figures"
    )
    headline_metrics: dict[str, float] = Field(default_factory=dict)
