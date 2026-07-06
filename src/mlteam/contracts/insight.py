"""InsightNarrative: plain-language findings a stakeholder can actually read.

The platform produces excellent structured evidence — importances, residual
slices, baseline deltas — but a business stakeholder does not read a
FeatureImportance tuple. This artifact is one LLM pass that turns that
evidence into a few plain-language findings ("carpet area and bathroom count
dominate price; the model is weakest on eastern-region listings") plus honest
caveats. It only ever READS existing artifacts, so it can never change a
model or a metric; the worst it can do on a bad generation is be skipped.
"""

from __future__ import annotations

from mlteam.contracts.base import Artifact, ArtifactRef


class InsightNarrative(Artifact):
    """Narrated findings + caveats derived from evaluation + explainability."""

    model_ref: ArtifactRef
    findings: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
