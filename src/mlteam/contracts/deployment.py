"""DeploymentBundle: everything needed to serve the model (Human Gate 3)."""

from __future__ import annotations

from pydantic import Field

from mlteam.contracts.base import Artifact, ArtifactRef


class DeploymentBundle(Artifact):
    model_ref: ArtifactRef
    inference_schema_blob: str = Field(
        min_length=1, description="Blob hash of the Pandera schema enforced at inference"
    )
    app_blob: str = Field(min_length=1, description="Blob hash of the generated FastAPI app bundle")
    dockerfile_blob: str | None = None
    drift_baseline_blob: str | None = Field(
        default=None, description="Blob hash of the Evidently drift reference data"
    )
    human_signed_off: bool = False
