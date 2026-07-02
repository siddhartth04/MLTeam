"""TrainedModel: a fully serialized pipeline (preprocessing + model, one object)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from mlteam.contracts.base import Artifact, ArtifactRef


class TrainedModel(Artifact):
    experiment_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    pipeline_blob: str = Field(
        min_length=1, description="Blob hash of the serialized sklearn Pipeline"
    )
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int
    env_hash: str = Field(min_length=1, description="Hash of the locked dependency environment")
    train_data_ref: ArtifactRef
    mlflow_run_id: str | None = None
