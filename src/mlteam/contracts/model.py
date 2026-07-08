"""TrainedModel: a fully serialized pipeline (preprocessing + model, one object)."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from mlteam.contracts.base import Artifact, ArtifactRef


class TrainedModel(Artifact):
    # mlflow_run_id is a per-run external tracking handle; hashing it would
    # give identical training work a different identity on every run.
    HASH_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({"mlflow_run_id"})

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
    # Regression-only target transform applied before fitting (e.g. "log1p"),
    # or None for a raw-scale fit. INCLUDED in hashing (not HASH_EXCLUDED) --
    # a transformed model is a genuinely different model. Optional-with-
    # default so models stored before this field existed still load.
    target_transform: str | None = None
