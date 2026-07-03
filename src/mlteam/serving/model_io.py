"""Load a serialized pipeline (the FULL preprocessing+model object) from a
TrainedModel artifact. One place so serving, explainability, and reporting all
deserialize models identically."""

from __future__ import annotations

from typing import Any

from mlteam.contracts import TrainedModel
from mlteam.store.interface import ArtifactStore


def load_pipeline(store: ArtifactStore, model: TrainedModel) -> Any:
    """Deserialize the sklearn Pipeline stored under model.pipeline_blob."""
    import joblib

    with store.blob_path(model.pipeline_blob).open("rb") as handle:
        return joblib.load(handle)
