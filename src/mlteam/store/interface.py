"""ArtifactStore protocol — the seam that lets S3/MinIO replace local disk later."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from mlteam.contracts import Artifact, ArtifactRef


class ArtifactNotFoundError(KeyError):
    """Raised when an artifact or blob id is not present in the store."""


@runtime_checkable
class ArtifactStore(Protocol):
    """Typed artifact + blob storage with content-hash addressing.

    Artifacts are JSON documents validated against their Pydantic contract.
    Blobs are opaque bytes (datasets, serialized pipelines, reports) addressed
    by the SHA-256 of their content.
    """

    @property
    def root(self) -> Path:
        """The store's root directory on disk."""
        ...

    def put(self, artifact: Artifact) -> ArtifactRef:
        """Persist an artifact and record its lineage. Idempotent by content."""
        ...

    def get(self, artifact_id: str) -> Artifact:
        """Load and validate an artifact by id.

        Raises ArtifactNotFoundError if absent.
        """
        ...

    def exists(self, artifact_id: str) -> bool: ...

    def list_ids(self, artifact_type: str | None = None) -> tuple[str, ...]: ...

    def put_blob(self, data: bytes) -> str:
        """Store opaque bytes; returns the content hash (blob id)."""
        ...

    def put_blob_file(self, path: Path) -> str:
        """Store a file's bytes as a blob without loading it all into memory."""
        ...

    def blob_path(self, blob_hash: str) -> Path:
        """Path to a blob's read-only content.

        Raises ArtifactNotFoundError if absent.
        """
        ...

    def blob_exists(self, blob_hash: str) -> bool: ...
