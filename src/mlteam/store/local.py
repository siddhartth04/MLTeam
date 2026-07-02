"""Local filesystem artifact store with content-hash addressing.

Layout under the store root:
    objects/<artifact_id>.json   # {"artifact_type": ..., "data": <model dump>}
    blobs/<sha256>               # opaque bytes
    lineage.db                   # SQLite provenance index
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import structlog

from mlteam.contracts import ARTIFACT_REGISTRY, Artifact, ArtifactRef
from mlteam.store.interface import ArtifactNotFoundError
from mlteam.store.lineage import LineageIndex

_log = structlog.get_logger(__name__)

_HASH_CHUNK_BYTES = 1 << 20


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write via temp file + rename so readers never see partial content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class LocalArtifactStore:
    """Filesystem implementation of the ArtifactStore protocol."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._objects = root / "objects"
        self._blobs = root / "blobs"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._blobs.mkdir(parents=True, exist_ok=True)
        self.lineage = LineageIndex(root / "lineage.db")

    @property
    def root(self) -> Path:
        return self._root

    # -- artifacts ----------------------------------------------------------

    def put(self, artifact: Artifact) -> ArtifactRef:
        artifact_id = artifact.meta.artifact_id
        target = self._objects / f"{artifact_id}.json"
        if not target.exists():
            document = {
                "artifact_type": artifact.meta.artifact_type,
                "data": artifact.model_dump(mode="json"),
            }
            _atomic_write_bytes(target, json.dumps(document, indent=2).encode("utf-8"))
            _log.info(
                "artifact_stored",
                artifact_id=artifact_id,
                artifact_type=artifact.meta.artifact_type,
                producer=artifact.meta.producer_agent,
            )
        self.lineage.record(artifact.meta)
        return artifact.ref()

    def get(self, artifact_id: str) -> Artifact:
        target = self._objects / f"{artifact_id}.json"
        if not target.exists():
            raise ArtifactNotFoundError(f"artifact not found: {artifact_id}")
        document = json.loads(target.read_text(encoding="utf-8"))
        artifact_type = document["artifact_type"]
        cls = ARTIFACT_REGISTRY.get(artifact_type)
        if cls is None:
            raise ArtifactNotFoundError(f"unknown artifact type: {artifact_type}")
        return cls.model_validate(document["data"])

    def exists(self, artifact_id: str) -> bool:
        return (self._objects / f"{artifact_id}.json").exists()

    def list_ids(self, artifact_type: str | None = None) -> tuple[str, ...]:
        return self.lineage.list_ids(artifact_type)

    # -- blobs --------------------------------------------------------------

    def put_blob(self, data: bytes) -> str:
        blob_hash = hashlib.sha256(data).hexdigest()
        target = self._blobs / blob_hash
        if not target.exists():
            _atomic_write_bytes(target, data)
        return blob_hash

    def put_blob_file(self, path: Path) -> str:
        """Single streaming pass: hash and copy chunk-by-chunk to a unique
        temp file, then atomically rename into place."""
        digest = hashlib.sha256()
        fd, tmp_name = tempfile.mkstemp(dir=self._blobs, suffix=".tmp")
        try:
            with path.open("rb") as source, os.fdopen(fd, "wb") as sink:
                while chunk := source.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    sink.write(chunk)
            blob_hash = digest.hexdigest()
            target = self._blobs / blob_hash
            if target.exists():
                Path(tmp_name).unlink(missing_ok=True)
            else:
                os.replace(tmp_name, target)
            return blob_hash
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def blob_path(self, blob_hash: str) -> Path:
        target = self._blobs / blob_hash
        if not target.exists():
            raise ArtifactNotFoundError(f"blob not found: {blob_hash}")
        return target

    def blob_exists(self, blob_hash: str) -> bool:
        return (self._blobs / blob_hash).exists()
