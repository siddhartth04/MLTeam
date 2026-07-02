"""Base artifact plumbing: metadata, references, and content hashing.

Every artifact in the system is an immutable Pydantic model carrying an
``ArtifactMeta``. Artifact identity is a content hash of the payload plus the
producing config hash, so identical work always yields the identical id
(which is what makes hash-based memoization possible downstream).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"

_HASH_CHUNK_BYTES = 1 << 20


def utcnow() -> datetime:
    """Timezone-aware current time; the only clock artifacts are allowed to use."""
    return datetime.now(UTC)


def canonical_json(payload: Any) -> str:
    """Deterministic, RFC 8259-compliant JSON encoding used for all hashing.

    NaN/Infinity are rejected: they would produce non-standard JSON tokens and
    non-portable hashes. Producers must convert them to None explicitly.
    """
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
        )
    except ValueError as exc:
        raise ValueError(
            "artifact payload contains NaN or Infinity, which cannot be hashed "
            "portably; replace with None before creating the artifact"
        ) from exc


def content_hash(payload: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_content_hash(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactMeta(BaseModel):
    """Provenance attached to every artifact; recorded in the lineage index."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    artifact_type: str
    parent_ids: tuple[str, ...] = ()
    producer_agent: str
    config_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)


class ArtifactRef(BaseModel):
    """Lightweight pointer to an artifact in the store."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    artifact_type: str


class Artifact(BaseModel):
    """Base class for every artifact contract. Subclasses add payload fields."""

    model_config = ConfigDict(frozen=True)

    meta: ArtifactMeta

    @classmethod
    def artifact_type(cls) -> str:
        return cls.__name__

    def payload_dict(self) -> dict[str, Any]:
        """JSON-mode payload excluding meta; the basis of the content hash."""
        return self.model_dump(mode="json", exclude={"meta"})

    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.meta.artifact_id,
            artifact_type=self.meta.artifact_type,
        )

    def guard_protected_items(self) -> tuple[tuple[str, str], ...]:
        """(item_id, reason) pairs the store guard must protect on put.

        Default: nothing. Artifact types holding leakage-sensitive data (e.g.
        holdout datasets) override this so the guard needs no per-type logic.
        """
        return ()


ArtifactT = TypeVar("ArtifactT", bound=Artifact)


def new_artifact(
    cls: type[ArtifactT],
    *,
    producer_agent: str,
    run_id: str,
    parent_ids: tuple[str, ...] = (),
    config_hash: str = "",
    **payload: Any,
) -> ArtifactT:
    """Validate a payload and mint an artifact with a content-derived id.

    The id covers artifact type, payload, and config hash — deliberately NOT
    run_id or timestamps — so re-running identical work deduplicates.
    """
    placeholder = ArtifactMeta(
        artifact_id="pending",
        artifact_type=cls.artifact_type(),
        parent_ids=tuple(parent_ids),
        producer_agent=producer_agent,
        config_hash=config_hash,
        run_id=run_id,
    )
    draft = cls(meta=placeholder, **payload)
    artifact_id = content_hash(
        {
            "type": cls.artifact_type(),
            "payload": draft.payload_dict(),
            "config_hash": config_hash,
        }
    )
    final_meta = placeholder.model_copy(update={"artifact_id": artifact_id})
    return draft.model_copy(update={"meta": final_meta})
