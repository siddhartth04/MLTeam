"""API settings: where the store lives, and demo/CORS toggles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLTEAM_API_", env_file=".env", extra="ignore")

    store_root: Path = Path("./artifact_store")
    upload_dir: Path = Path("./uploads")
    seed_demo: bool = True
    # The Next.js dev server origin; CORS is permissive in dev, tighten in prod.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    anthropic_model: str = "claude-sonnet-4-6"


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
