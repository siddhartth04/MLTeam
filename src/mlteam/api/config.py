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
    # Set to any LiteLLM model string (e.g. "openai/gpt-4o", "ollama/llama3",
    # "gemini/gemini-2.5-pro") to run live via LiteLLM instead of the direct
    # Anthropic SDK. Requires `uv sync --extra litellm`. Leave unset to keep
    # using AnthropicTransport for "claude-*" models.
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_api_base: str | None = None

    # Evidence strength (audit finding M4): these were hardcoded at the call
    # site with no way to raise them without editing source. 5, not 3: k=3
    # is the weakest defensible CV, and combined with the paired
    # significance test it makes a genuinely large improvement come out
    # borderline-non-significant purely from having too few folds, not from
    # the improvement being small.
    n_folds: int = 5
    hpo_trials: int = 8
    max_iterations: int = 2
    max_llm_cost_usd: float = 5.0


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
