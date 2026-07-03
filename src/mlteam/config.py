"""Global settings, loaded from environment variables (MLTEAM_ prefix) or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLTeamSettings(BaseSettings):
    """Runtime configuration. Every knob that affects reproducibility or cost
    lives here so it can be logged with each run."""

    model_config = SettingsConfigDict(env_prefix="MLTEAM_", env_file=".env", extra="ignore")

    artifact_store_root: Path = Path("./artifact_store")
    # MLflow 3.x deprecated the ./mlruns file backend; SQLite is the local default.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    trace_dir: Path = Path("./traces")

    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    global_seed: int = 42
    max_iterations: int = Field(default=3, ge=1)
    run_budget_usd: float = Field(default=5.0, gt=0)


def load_settings() -> MLTeamSettings:
    """Load settings fresh from the environment (no global mutable singleton)."""
    return MLTeamSettings()
