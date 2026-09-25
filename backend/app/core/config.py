from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is three levels above this package
REPO_ROOT = Path(__file__).resolve().parents[3]

# Bump when decision logic changes so evaluation runs can be compared across versions.
ENGINE_VERSION = "rules-v1"


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (or a repo-root .env file)."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://claimflow:claimflow@localhost:5432/claimflow"
    data_dir: Path = REPO_ROOT / "data"
    evaluation_output_dir: Path = REPO_ROOT / "evaluation_results"

    @property
    def synthetic_claims_dir(self) -> Path:
        return self.data_dir / "synthetic_claims"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"


@lru_cache
def get_settings() -> Settings:
    return Settings()
