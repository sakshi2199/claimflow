from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is three levels above this package
REPO_ROOT = Path(__file__).resolve().parents[3]

# Bump when decision logic changes so evaluation runs can be compared across versions.
ENGINE_VERSION = "rules-v1"  # Phase 1: deterministic only
ENGINE_VERSION_RAG_LLM = "rag-llm-v1"  # Phase 2: deterministic rules + RAG/LLM note interpretation


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (or a repo-root .env file)."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://claimflow:claimflow@localhost:5432/claimflow"
    data_dir: Path = REPO_ROOT / "data"
    evaluation_output_dir: Path = REPO_ROOT / "evaluation_results"
    chroma_dir: Path = REPO_ROOT / "chroma_db"

    # --- RAG ---
    embedder: Literal["minilm", "hashing"] = "minilm"  # "hashing" is an offline stand-in used by tests
    retrieval_config: str = "filtered"  # preset in app/rag/config.py; chosen by best overall Recall@5 (docs/rag_evaluation.md)

    # --- LLM (disabled unless llm_provider is set) ---
    llm_provider: Literal["none", "anthropic", "openai"] = "none"
    llm_model: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2  # retries after the first attempt, so at most 1 + 2 calls per claim
    llm_backoff_seconds: float = 1.0  # base for exponential backoff between retries
    llm_confidence_threshold: float = 0.80
    # Generous because models with built-in reasoning spend output tokens before the JSON answer.
    llm_max_output_tokens: int = 4000
    llm_effort: str | None = None  # Anthropic only: low|medium|high|xhigh|max; unset = model default
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible server (vLLM, Ollama, ...)
    # Optional pricing (USD per million tokens). Deliberately no defaults: prices change.
    llm_input_price_per_mtok: float | None = None
    llm_output_price_per_mtok: float | None = None

    @property
    def synthetic_claims_dir(self) -> Path:
        return self.data_dir / "synthetic_claims"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def policy_documents_dir(self) -> Path:
        return self.data_dir / "policies" / "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
