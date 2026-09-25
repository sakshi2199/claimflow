from __future__ import annotations

from pydantic import SecretStr

from app.core.config import Settings
from app.services.llm.base import LLMConfigError, LLMProvider


def _has_value(secret: SecretStr | None) -> bool:
    """Empty strings count as unset (docker compose passes unset variables through as empty)."""
    return secret is not None and bool(secret.get_secret_value().strip())


def build_provider(settings: Settings) -> LLMProvider | None:
    """Create the configured provider, or None when the LLM is disabled (LLM_PROVIDER=none)."""
    if settings.llm_provider == "none":
        return None
    if not settings.llm_model:
        raise LLMConfigError("LLM_MODEL must be set when LLM_PROVIDER is enabled")

    if settings.llm_provider == "anthropic":
        if not _has_value(settings.anthropic_api_key):
            raise LLMConfigError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic")
        from app.services.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings.anthropic_api_key.get_secret_value(), settings.llm_model, effort=settings.llm_effort
        )

    if settings.llm_provider == "openai":
        if not _has_value(settings.openai_api_key):
            raise LLMConfigError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        from app.services.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            settings.openai_api_key.get_secret_value(), settings.llm_model, base_url=settings.openai_base_url
        )

    raise LLMConfigError(f"unsupported LLM_PROVIDER {settings.llm_provider!r}")
