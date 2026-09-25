"""Provider-neutral LLM interface. The workflow depends only on this, never on a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """Base class for provider failures."""


class TransientLLMError(LLMError):
    """Worth retrying: timeout, connection problem, rate limit, 5xx."""


class PermanentLLMError(LLMError):
    """Not worth retrying: bad credentials, invalid request, refusal."""


class LLMConfigError(ValueError):
    """The LLM is misconfigured (missing model name or API key)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    model: str
    stop_reason: str | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, *, timeout: float, max_output_tokens: int) -> LLMResponse:
        """Send one request and return the text. Must raise TransientLLMError / PermanentLLMError on failure."""
        ...
