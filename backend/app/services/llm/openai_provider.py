"""OpenAI-compatible provider (OpenAI itself, or any server exposing the Chat Completions API)."""

from __future__ import annotations

from typing import Any

from app.services.llm.base import LLMResponse, PermanentLLMError, TransientLLMError

TRANSIENT_STATUS_CODES = {408, 409, 429}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None = None, client: Any = None) -> None:
        import openai

        self.model = model
        self._openai = openai
        # Servers other than OpenAI generally still expect the older `max_tokens` name.
        self._token_param = "max_tokens" if base_url else "max_completion_tokens"
        self._client = client or openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def complete(self, system: str, user: str, *, timeout: float, max_output_tokens: int) -> LLMResponse:
        openai = self._openai
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=timeout,
                **{self._token_param: max_output_tokens},
            )
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransientLLMError(f"{type(exc).__name__}: {exc}") from exc
        except openai.APIStatusError as exc:
            message = f"{type(exc).__name__} (HTTP {exc.status_code}): {exc}"
            if exc.status_code in TRANSIENT_STATUS_CODES or exc.status_code >= 500:
                raise TransientLLMError(message) from exc
            raise PermanentLLMError(message) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            model=response.model or self.model,
            stop_reason=choice.finish_reason,
        )
