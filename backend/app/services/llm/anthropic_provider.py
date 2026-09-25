"""Anthropic (Claude) provider, using the official `anthropic` SDK."""

from __future__ import annotations

from typing import Any

from app.services.llm.base import LLMResponse, PermanentLLMError, TransientLLMError

TRANSIENT_STATUS_CODES = {408, 409, 429}  # plus every 5xx (includes 529 "overloaded")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, effort: str | None = None, client: Any = None) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self._anthropic = anthropic
        # max_retries=0: the SDK would retry silently; retries are owned by run_structured so they are counted.
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=0)

    def complete(self, system: str, user: str, *, timeout: float, max_output_tokens: int) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "timeout": timeout,
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}

        anthropic = self._anthropic
        try:
            response = self._client.messages.create(**request)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise TransientLLMError(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            message = f"{type(exc).__name__} (HTTP {exc.status_code}): {exc}"
            if exc.status_code in TRANSIENT_STATUS_CODES or exc.status_code >= 500:
                raise TransientLLMError(message) from exc
            raise PermanentLLMError(message) from exc

        if response.stop_reason == "refusal":
            raise PermanentLLMError("model refused the request")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model or self.model,
            stop_reason=response.stop_reason,
        )
