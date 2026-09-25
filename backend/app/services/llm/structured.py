"""Calls an LLM provider and returns a validated, grounded LLMDecision, with bounded retries.

Retried (up to `max_retries` extra attempts): transient provider errors, malformed output, and citations of
policy IDs that were not in the retrieved context. Not retried: permanent provider errors (bad key, bad request).
On exhaustion the caller gets `decision=None` plus the failure category and falls back to HUMAN_REVIEW.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.services.llm.base import LLMProvider, PermanentLLMError, TransientLLMError
from app.services.llm.prompts import correction_note
from app.services.llm.schemas import LLMDecision

logger = logging.getLogger(__name__)

OK = "ok"
MALFORMED_OUTPUT = "malformed_output"
INVALID_CITATION = "invalid_citation"
TRANSIENT_ERROR = "transient_error"
PROVIDER_ERROR = "provider_error"


@dataclass
class Attempt:
    number: int
    outcome: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    unsupported_citations: list[str] = field(default_factory=list)
    cited_count: int = 0


@dataclass
class StructuredResult:
    decision: LLMDecision | None
    attempts: list[Attempt]
    failure: str | None  # outcome of the last attempt when no decision was produced
    model: str | None = None

    @property
    def calls(self) -> int:
        return len(self.attempts)

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens or 0 for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens or 0 for a in self.attempts)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences and surrounding prose."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("reply JSON is not an object")
    return parsed


def run_structured(
    provider: LLMProvider,
    system: str,
    user: str,
    allowed_policy_ids: list[str],
    *,
    timeout: float,
    max_retries: int,
    max_output_tokens: int,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> StructuredResult:
    allowed = set(allowed_policy_ids)
    attempts: list[Attempt] = []
    prompt = user
    model: str | None = None
    total_attempts = 1 + max(0, max_retries)  # hard upper bound: never retries indefinitely

    for number in range(1, total_attempts + 1):
        start = time.perf_counter()
        try:
            response = provider.complete(system, prompt, timeout=timeout, max_output_tokens=max_output_tokens)
        except TransientLLMError as exc:
            attempts.append(Attempt(number, TRANSIENT_ERROR, _ms(start), error=str(exc)))
            if number < total_attempts:
                sleep(backoff_seconds * 2 ** (number - 1))
            continue
        except PermanentLLMError as exc:
            attempts.append(Attempt(number, PROVIDER_ERROR, _ms(start), error=str(exc)))
            break
        except Exception as exc:  # noqa: BLE001 - a misbehaving provider adapter must not crash the workflow
            logger.exception("Unexpected LLM provider failure")
            attempts.append(Attempt(number, PROVIDER_ERROR, _ms(start), error=f"{type(exc).__name__}: {exc}"))
            break

        latency = _ms(start)
        model = response.model
        tokens = {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}

        try:
            decision = LLMDecision.model_validate(extract_json(response.text))
        except (ValueError, ValidationError) as exc:  # JSONDecodeError and pydantic's ValidationError are ValueErrors
            reason = _short(str(exc))
            attempts.append(Attempt(number, MALFORMED_OUTPUT, latency, error=reason, **tokens))
            prompt = user + correction_note(reason, allowed_policy_ids)
            continue

        unsupported = [pid for pid in decision.cited_policy_ids if pid not in allowed]
        if unsupported:
            attempts.append(
                Attempt(
                    number, INVALID_CITATION, latency, error=f"cited unknown policy ids {unsupported}",
                    unsupported_citations=unsupported, cited_count=len(decision.cited_policy_ids), **tokens,
                )
            )  # fmt: skip
            prompt = user + correction_note(f"you cited policy IDs that were not provided: {unsupported}", allowed_policy_ids)
            continue

        attempts.append(Attempt(number, OK, latency, cited_count=len(decision.cited_policy_ids), **tokens))
        return StructuredResult(decision, attempts, None, model)

    return StructuredResult(None, attempts, attempts[-1].outcome, model)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _short(message: str, limit: int = 300) -> str:
    return " ".join(message.split())[:limit]
