"""Test doubles for the LLM provider. No test in this suite calls a real model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.services.llm.base import LLMResponse


class ScriptedProvider:
    """Answers with whatever `responder(call_number, system, user)` returns: a str, or an Exception to raise."""

    name = "fake"
    model = "fake-model"

    def __init__(self, responder: Callable[[int, str, str], str | Exception], input_tokens: int = 100, output_tokens: int = 20):
        self.responder = responder
        self.calls: list[dict] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(self, system: str, user: str, *, timeout: float, max_output_tokens: int) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "timeout": timeout, "max_output_tokens": max_output_tokens})
        result = self.responder(len(self.calls), system, user)
        if isinstance(result, Exception):
            raise result
        return LLMResponse(result, self.input_tokens, self.output_tokens, self.model, "end_turn")


def policy_ids_in(prompt: str) -> list[str]:
    """Policy ids offered to the model in a prompt, in the order shown."""
    return re.findall(r"^\[(POL-[A-Z0-9-]+)\]", prompt, flags=re.MULTILINE)


REASON_FOR = {
    "APPROVED": "VALID_STANDARD_CLAIM",
    "REJECTED": "DIAGNOSIS_PROCEDURE_MISMATCH",
    "HUMAN_REVIEW": "AMBIGUOUS_CLINICAL_NOTES",
}


def decision_json(decision: str = "APPROVED", confidence: float = 0.9, reason: str | None = None, cited: list[str] | None = None) -> str:
    return json.dumps(
        {
            "decision": decision,
            "confidence": confidence,
            "reason_code": reason or REASON_FOR[decision],
            "reasoning_summary": "Notes reviewed against the provided policy excerpts.",
            "cited_policy_ids": cited if cited is not None else ["POL-DOC-001"],
        }
    )


def answer(decision: str = "APPROVED", confidence: float = 0.9) -> Callable[[int, str, str], str]:
    """Responder that always gives this decision and cites the first policy actually shown in the prompt."""

    def respond(_n: int, _system: str, user: str) -> str:
        return decision_json(decision, confidence, cited=policy_ids_in(user)[:1])

    return respond
