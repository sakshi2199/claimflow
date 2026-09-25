"""Phase 2 notes interpretation: policy retrieval + LLM + confidence-based escalation.

Called by the workflow only for claims that passed every deterministic rule but whose clinical notes still
need to be read for meaning. Never raises: any failure becomes a HUMAN_REVIEW fallback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings
from app.models.claim import Claim
from app.models.enums import ClaimOutcome, ReasonCode
from app.rag.retrieval import RetrievalQuery, RetrievalResult, Retriever, build_retriever
from app.services.decision import Decision
from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_provider
from app.services.llm.prompts import SYSTEM_PROMPT, ClaimView, PolicyExcerpt, build_user_prompt
from app.services.llm.structured import (
    INVALID_CITATION,
    MALFORMED_OUTPUT,
    StructuredResult,
    run_structured,
)
from app.services.policies import PolicyData, load_policies
from app.services.rules import keyword_notes_decision

logger = logging.getLogger(__name__)

SOURCE_DETERMINISTIC = "deterministic"
SOURCE_LLM = "llm"
SOURCE_FALLBACK = "human_review_fallback"


def apply_confidence_policy(
    decision: ClaimOutcome, confidence: float, reason: ReasonCode, threshold: float
) -> tuple[ClaimOutcome, ReasonCode, str | None]:
    """Accept an LLM decision or escalate it. Returns (outcome, reason, fallback_reason).

    Only auto-decisions (APPROVED / REJECTED) need confidence to clear the threshold. A HUMAN_REVIEW answer is
    always accepted, because escalating is the safe direction. Shared by the live workflow and the offline
    threshold experiments, so both apply exactly the same rule.
    """
    if decision == ClaimOutcome.HUMAN_REVIEW or confidence >= threshold:
        return decision, reason, None
    return ClaimOutcome.HUMAN_REVIEW, ReasonCode.LLM_LOW_CONFIDENCE, "low_confidence"


def _fallback_reason(failure: str | None) -> str:
    if failure == MALFORMED_OUTPUT:
        return "invalid_output"
    if failure == INVALID_CITATION:
        return "invalid_citation"
    return "provider_error"  # transient errors that exhausted retries, or permanent provider errors


@dataclass
class Interpretation:
    decision: Decision
    source: str  # SOURCE_LLM or SOURCE_FALLBACK
    fallback_reason: str | None
    retrieval: RetrievalResult | None
    llm: StructuredResult | None
    threshold: float
    provider_name: str
    keyword_baseline_outcome: str

    def to_details(self) -> dict[str, Any]:
        """The Phase 2 part of WorkflowRun.details (see docs/architecture.md)."""
        llm: dict[str, Any] | None = None
        if self.llm is not None:
            accepted = self.llm.decision
            llm = {
                "provider": self.provider_name,
                "model": self.llm.model,
                "calls": self.llm.calls,
                "retries": self.llm.retries,
                "latency_ms": self.llm.latency_ms,
                "input_tokens": self.llm.input_tokens,
                "output_tokens": self.llm.output_tokens,
                "valid": accepted is not None,
                "failure": self.llm.failure,
                # Raw answer, before the confidence threshold: lets thresholds be re-evaluated offline.
                "raw_decision": accepted.decision.value if accepted else None,
                "raw_reason_code": accepted.reason_code.value if accepted else None,
                "confidence": accepted.confidence if accepted else None,
                "cited_policy_ids": accepted.cited_policy_ids if accepted else [],
                "reasoning_summary": accepted.reasoning_summary if accepted else None,
                "unsupported_citations": [c for a in self.llm.attempts for c in a.unsupported_citations],
                "attempts": [
                    {
                        "outcome": a.outcome,
                        "latency_ms": a.latency_ms,
                        "input_tokens": a.input_tokens,
                        "output_tokens": a.output_tokens,
                        "cited": a.cited_count,
                        "unsupported": len(a.unsupported_citations),
                    }
                    for a in self.llm.attempts
                ],
            }
        return {
            "llm_used": True,
            "decision_source": self.source,
            "fallback_reason": self.fallback_reason,
            "confidence_threshold": self.threshold,
            "keyword_baseline_outcome": self.keyword_baseline_outcome,
            "retrieval": self.retrieval.to_log() if self.retrieval else None,
            "llm": llm,
        }


class NotesInterpreter:
    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        *,
        threshold: float,
        timeout: float,
        max_retries: int,
        max_output_tokens: int,
        backoff_seconds: float = 1.0,
        policies: PolicyData | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.threshold = threshold
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.backoff_seconds = backoff_seconds
        self.policies = policies or load_policies()
        self.sleep = sleep

    def interpret(self, claim: Claim) -> Interpretation:
        keyword_outcome = keyword_notes_decision(claim).outcome.value
        procedure = self.policies.procedures.get((claim.procedure_code or "").strip())
        notes = claim.clinical_notes or ""

        try:
            retrieval = self.retriever.retrieve(
                RetrievalQuery(notes, claim.procedure_code, procedure.description if procedure else None)
            )
        except Exception:  # noqa: BLE001 - vector store trouble must degrade to human review, not crash
            logger.exception("Policy retrieval failed for claim %s", claim.id)
            return self._fallback("retrieval_error", None, None, keyword_outcome)

        context = retrieval.context
        if not context:
            return self._fallback("no_policy_context", retrieval, None, keyword_outcome)

        prompt = build_user_prompt(
            ClaimView(claim.procedure_code, procedure.description if procedure else None, claim.diagnosis_code, claim.claim_amount, notes),
            [PolicyExcerpt(p.policy_id, p.title, p.text) for p in context],
        )  # fmt: skip
        result = run_structured(
            self.provider,
            SYSTEM_PROMPT,
            prompt,
            [p.policy_id for p in context],
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_output_tokens=self.max_output_tokens,
            backoff_seconds=self.backoff_seconds,
            sleep=self.sleep,
        )
        if result.decision is None:
            return self._fallback(_fallback_reason(result.failure), retrieval, result, keyword_outcome)

        answer = result.decision
        outcome, reason, fallback_reason = apply_confidence_policy(
            answer.decision, answer.confidence, answer.reason_code, self.threshold
        )
        cited = ", ".join(answer.cited_policy_ids)
        explanation = f"{answer.reasoning_summary} (LLM confidence {answer.confidence:.2f}; cited {cited})"
        if fallback_reason:
            explanation = f"LLM suggested {answer.decision.value} but confidence {answer.confidence:.2f} is below threshold {self.threshold:.2f}. {explanation}"
        decision = Decision(outcome, reason, "llm_notes_interpretation", explanation, confidence=answer.confidence)
        source = SOURCE_FALLBACK if fallback_reason else SOURCE_LLM
        return Interpretation(decision, source, fallback_reason, retrieval, result, self.threshold, self.provider.name, keyword_outcome)

    def _fallback(
        self, reason: str, retrieval: RetrievalResult | None, result: StructuredResult | None, keyword_outcome: str
    ) -> Interpretation:
        decision = Decision(
            ClaimOutcome.HUMAN_REVIEW,
            ReasonCode.LLM_PROCESSING_FALLBACK,
            "llm_fallback",
            f"Automated interpretation was not possible ({reason}); routed to a human reviewer.",
            confidence=0.0,
        )
        return Interpretation(decision, SOURCE_FALLBACK, reason, retrieval, result, self.threshold, self.provider.name, keyword_outcome)


def build_interpreter(
    settings: Settings | None = None, provider: LLMProvider | None = None, retriever: Retriever | None = None
) -> NotesInterpreter | None:
    """Assemble the Phase 2 interpreter from configuration, or None when the LLM is disabled."""
    settings = settings or get_settings()
    provider = provider or build_provider(settings)
    if provider is None:
        return None
    return NotesInterpreter(
        retriever or build_retriever(settings),
        provider,
        threshold=settings.llm_confidence_threshold,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_output_tokens=settings.llm_max_output_tokens,
        backoff_seconds=settings.llm_backoff_seconds,
    )


@lru_cache
def get_default_interpreter() -> NotesInterpreter | None:
    """Process-wide interpreter for the API (None unless LLM_PROVIDER is configured)."""
    return build_interpreter()
