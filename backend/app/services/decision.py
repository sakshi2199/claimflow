from dataclasses import dataclass

from app.models.enums import ClaimOutcome, ReasonCode


@dataclass(frozen=True)
class Decision:
    outcome: ClaimOutcome
    reason: ReasonCode
    rule: str  # identifier of the rule that fired, e.g. "duplicate_check"
    explanation: str  # human-readable sentence
    # Rule-assigned constant, NOT a calibrated probability. Deterministic rules use 1.0;
    # heuristic text rules use lower values. Later phases can replace this with model confidence.
    confidence: float = 1.0


def reject(reason: ReasonCode, rule: str, explanation: str) -> Decision:
    return Decision(ClaimOutcome.REJECTED, reason, rule, explanation, confidence=1.0)


def review(reason: ReasonCode, rule: str, explanation: str, confidence: float = 1.0) -> Decision:
    return Decision(ClaimOutcome.HUMAN_REVIEW, reason, rule, explanation, confidence=confidence)


def approve(reason: ReasonCode, rule: str, explanation: str, confidence: float = 0.9) -> Decision:
    return Decision(ClaimOutcome.APPROVED, reason, rule, explanation, confidence=confidence)
