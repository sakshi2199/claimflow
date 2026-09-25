"""The structured answer the LLM must return."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import ClaimOutcome, ReasonCode

# Reason codes the LLM may use, and which decision each one belongs to.
REASONS_BY_DECISION: dict[ClaimOutcome, frozenset[ReasonCode]] = {
    ClaimOutcome.APPROVED: frozenset({ReasonCode.VALID_STANDARD_CLAIM}),
    ClaimOutcome.REJECTED: frozenset({ReasonCode.DIAGNOSIS_PROCEDURE_MISMATCH}),
    ClaimOutcome.HUMAN_REVIEW: frozenset(
        {ReasonCode.AMBIGUOUS_CLINICAL_NOTES, ReasonCode.INSUFFICIENT_SUPPORTING_INFO}
    ),
}


class LLMDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: ClaimOutcome
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: ReasonCode
    # A short summary of the justification. The model's reasoning process is neither requested nor stored.
    reasoning_summary: str = Field(min_length=1, max_length=600)
    cited_policy_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reason_matches_decision(self) -> LLMDecision:
        if self.reason_code not in REASONS_BY_DECISION[self.decision]:
            allowed = sorted(r.value for r in REASONS_BY_DECISION[self.decision])
            raise ValueError(f"reason_code {self.reason_code.value} is not valid for {self.decision.value}; use {allowed}")
        return self
