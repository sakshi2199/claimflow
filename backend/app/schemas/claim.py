from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.models.enums import ClaimOutcome, ProcessingStatus, ReasonCode


class ClaimCreate(BaseModel):
    """Intake is intentionally lenient: malformed or incomplete claims are accepted and then rejected
    (with a reason code) by the workflow, instead of being bounced at the API boundary."""

    claim_number: str = Field(min_length=1, max_length=64)
    patient_id: str = Field(min_length=1, max_length=64)
    provider_id: str = Field(min_length=1, max_length=64)
    procedure_code: str | None = Field(default=None, max_length=32)
    diagnosis_code: str | None = Field(default=None, max_length=32)
    claim_amount: Decimal = Field(max_digits=12, decimal_places=2)
    clinical_notes: str | None = Field(default=None, max_length=20000)
    submission_date: date
    expected_outcome: ClaimOutcome | None = None


class WorkflowRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    claim_id: int
    started_at: datetime
    completed_at: datetime | None
    current_state: ProcessingStatus
    final_decision: ClaimOutcome | None
    decision_reason: ReasonCode | None
    decision_explanation: str | None
    rule_triggered: str | None
    latency_ms: float | None
    retry_count: int
    error_message: str | None
    engine_version: str | None
    details: dict | None
    created_at: datetime


class ClaimRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    claim_number: str
    patient_id: str
    provider_id: str
    procedure_code: str | None
    diagnosis_code: str | None
    claim_amount: Decimal
    clinical_notes: str | None
    submission_date: date
    expected_outcome: ClaimOutcome | None
    actual_outcome: ClaimOutcome | None
    confidence_score: float | None
    processing_status: ProcessingStatus
    evaluation_run_id: int | None
    benchmark_category: str | None
    created_at: datetime
    updated_at: datetime

    @field_serializer("claim_amount")
    def _two_decimal_places(self, amount: Decimal) -> str:
        # Same representation whether the value came from the request or from the database.
        return f"{amount:.2f}"


class ClaimDetail(ClaimRead):
    workflow_runs: list[WorkflowRunRead]


class ProcessResponse(BaseModel):
    claim: ClaimRead
    workflow_run: WorkflowRunRead
