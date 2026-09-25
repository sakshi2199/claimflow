from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Enum, ForeignKey, Index, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow
from app.models.enums import ClaimOutcome, ProcessingStatus


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        # Benchmark claims are unique per evaluation run. Live claims (evaluation_run_id NULL)
        # are not covered by this constraint in SQL, so the API checks them explicitly.
        UniqueConstraint("evaluation_run_id", "claim_number", name="uq_claims_scope_number"),
        Index("ix_claims_duplicate_key", "patient_id", "provider_id", "procedure_code", "submission_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_number: Mapped[str] = mapped_column(String(64), index=True)
    patient_id: Mapped[str] = mapped_column(String(64))
    provider_id: Mapped[str] = mapped_column(String(64))
    procedure_code: Mapped[str | None] = mapped_column(String(32))
    diagnosis_code: Mapped[str | None] = mapped_column(String(32))
    claim_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    clinical_notes: Mapped[str | None] = mapped_column(Text)
    submission_date: Mapped[date] = mapped_column(Date)

    expected_outcome: Mapped[ClaimOutcome | None] = mapped_column(
        Enum(ClaimOutcome, native_enum=False, length=32)
    )
    actual_outcome: Mapped[ClaimOutcome | None] = mapped_column(
        Enum(ClaimOutcome, native_enum=False, length=32)
    )
    confidence_score: Mapped[float | None]
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, native_enum=False, length=32), default=ProcessingStatus.RECEIVED
    )

    # Set only for claims created by an evaluation run (see app/evaluation/evaluator.py).
    evaluation_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    benchmark_category: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(  # noqa: F821
        back_populates="claim", cascade="all, delete-orphan", order_by="WorkflowRun.id"
    )
