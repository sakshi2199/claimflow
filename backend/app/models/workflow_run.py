from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow
from app.models.claim import Claim
from app.models.enums import ClaimOutcome, ProcessingStatus, ReasonCode


class WorkflowRun(Base):
    """One processing attempt of one claim. A claim can have several runs (e.g. retry after FAILED)."""

    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_state: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus, native_enum=False, length=32))
    final_decision: Mapped[ClaimOutcome | None] = mapped_column(Enum(ClaimOutcome, native_enum=False, length=32))
    decision_reason: Mapped[ReasonCode | None] = mapped_column(Enum(ReasonCode, native_enum=False, length=48))
    latency_ms: Mapped[float | None]
    retry_count: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    rule_triggered: Mapped[str | None] = mapped_column(String(64))

    # Extension points for later phases. Phase 1 only fills `engine_version`, `decision_explanation`
    # and `details["states"]`; RAG/LLM phases can add retrieved passages, model name, token usage
    # and prompts under `details` without a schema change.
    engine_version: Mapped[str | None] = mapped_column(String(32))
    decision_explanation: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    claim: Mapped[Claim] = relationship(back_populates="workflow_runs")
