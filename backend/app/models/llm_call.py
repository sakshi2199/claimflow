from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class LLMCall(Base):
    """One request to the LLM provider (a workflow run can make several because of retries)."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    attempt_number: Mapped[int]
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(128))
    # ok | malformed_output | invalid_citation | transient_error | provider_error
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    latency_ms: Mapped[float]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    error_message: Mapped[str | None] = mapped_column(Text)
    unsupported_citations: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
