from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class RetrievalLog(Base):
    """One policy retrieval performed during a workflow run. List columns are ordered by rank (best first)."""

    __tablename__ = "retrieval_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    config_name: Mapped[str] = mapped_column(String(32))
    query: Mapped[str] = mapped_column(Text)
    chunk_ids: Mapped[list] = mapped_column(JSON)
    chunk_scores: Mapped[list] = mapped_column(JSON)
    policy_ids: Mapped[list] = mapped_column(JSON)
    policy_scores: Mapped[list] = mapped_column(JSON)
    context_policy_ids: Mapped[list] = mapped_column(JSON)  # the policies actually shown to the LLM
    latency_ms: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
