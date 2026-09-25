from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import DEFAULT_LABEL


class EvaluationRunRequest(BaseModel):
    label: str = Field(default=DEFAULT_LABEL, min_length=1, max_length=64)
    # A file name inside data/synthetic_claims, never a path (prevents reading arbitrary files).
    dataset_name: str = Field(default=DATASET_NAME, pattern=r"^[A-Za-z0-9_.-]+\.jsonl$", max_length=128)


class EvaluationRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    engine_version: str
    dataset_name: str
    dataset_sha256: str
    total_claims: int
    status: str
    started_at: datetime
    completed_at: datetime | None
    report: dict[str, Any] | None
