"""Runs the benchmark through the real workflow and saves the result as an EvaluationRun."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import ENGINE_VERSION
from app.db.base import utcnow
from app.evaluation.metrics import EvalRecord, compute_metrics
from app.models.claim import Claim
from app.models.enums import ClaimOutcome, ProcessingStatus
from app.models.evaluation_run import EvaluationRun
from app.services.workflow import process_claim

REQUIRED_FIELDS = ("claim_number", "patient_id", "provider_id", "claim_amount", "submission_date", "category", "expected_outcome")
DEFAULT_LABEL = "baseline-deterministic"


class DatasetError(ValueError):
    pass


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read a JSONL benchmark. Returns the rows and the SHA-256 of the file (to pin results to a dataset)."""
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name} line {line_number}: invalid JSON ({exc.msg})") from exc
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise DatasetError(f"{path.name} line {line_number}: missing fields {missing}")
        if row["expected_outcome"] not in {o.value for o in ClaimOutcome}:
            raise DatasetError(f"{path.name} line {line_number}: invalid expected_outcome {row['expected_outcome']!r}")
        rows.append(row)
    if not rows:
        raise DatasetError(f"{path.name} contains no claims")
    return rows, hashlib.sha256(raw).hexdigest()


def _claim_from_row(row: dict[str, Any], evaluation_run_id: int) -> Claim:
    return Claim(
        claim_number=row["claim_number"],
        patient_id=row["patient_id"],
        provider_id=row["provider_id"],
        procedure_code=row.get("procedure_code"),
        diagnosis_code=row.get("diagnosis_code"),
        claim_amount=Decimal(str(row["claim_amount"])),
        clinical_notes=row.get("clinical_notes"),
        submission_date=date.fromisoformat(row["submission_date"]),
        expected_outcome=ClaimOutcome(row["expected_outcome"]),
        benchmark_category=row["category"],
        evaluation_run_id=evaluation_run_id,
    )


def run_evaluation(db: Session, dataset_path: Path, label: str = DEFAULT_LABEL) -> EvaluationRun:
    rows, sha256 = load_dataset(dataset_path)
    evaluation = EvaluationRun(
        label=label,
        engine_version=ENGINE_VERSION,
        dataset_name=dataset_path.name,
        dataset_sha256=sha256,
        total_claims=len(rows),
        status="RUNNING",
        started_at=utcnow(),
    )
    db.add(evaluation)
    db.commit()

    try:
        records = []
        for row in rows:  # dataset order = submission order, which duplicate detection relies on
            claim = _claim_from_row(row, evaluation.id)
            db.add(claim)
            db.commit()  # commit first so a workflow failure cannot roll the claim back
            run = process_claim(db, claim)
            records.append(
                EvalRecord(
                    claim_number=claim.claim_number,
                    category=row["category"],
                    expected_outcome=row["expected_outcome"],
                    actual_outcome=claim.actual_outcome.value if claim.actual_outcome else None,
                    expected_reason=row.get("expected_reason"),
                    actual_reason=run.decision_reason.value if run.decision_reason else None,
                    workflow_succeeded=run.current_state == ProcessingStatus.COMPLETED,
                    latency_ms=run.latency_ms,
                )
            )

        report = compute_metrics(records)
        report["label"] = label
        report["engine_version"] = ENGINE_VERSION
        report["dataset"] = {"name": dataset_path.name, "sha256": sha256}
    except Exception:
        db.rollback()
        evaluation.status = "FAILED"
        evaluation.completed_at = utcnow()
        db.commit()
        raise

    evaluation.report = report
    evaluation.status = "COMPLETED"
    evaluation.completed_at = utcnow()
    db.commit()
    return evaluation
