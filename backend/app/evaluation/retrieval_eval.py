"""Retrieval-only evaluation: no LLM involved. Runs every labeled benchmark claim through a Retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.evaluation.evaluator import load_dataset
from app.evaluation.policy_labels import labels_path_for, load_policy_labels
from app.evaluation.retrieval_metrics import RetrievalRecord, compute_retrieval_metrics
from app.models.evaluation_run import EvaluationRun
from app.rag.retrieval import RetrievalQuery, Retriever
from app.services.policies import load_policies


def collect_retrieval_records(
    retriever: Retriever, rows: list[dict[str, Any]], labels: dict[str, list[str]]
) -> list[RetrievalRecord]:
    procedures = load_policies().procedures
    records = []
    for row in rows:
        relevant = labels.get(row["claim_number"])
        if relevant is None:  # claim does not need a policy lookup
            continue
        procedure = procedures.get(row["procedure_code"])
        result = retriever.retrieve(
            RetrievalQuery(row["clinical_notes"] or "", row["procedure_code"], procedure.description if procedure else None)
        )
        records.append(RetrievalRecord(row["claim_number"], row["category"], relevant, result.policy_ids, result.latency_ms))
    return records


def run_retrieval_evaluation(db: Session, dataset_path: Path, retriever: Retriever, embedder_name: str) -> EvaluationRun:
    """Evaluate one retrieval configuration and store the result as an EvaluationRun (label `retrieval-<config>`)."""
    rows, sha256 = load_dataset(dataset_path)
    labels = load_policy_labels(labels_path_for(dataset_path))
    started = utcnow()

    records = collect_retrieval_records(retriever, rows, labels)
    report = {
        "label": f"retrieval-{retriever.config.name}",
        "retrieval_config": retriever.config.to_dict(),
        "embedder": embedder_name,
        "dataset": {"name": dataset_path.name, "sha256": sha256},
        "labels_file": labels_path_for(dataset_path).name,
        "retrieval": compute_retrieval_metrics(records),
    }
    evaluation = EvaluationRun(
        label=report["label"],
        engine_version=f"retrieval:{retriever.config.name}"[:32],
        dataset_name=dataset_path.name,
        dataset_sha256=sha256,
        total_claims=len(records),
        status="COMPLETED",
        started_at=started,
        completed_at=utcnow(),
        report=report,
    )
    db.add(evaluation)
    db.commit()
    return evaluation
