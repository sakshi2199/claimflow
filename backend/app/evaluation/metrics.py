"""Evaluation metrics. Pure functions over a list of EvalRecord: no database, no clock.

Formulas are documented in docs/evaluation_metrics.md.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from app.models.enums import ClaimOutcome

LABELS = [o.value for o in ClaimOutcome]
NO_DECISION = "NO_DECISION"  # column used when the workflow failed and produced no outcome
AUTOMATED = {ClaimOutcome.APPROVED.value, ClaimOutcome.REJECTED.value}


@dataclass(frozen=True)
class EvalRecord:
    claim_number: str
    category: str
    expected_outcome: str
    actual_outcome: str | None  # None when the workflow failed
    expected_reason: str | None
    actual_reason: str | None
    workflow_succeeded: bool
    latency_ms: float

    @property
    def correct(self) -> bool:
        return self.actual_outcome == self.expected_outcome


def _ratio(numerator: int, denominator: int) -> float | None:
    """Returns None (not 0) when the denominator is zero, so "undefined" is never mistaken for "bad"."""
    return numerator / denominator if denominator else None


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value with at least `pct`% of the data at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def confusion_matrix(records: list[EvalRecord]) -> dict[str, Any]:
    columns = LABELS + [NO_DECISION]
    matrix = {expected: {column: 0 for column in columns} for expected in LABELS}
    for record in records:
        matrix[record.expected_outcome][record.actual_outcome or NO_DECISION] += 1
    return {"rows": "expected", "columns": "predicted", "labels": columns, "matrix": matrix}


def per_class_metrics(records: list[EvalRecord]) -> dict[str, dict[str, Any]]:
    result = {}
    for label in LABELS:
        tp = sum(1 for r in records if r.expected_outcome == label and r.actual_outcome == label)
        fp = sum(1 for r in records if r.expected_outcome != label and r.actual_outcome == label)
        fn = sum(1 for r in records if r.expected_outcome == label and r.actual_outcome != label)
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return result


def category_metrics(records: list[EvalRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    return {
        category: {
            "expected_outcome": group[0].expected_outcome,
            "total": len(group),
            "correct": sum(r.correct for r in group),
            "accuracy": sum(r.correct for r in group) / len(group),
        }
        for category, group in sorted(grouped.items())
    }


def latency_metrics(records: list[EvalRecord]) -> dict[str, float]:
    values = [r.latency_ms for r in records]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def compute_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot compute metrics for an empty evaluation.")

    total = len(records)
    correct = sum(r.correct for r in records)
    automated = [r for r in records if r.actual_outcome in AUTOMATED]
    approved = [r for r in records if r.actual_outcome == ClaimOutcome.APPROVED.value]
    succeeded = sum(r.workflow_succeeded for r in records)
    with_reason = [r for r in records if r.expected_reason is not None]

    return {
        "summary": {
            "total_claims": total,
            "correct_decisions": correct,
            "accuracy": correct / total,
            "automation_rate": len(automated) / total,
            "human_review_rate": sum(1 for r in records if r.actual_outcome == ClaimOutcome.HUMAN_REVIEW.value) / total,
            "workflow_success_rate": succeeded / total,
            "workflow_failure_rate": (total - succeeded) / total,
            # Of the claims the system decided on its own, how many were decided correctly?
            "automated_decision_accuracy": _ratio(sum(r.correct for r in automated), len(automated)),
            # Claims approved that should NOT have been approved (the costly error in claims operations).
            "false_approval_count": sum(1 for r in approved if not r.correct),
            "reason_code_accuracy": _ratio(sum(r.actual_reason == r.expected_reason for r in with_reason), len(with_reason)),
        },
        "latency_ms": latency_metrics(records),
        "per_class": per_class_metrics(records),
        "by_category": category_metrics(records),
        "confusion_matrix": confusion_matrix(records),
        "misclassified": [
            {
                "claim_number": r.claim_number,
                "category": r.category,
                "expected_outcome": r.expected_outcome,
                "actual_outcome": r.actual_outcome,
                "expected_reason": r.expected_reason,
                "actual_reason": r.actual_reason,
            }
            for r in records
            if not r.correct
        ],
    }
