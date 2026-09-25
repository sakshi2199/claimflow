"""Confidence-threshold experiments, replayed offline from the raw LLM answers stored with each claim.

No new LLM calls are needed: each claim's raw decision and confidence were recorded, and the threshold rule
(`apply_confidence_policy`, shared with the live workflow) is a pure function. Replaying at the run's own
threshold therefore reproduces the run's actual outcomes exactly (verified by a test).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.evaluation.comparison import false_rejection_count
from app.evaluation.metrics import EvalRecord, compute_metrics
from app.evaluation.phase2_metrics import FOCUS_CATEGORIES
from app.models.enums import ClaimOutcome, ReasonCode
from app.services.interpretation import apply_confidence_policy

DEFAULT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)


def outcome_at_threshold(record: EvalRecord, threshold: float) -> tuple[str | None, str | None]:
    """The (outcome, reason) this claim would have received at `threshold`."""
    llm = (record.details or {}).get("llm")
    if not llm or not llm["valid"]:
        return record.actual_outcome, record.actual_reason  # deterministic decision or fallback: threshold-independent
    outcome, reason, _ = apply_confidence_policy(
        ClaimOutcome(llm["raw_decision"]), llm["confidence"], ReasonCode(llm["raw_reason_code"]), threshold
    )
    return outcome.value, reason.value


def threshold_table(records: list[EvalRecord], thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        replayed = []
        for record in records:
            outcome, reason = outcome_at_threshold(record, threshold)
            replayed.append(replace(record, actual_outcome=outcome, actual_reason=reason))
        metrics = compute_metrics(replayed)
        summary = metrics["summary"]
        by_category = metrics["by_category"]
        rows.append(
            {
                "threshold": threshold,
                "accuracy": summary["accuracy"],
                "automation_rate": summary["automation_rate"],
                "human_review_rate": summary["human_review_rate"],
                "false_approvals": summary["false_approval_count"],
                "false_rejections": false_rejection_count(metrics),
                "automated_decision_accuracy": summary["automated_decision_accuracy"],
                "focus_categories": {
                    c: {"correct": by_category[c]["correct"], "total": by_category[c]["total"]}
                    for c in FOCUS_CATEGORIES
                    if c in by_category
                },
            }
        )
    return rows


def format_threshold_table(rows: list[dict[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    header = f"{'threshold':>9}{'accuracy':>10}{'automation':>12}{'human rev.':>12}{'false appr.':>13}{'false rej.':>12}{'auto acc.':>11}{'subtle':>8}{'negated':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        focus = row["focus_categories"]
        subtle, negated = focus.get("ambiguous_notes_subtle"), focus.get("negated_ambiguity")
        lines.append(
            f"{row['threshold']:>9.2f}{pct(row['accuracy']):>10}{pct(row['automation_rate']):>12}{pct(row['human_review_rate']):>12}"
            f"{row['false_approvals']:>13}{row['false_rejections']:>12}{pct(row['automated_decision_accuracy']):>11}"
            f"{(str(subtle['correct']) + '/' + str(subtle['total'])) if subtle else 'n/a':>8}"
            f"{(str(negated['correct']) + '/' + str(negated['total'])) if negated else 'n/a':>9}"
        )
    return "\n".join(lines)
