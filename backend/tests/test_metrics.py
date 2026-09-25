"""Metric calculations checked against small hand-computed examples."""

import pytest

from app.evaluation.metrics import EvalRecord, compute_metrics, percentile
from app.evaluation.report import format_report


def rec(expected: str, actual: str | None, category: str = "c", latency: float = 1.0, ok: bool = True, n: int = 0):
    return EvalRecord(
        claim_number=f"CLM-{n}", category=category, expected_outcome=expected, actual_outcome=actual,
        expected_reason="R", actual_reason="R" if actual == expected else "X",
        workflow_succeeded=ok, latency_ms=latency,
    )  # fmt: skip


# 10 records. Rows = expected, columns = predicted:
#                 A  R  H
#   APPROVED      3  0  1      (4)
#   REJECTED      1  2  0      (3)
#   HUMAN_REVIEW  0  1  2      (3)
RECORDS = (
    [rec("APPROVED", "APPROVED", "valid", n=i) for i in range(3)]
    + [rec("APPROVED", "HUMAN_REVIEW", "valid", n=3)]
    + [rec("REJECTED", "REJECTED", "dup", n=4), rec("REJECTED", "REJECTED", "dup", n=5)]
    + [rec("REJECTED", "APPROVED", "dup", n=6)]
    + [rec("HUMAN_REVIEW", "HUMAN_REVIEW", "amb", n=7), rec("HUMAN_REVIEW", "HUMAN_REVIEW", "amb", n=8)]
    + [rec("HUMAN_REVIEW", "REJECTED", "amb", n=9)]
)


def test_summary_metrics() -> None:
    s = compute_metrics(RECORDS)["summary"]
    assert s["total_claims"] == 10
    assert s["correct_decisions"] == 7
    assert s["accuracy"] == pytest.approx(0.7)
    # predicted APPROVED=4, REJECTED=3, HUMAN_REVIEW=3
    assert s["automation_rate"] == pytest.approx(0.7)
    assert s["human_review_rate"] == pytest.approx(0.3)
    assert s["automation_rate"] + s["human_review_rate"] == pytest.approx(1.0)
    # automated = 7 records (4 approved + 3 rejected), of which 3 + 2 = 5 correct
    assert s["automated_decision_accuracy"] == pytest.approx(5 / 7)
    assert s["false_approval_count"] == 1
    assert s["workflow_success_rate"] == 1.0 and s["workflow_failure_rate"] == 0.0


def test_per_class_precision_recall_f1() -> None:
    per_class = compute_metrics(RECORDS)["per_class"]

    approved = per_class["APPROVED"]  # TP=3, FP=1 (the wrongly approved reject), FN=1
    assert approved["precision"] == pytest.approx(3 / 4)
    assert approved["recall"] == pytest.approx(3 / 4)
    assert approved["f1"] == pytest.approx(0.75)

    rejected = per_class["REJECTED"]  # TP=2, FP=1, FN=1
    assert rejected["precision"] == pytest.approx(2 / 3)
    assert rejected["recall"] == pytest.approx(2 / 3)

    review = per_class["HUMAN_REVIEW"]  # TP=2, FP=1, FN=1
    assert review["support"] == 3
    assert (review["true_positives"], review["false_positives"], review["false_negatives"]) == (2, 1, 1)


def test_confusion_matrix() -> None:
    cm = compute_metrics(RECORDS)["confusion_matrix"]
    assert cm["matrix"]["APPROVED"] == {"APPROVED": 3, "REJECTED": 0, "HUMAN_REVIEW": 1, "NO_DECISION": 0}
    assert cm["matrix"]["REJECTED"] == {"APPROVED": 1, "REJECTED": 2, "HUMAN_REVIEW": 0, "NO_DECISION": 0}
    assert cm["matrix"]["HUMAN_REVIEW"] == {"APPROVED": 0, "REJECTED": 1, "HUMAN_REVIEW": 2, "NO_DECISION": 0}
    assert sum(sum(row.values()) for row in cm["matrix"].values()) == 10


def test_accuracy_by_category() -> None:
    by_category = compute_metrics(RECORDS)["by_category"]
    assert by_category["valid"] == {"expected_outcome": "APPROVED", "total": 4, "correct": 3, "accuracy": 0.75}
    assert by_category["dup"]["accuracy"] == pytest.approx(2 / 3)
    assert by_category["amb"]["correct"] == 2


def test_misclassified_list_and_reason_accuracy() -> None:
    report = compute_metrics(RECORDS)
    assert {m["claim_number"] for m in report["misclassified"]} == {"CLM-3", "CLM-6", "CLM-9"}
    assert report["summary"]["reason_code_accuracy"] == pytest.approx(0.7)


def test_undefined_precision_is_none_not_zero() -> None:
    # Nothing is ever predicted HUMAN_REVIEW and nothing is expected to be REJECTED.
    records = [rec("APPROVED", "APPROVED", n=1), rec("HUMAN_REVIEW", "APPROVED", n=2)]
    per_class = compute_metrics(records)["per_class"]
    assert per_class["HUMAN_REVIEW"]["precision"] is None
    assert per_class["HUMAN_REVIEW"]["recall"] == 0.0
    assert per_class["HUMAN_REVIEW"]["f1"] is None
    assert per_class["REJECTED"]["precision"] is None and per_class["REJECTED"]["recall"] is None


def test_workflow_failures_count_as_incorrect_and_not_automated() -> None:
    records = [rec("APPROVED", "APPROVED", n=1), rec("APPROVED", None, ok=False, n=2)]
    report = compute_metrics(records)
    s = report["summary"]
    assert s["accuracy"] == 0.5
    assert s["workflow_success_rate"] == 0.5 and s["workflow_failure_rate"] == 0.5
    assert s["automation_rate"] == 0.5
    assert report["confusion_matrix"]["matrix"]["APPROVED"]["NO_DECISION"] == 1


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 95) == 95.0
    assert percentile(values, 50) == 50.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([3.0, 1.0, 2.0], 100) == 3.0  # unsorted input is fine


def test_latency_metrics() -> None:
    records = [rec("APPROVED", "APPROVED", latency=v, n=i) for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])]
    latency = compute_metrics(records)["latency_ms"]
    assert latency["mean"] == pytest.approx(22.0)
    assert latency["median"] == 3.0
    assert latency["p95"] == 100.0  # ceil(0.95 * 5) = 5th value
    assert (latency["min"], latency["max"]) == (1.0, 100.0)


def test_empty_evaluation_is_an_error() -> None:
    with pytest.raises(ValueError):
        compute_metrics([])


def test_console_report_contains_headline_numbers() -> None:
    report = compute_metrics(RECORDS)
    report["label"] = "baseline-deterministic"
    report["dataset"] = {"name": "x.jsonl"}
    text = format_report(report)
    assert "CLAIMFLOW BASELINE EVALUATION" in text
    assert "Claims evaluated: 10" in text
    assert "Decision accuracy: 70.0%" in text
    assert "Automation rate: 70.0%" in text
    assert "Human review rate: 30.0%" in text
    assert "P95 latency:" in text
