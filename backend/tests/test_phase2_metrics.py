"""Phase 2 metrics, threshold experiments and the Phase 1 vs Phase 2 comparison, on hand-built records."""

import copy

import pytest

from app.evaluation.comparison import ComparisonError, build_comparison, false_rejection_count, format_comparison
from app.evaluation.metrics import EvalRecord, compute_metrics
from app.evaluation.phase2_metrics import compute_phase2_metrics
from app.evaluation.report_phase2 import format_phase2_section
from app.evaluation.thresholds import format_threshold_table, outcome_at_threshold, threshold_table


def det(n: str, category: str, expected: str, actual: str) -> EvalRecord:
    details = {"llm_used": False, "decision_source": "deterministic", "states": ["RECEIVED"]}
    return EvalRecord(n, category, expected, actual, "R", "R", True, 1.0, details)


def llm(
    n, category, expected, actual, *, raw, conf, outcomes, keyword, retrieved=("A", "B", "X"), source="llm", fallback=None,
    tokens=(100, 20), latency=100.0, valid=True, failure=None,
) -> EvalRecord:  # fmt: skip
    attempts = [
        {"outcome": o, "latency_ms": latency / len(outcomes), "input_tokens": 0, "output_tokens": 0, "cited": 1 if o in ("ok", "invalid_citation") else 0, "unsupported": 1 if o == "invalid_citation" else 0}
        for o in outcomes
    ]  # fmt: skip
    details = {
        "llm_used": True, "decision_source": source, "fallback_reason": fallback, "keyword_baseline_outcome": keyword,
        "retrieval": {"policy_ids": list(retrieved), "latency_ms": 10.0},
        "llm": {
            "calls": len(outcomes), "retries": len(outcomes) - 1, "latency_ms": latency, "input_tokens": tokens[0],
            "output_tokens": tokens[1], "valid": valid, "failure": failure, "raw_decision": raw if valid else None,
            "raw_reason_code": ("VALID_STANDARD_CLAIM" if raw == "APPROVED" else "AMBIGUOUS_CLINICAL_NOTES") if valid else None,
            "confidence": conf if valid else None, "attempts": attempts,
        },
    }  # fmt: skip
    return EvalRecord(n, category, expected, actual, "R", "R", True, latency + 10, details)


A, R, H = "APPROVED", "REJECTED", "HUMAN_REVIEW"
RECORDS = [
    det("D1", "duplicate_claim", R, R),
    det("D2", "high_cost", H, H),
    llm("L1", "negated_ambiguity", A, A, raw=A, conf=0.90, outcomes=["ok"], keyword=H, tokens=(100, 20), latency=100),
    llm("L2", "ambiguous_notes_subtle", H, H, raw=H, conf=0.70, outcomes=["malformed_output", "ok"], keyword=A, tokens=(200, 40), latency=200),
    llm("L3", "negated_ambiguity", A, H, raw=A, conf=0.65, outcomes=["ok"], keyword=H, tokens=(100, 20), latency=100, source="human_review_fallback", fallback="low_confidence"),
    llm("L4", "valid_routine", A, H, raw=None, conf=None, outcomes=["transient_error"] * 3, keyword=A, tokens=(0, 0), latency=300, source="human_review_fallback", fallback="provider_error", valid=False, failure="transient_error"),
    llm("L5", "ambiguous_notes_explicit", H, H, raw=H, conf=0.95, outcomes=["invalid_citation", "ok"], keyword=H, tokens=(200, 40), latency=200),
    llm("L6", "ambiguous_notes_subtle", H, A, raw=A, conf=0.95, outcomes=["ok"], keyword=A, tokens=(100, 20), latency=100),
]  # fmt: skip
# L4's relevant policies are ranked lower (rank 2 and 3) to give a non-trivial reciprocal rank
RECORDS[5].details["retrieval"]["policy_ids"] = ["X", "A", "B"]
LABELS = {r.claim_number: ["A", "B"] for r in RECORDS if r.claim_number.startswith("L")}


def test_llm_invocation_and_sources() -> None:
    p = compute_phase2_metrics(RECORDS, LABELS)
    inv = p["llm_invocation"]
    assert (inv["claims_total"], inv["claims_with_llm"], inv["deterministic_decisions"]) == (8, 6, 2)
    assert inv["invocation_rate"] == 0.75
    assert inv["decision_sources"] == {"deterministic": 2, "human_review_fallback": 2, "llm": 4}
    assert inv["fallback_reasons"] == {"low_confidence": 1, "provider_error": 1}


def test_call_and_retry_statistics() -> None:
    calls = compute_phase2_metrics(RECORDS, LABELS)["llm_calls"]
    assert calls["total_calls"] == 10 and calls["avg_calls_per_claim"] == pytest.approx(10 / 8)
    assert calls["avg_calls_per_llm_claim"] == pytest.approx(10 / 6)
    assert calls["retries_total"] == 4 and calls["claims_with_retry"] == 3 and calls["retry_rate"] == 0.5
    assert calls["attempt_outcomes"] == {"invalid_citation": 1, "malformed_output": 1, "ok": 5, "transient_error": 3}


def test_reliability_and_grounding_rates() -> None:
    p = compute_phase2_metrics(RECORDS, LABELS)
    assert p["reliability"]["llm_failure_rate"] == pytest.approx(1 / 6)
    assert p["reliability"]["provider_error_rate"] == pytest.approx(1 / 6)
    assert p["reliability"]["structured_output_failure_rate_per_call"] == pytest.approx(1 / 10)
    assert p["reliability"]["structured_output_failure_rate_per_claim"] == 0.0
    # 6 parseable responses (5 ok + 1 invalid citation), one of which cited an unsupported id
    assert p["grounding"]["unsupported_citation_rate_per_response"] == pytest.approx(1 / 6)
    assert p["grounding"]["unsupported_citation_rate_per_cited_id"] == pytest.approx(1 / 6)
    assert p["grounding"]["claims_with_unsupported_citation_after_retries"] == 0


def test_latency_and_token_metrics() -> None:
    p = compute_phase2_metrics(RECORDS, LABELS, input_price=3.0, output_price=15.0)
    lat = p["latency_ms"]
    assert lat["avg_retrieval"] == 10.0
    assert lat["avg_llm_per_llm_claim"] == pytest.approx(1000 / 6)
    assert lat["median_llm_per_llm_claim"] == 150.0 and lat["p95_llm_per_llm_claim"] == 300.0
    assert lat["avg_total_deterministic_claims"] == 1.0
    tokens = p["tokens"]
    assert (tokens["input_tokens"], tokens["output_tokens"], tokens["total_tokens"]) == (700, 140, 840)
    assert tokens["avg_tokens_per_llm_claim"] == 140
    assert tokens["estimated_cost_usd"] == pytest.approx((700 * 3 + 140 * 15) / 1e6)


def test_cost_is_not_estimated_without_configured_prices() -> None:
    tokens = compute_phase2_metrics(RECORDS, LABELS)["tokens"]
    assert tokens["estimated_cost_usd"] is None and "LLM_INPUT_PRICE_PER_MTOK" in tokens["cost_note"]
    assert compute_phase2_metrics(RECORDS, LABELS, input_price=3.0)["tokens"]["estimated_cost_usd"] is None  # needs both


def test_focus_categories_and_effect_versus_keyword_check() -> None:
    p = compute_phase2_metrics(RECORDS, LABELS)
    assert p["focus_categories"] == {
        "ambiguous_notes_subtle": {"correct": 1, "total": 2},
        "negated_ambiguity": {"correct": 1, "total": 2},
    }
    # L1 and L2 were wrong for the keyword check and right now; L4 was right for keywords and wrong now
    assert p["vs_phase1_keyword_check"]["fixed_by_llm_path"] == 2
    assert p["vs_phase1_keyword_check"]["broken_by_llm_path"] == 1


def test_retrieval_metrics_come_from_the_recorded_retrievals() -> None:
    retrieval = compute_phase2_metrics(RECORDS, LABELS)["retrieval"]
    assert retrieval["queries"] == 6
    assert retrieval["recall_at_k"]["1"] == pytest.approx(2.5 / 6)  # 5 queries hit rank 1 (one of two relevant)
    assert retrieval["recall_at_k"]["3"] == 1.0
    assert retrieval["mrr"] == pytest.approx(5.5 / 6)
    assert compute_phase2_metrics(RECORDS, None).get("retrieval") is None  # no labels, no retrieval metrics


def test_run_with_no_llm_claims_still_produces_a_report() -> None:
    p = compute_phase2_metrics([det("D1", "duplicate_claim", R, R)], {})
    assert p["llm_invocation"]["invocation_rate"] == 0 and p["llm_calls"]["retry_rate"] is None
    assert p["reliability"]["llm_failure_rate"] is None and p["retrieval"] is None


# ---- threshold experiments ----------------------------------------------------------------------------


def test_outcome_at_threshold_only_changes_llm_auto_decisions() -> None:
    l1, l3, l4 = RECORDS[2], RECORDS[4], RECORDS[5]
    assert outcome_at_threshold(RECORDS[0], 0.99) == (R, "R")  # deterministic: untouched
    assert outcome_at_threshold(l1, 0.9)[0] == A and outcome_at_threshold(l1, 0.91)[0] == H
    assert outcome_at_threshold(l3, 0.6)[0] == A and outcome_at_threshold(l3, 0.7)[0] == H
    assert outcome_at_threshold(l4, 0.0)[0] == H  # LLM failure stays a human handoff at any threshold


def test_threshold_table_matches_hand_computed_values() -> None:
    rows = {row["threshold"]: row for row in threshold_table(RECORDS, (0.6, 0.7, 0.96))}

    low = rows[0.6]  # L3 approved (correct); L6 wrongly approved
    assert low["accuracy"] == 0.75 and low["automation_rate"] == 0.5 and low["human_review_rate"] == 0.5
    assert low["false_approvals"] == 1 and low["false_rejections"] == 0
    assert low["focus_categories"]["negated_ambiguity"] == {"correct": 2, "total": 2}

    mid = rows[0.7]  # L3 now escalated (wrong)
    assert mid["accuracy"] == 0.625 and mid["automation_rate"] == 0.375 and mid["false_approvals"] == 1
    assert mid["automated_decision_accuracy"] == pytest.approx(2 / 3)

    strict = rows[0.96]  # nothing the LLM approves clears the bar: safest, least automated
    assert strict["false_approvals"] == 0 and strict["automation_rate"] == 0.125
    assert strict["focus_categories"]["ambiguous_notes_subtle"] == {"correct": 2, "total": 2}


def test_automation_never_increases_as_the_threshold_rises() -> None:
    rows = threshold_table(RECORDS, (0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0))
    automation = [r["automation_rate"] for r in rows]
    review = [r["human_review_rate"] for r in rows]
    approvals = [r["false_approvals"] for r in rows]
    assert automation == sorted(automation, reverse=True)
    assert review == sorted(review)
    assert approvals == sorted(approvals, reverse=True)


def test_threshold_table_formats_for_the_console() -> None:
    text = format_threshold_table(threshold_table(RECORDS, (0.6, 0.9)))
    assert "threshold" in text and "0.60" in text and "0.90" in text and "1/2" in text


# ---- comparison ---------------------------------------------------------------------------------------


def _reports():
    base_records = [
        EvalRecord(r.claim_number, r.category, r.expected_outcome, r.actual_outcome, "R", "R", True, 1.0) for r in RECORDS
    ]
    phase1 = compute_metrics(base_records)
    phase1.update(label="baseline-deterministic", engine_version="rules-v1", dataset={"name": "d.jsonl", "sha256": "abc"})
    phase2 = compute_metrics(RECORDS)
    phase2.update(label="rag-llm", engine_version="rag-llm-v1", dataset={"name": "d.jsonl", "sha256": "abc"})
    phase2["phase2"] = compute_phase2_metrics(RECORDS, LABELS)
    phase2["threshold_analysis"] = threshold_table(RECORDS, (0.6, 0.8))
    phase2["llm"] = {"provider": "fake", "model": "m", "confidence_threshold": 0.8, "max_retries": 2, "timeout_seconds": 30}
    phase2["retrieval_config"] = {"name": "filtered", "description": "d"}
    return phase1, phase2


def test_comparison_lists_metrics_side_by_side() -> None:
    phase1, phase2 = _reports()
    comparison = build_comparison(phase1, phase2)
    rows = {r["metric"]: r for r in comparison["rows"]}
    assert rows["Retrieval MRR"]["phase1"] == "N/A" and rows["Retrieval MRR"]["phase2"] == "0.917"
    assert rows["LLM invocation rate"]["phase1"] == "0.0%" and rows["LLM invocation rate"]["phase2"] == "75.0%"
    assert rows["Ambiguous subtle"]["phase2"] == "1/2" and rows["Negated ambiguity"]["phase2"] == "1/2"
    assert rows["False approvals"]["phase2"] == "1" and rows["Total tokens"]["phase2"] == "840"
    assert {c["category"] for c in comparison["by_category"]} == {r.category for r in RECORDS}

    text = format_comparison(comparison)
    assert "PHASE 1 (deterministic baseline) vs PHASE 2 (RAG + LLM)" in text and "Recall@5" in text


def test_comparison_refuses_different_datasets_or_sizes() -> None:
    phase1, phase2 = _reports()
    other = copy.deepcopy(phase1)
    other["dataset"]["sha256"] = "different"
    with pytest.raises(ComparisonError, match="Datasets differ"):
        build_comparison(other, phase2)
    smaller = copy.deepcopy(phase1)
    smaller["summary"]["total_claims"] = 5
    with pytest.raises(ComparisonError, match="different number"):
        build_comparison(smaller, phase2)


def test_false_rejections_are_read_from_the_confusion_matrix() -> None:
    records = [det("a", "c", A, R), det("b", "c", H, R), det("c", "c", R, R), det("d", "c", A, A)]
    assert false_rejection_count(compute_metrics(records)) == 2


def test_phase2_console_section_reports_the_key_numbers() -> None:
    _phase1, phase2 = _reports()
    text = format_phase2_section(phase2)
    assert "LLM invocation rate: 75.0% (6/8 claims" in text
    assert "Retry rate (LLM claims): 50.0%" in text and "Unsupported citation rate" in text
    assert "cost not estimated" in text and "ambiguous_notes_subtle: 1/2 correct" in text
    assert "Confidence threshold experiment" in text
