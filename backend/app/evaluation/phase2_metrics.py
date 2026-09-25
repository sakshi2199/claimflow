"""Phase 2 workflow metrics (LLM usage, reliability, grounding, cost, retrieval) computed from EvalRecords.

Everything is derived from the recorded WorkflowRun.details of an actual run: nothing here calls a model.
"""

from __future__ import annotations

import statistics
from typing import Any

from app.evaluation.metrics import EvalRecord, _ratio, percentile
from app.evaluation.retrieval_metrics import RetrievalRecord, compute_retrieval_metrics

FOCUS_CATEGORIES = ("ambiguous_notes_subtle", "negated_ambiguity")


def _llm_records(records: list[EvalRecord]) -> list[EvalRecord]:
    return [r for r in records if r.details and r.details.get("llm_used")]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def token_and_cost_metrics(
    llm_records: list[EvalRecord], input_price: float | None, output_price: float | None
) -> dict[str, Any]:
    llm = [r.details["llm"] for r in llm_records if r.details.get("llm")]
    input_tokens = sum(x["input_tokens"] or 0 for x in llm)
    output_tokens = sum(x["output_tokens"] or 0 for x in llm)
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "avg_tokens_per_llm_claim": (input_tokens + output_tokens) / len(llm_records) if llm_records else None,
        "estimated_cost_usd": None,
        "cost_note": "Set LLM_INPUT_PRICE_PER_MTOK and LLM_OUTPUT_PRICE_PER_MTOK to estimate cost; none is assumed.",
    }
    if input_price is not None and output_price is not None:
        result["estimated_cost_usd"] = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        result["cost_note"] = "Estimated from the configured per-million-token prices and the provider-reported usage."
    return result


def compute_phase2_metrics(
    records: list[EvalRecord],
    labels: dict[str, list[str]] | None = None,
    input_price: float | None = None,
    output_price: float | None = None,
) -> dict[str, Any]:
    total = len(records)
    invoked = _llm_records(records)
    invoked_ids = {id(r) for r in invoked}
    llm = [r.details["llm"] for r in invoked if r.details.get("llm")]
    attempts = [a for x in llm for a in x["attempts"]]

    parseable = [a for a in attempts if a["outcome"] in ("ok", "invalid_citation")]
    invalid_citation_attempts = [a for a in attempts if a["outcome"] == "invalid_citation"]
    cited_total = sum(a["cited"] for a in parseable)
    unsupported_total = sum(a["unsupported"] for a in parseable)
    no_decision = [x for x in llm if not x["valid"]]

    retrievals = [r.details["retrieval"] for r in invoked if r.details.get("retrieval")]
    llm_latencies = [x["latency_ms"] for x in llm]
    invoked_latencies = [r.latency_ms for r in invoked]

    fixed = broken = 0  # comparison with what the Phase 1 keyword check said for the same claim
    for r in invoked:
        keyword = r.details.get("keyword_baseline_outcome")
        if keyword is None:
            continue
        fixed += keyword != r.expected_outcome and r.correct
        broken += keyword == r.expected_outcome and not r.correct

    report: dict[str, Any] = {
        "llm_invocation": {
            "claims_total": total,
            "claims_with_llm": len(invoked),
            "invocation_rate": len(invoked) / total,
            "deterministic_decisions": total - len(invoked),
            "decision_sources": _count(r.details["decision_source"] for r in records if r.details),
            "fallback_reasons": _count(r.details["fallback_reason"] for r in invoked if r.details.get("fallback_reason")),
        },
        "llm_calls": {
            "total_calls": len(attempts),
            "avg_calls_per_claim": len(attempts) / total,
            "avg_calls_per_llm_claim": len(attempts) / len(invoked) if invoked else None,
            "retries_total": sum(x["retries"] for x in llm),
            "claims_with_retry": sum(1 for x in llm if x["retries"] > 0),
            "retry_rate": _ratio(sum(1 for x in llm if x["retries"] > 0), len(invoked)),
            "attempt_outcomes": _count(a["outcome"] for a in attempts),
        },
        "reliability": {
            # A claim "failed" if the LLM path produced no accepted decision (after all retries).
            "llm_failure_rate": _ratio(len(no_decision), len(invoked)),
            "claims_without_llm_decision": len(no_decision),
            "provider_error_rate": _ratio(
                sum(1 for x in no_decision if x["failure"] in ("transient_error", "provider_error")), len(invoked)
            ),
            # Share of individual LLM calls whose output could not be parsed/validated as the required schema.
            "structured_output_failure_rate_per_call": _ratio(
                sum(1 for a in attempts if a["outcome"] == "malformed_output"), len(attempts)
            ),
            "structured_output_failure_rate_per_claim": _ratio(
                sum(1 for x in no_decision if x["failure"] == "malformed_output"), len(invoked)
            ),
        },
        "grounding": {
            # Responses that parsed but cited a policy id that was not in the retrieved context.
            "unsupported_citation_rate_per_response": _ratio(len(invalid_citation_attempts), len(parseable)),
            "unsupported_citation_rate_per_cited_id": _ratio(unsupported_total, cited_total),
            "responses_with_unsupported_citation": len(invalid_citation_attempts),
            "claims_with_unsupported_citation_after_retries": sum(1 for x in llm if x["failure"] == "invalid_citation"),
        },
        "latency_ms": {
            "avg_retrieval": _mean([q["latency_ms"] for q in retrievals]),
            "avg_llm_per_llm_claim": _mean(llm_latencies),
            "median_llm_per_llm_claim": statistics.median(llm_latencies) if llm_latencies else None,
            "p95_llm_per_llm_claim": percentile(llm_latencies, 95) if llm_latencies else None,
            "avg_total_llm_claims": _mean(invoked_latencies),
            "avg_total_deterministic_claims": _mean([r.latency_ms for r in records if id(r) not in invoked_ids]),
        },
        "tokens": token_and_cost_metrics(invoked, input_price, output_price),
        "vs_phase1_keyword_check": {
            "note": "For LLM-routed claims only: outcome of the Phase 1 keyword check vs the final Phase 2 outcome.",
            "fixed_by_llm_path": int(fixed),
            "broken_by_llm_path": int(broken),
        },
        "focus_categories": {
            category: _focus(records, category) for category in FOCUS_CATEGORIES
        },
    }

    if labels is not None:
        retrieval_records = [
            RetrievalRecord(r.claim_number, r.category, labels[r.claim_number], r.details["retrieval"]["policy_ids"], r.details["retrieval"]["latency_ms"])
            for r in invoked
            if r.claim_number in labels and r.details.get("retrieval")
        ]
        report["retrieval"] = compute_retrieval_metrics(retrieval_records) if retrieval_records else None
    return report


def _focus(records: list[EvalRecord], category: str) -> dict[str, Any]:
    group = [r for r in records if r.category == category]
    return {"correct": sum(r.correct for r in group), "total": len(group)}


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
