"""Console section for Phase 2 runs: LLM usage, reliability, grounding, latency, tokens, threshold table."""

from __future__ import annotations

from typing import Any

from app.evaluation.report import _pct
from app.evaluation.thresholds import format_threshold_table


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def format_phase2_section(report: dict[str, Any]) -> str:
    p = report["phase2"]
    inv, calls, rel, ground, lat, tok = (
        p["llm_invocation"], p["llm_calls"], p["reliability"], p["grounding"], p["latency_ms"], p["tokens"],
    )  # fmt: skip
    llm = report["llm"]
    cfg = report["retrieval_config"]
    cost = (
        f" | estimated cost ${tok['estimated_cost_usd']:.4f}"
        if tok["estimated_cost_usd"] is not None
        else " | cost not estimated (no prices configured)"
    )

    lines = [
        "",
        "PHASE 2: RAG + LLM DETAILS",
        f"LLM: {llm['provider']} / {llm['model']} | confidence threshold {llm['confidence_threshold']:.2f} | max retries {llm['max_retries']}",
        f"Retrieval config: {cfg['name']} ({cfg['description']})",
        "",
        f"LLM invocation rate: {_pct(inv['invocation_rate'])} ({inv['claims_with_llm']}/{inv['claims_total']} claims; the rest were decided deterministically)",
        f"Decision sources: {inv['decision_sources']}",
        f"Fallback reasons: {inv['fallback_reasons'] or 'none'}",
        f"Avg LLM calls per claim: {calls['avg_calls_per_claim']:.2f} (total {calls['total_calls']}, retries {calls['retries_total']})",
        f"Retry rate (LLM claims): {_pct(calls['retry_rate'])} | attempt outcomes: {calls['attempt_outcomes']}",
        f"LLM failure rate: {_pct(rel['llm_failure_rate'])} | structured-output failure rate per call: {_pct(rel['structured_output_failure_rate_per_call'])}",
        f"Unsupported citation rate: {_pct(ground['unsupported_citation_rate_per_response'])} of responses, "
        f"{_pct(ground['unsupported_citation_rate_per_cited_id'])} of cited ids",
        f"Avg retrieval latency: {_ms(lat['avg_retrieval'])} | avg LLM latency (per LLM claim): {_ms(lat['avg_llm_per_llm_claim'])}",
        f"Tokens: {tok['input_tokens']:,} in / {tok['output_tokens']:,} out / {tok['total_tokens']:,} total{cost}",
    ]
    if p.get("retrieval"):
        r = p["retrieval"]
        lines.append(
            f"Retrieval in this run ({r['queries']} queries): Recall@1/3/5 = "
            f"{r['recall_at_k']['1']:.3f} / {r['recall_at_k']['3']:.3f} / {r['recall_at_k']['5']:.3f}, MRR = {r['mrr']:.3f}"
        )
    for category, entry in p["focus_categories"].items():
        lines.append(f"{category}: {entry['correct']}/{entry['total']} correct")
    fixed = p["vs_phase1_keyword_check"]
    lines.append(
        f"Vs Phase 1 keyword check on LLM-routed claims: fixed {fixed['fixed_by_llm_path']}, broken {fixed['broken_by_llm_path']}"
    )
    lines += [
        "",
        "Confidence threshold experiment (replayed from the recorded LLM answers)",
        format_threshold_table(report["threshold_analysis"]),
    ]
    return "\n".join(lines)
