"""Phase 1 (deterministic baseline) vs Phase 2 (RAG + LLM) comparison on the same benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FOCUS = (("Ambiguous subtle", "ambiguous_notes_subtle"), ("Negated ambiguity", "negated_ambiguity"))


class ComparisonError(ValueError):
    pass


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def false_rejection_count(report: dict[str, Any]) -> int:
    """Claims rejected that were not expected to be rejected, from the confusion matrix."""
    matrix = report["confusion_matrix"]["matrix"]
    return sum(row["REJECTED"] for expected, row in matrix.items() if expected != "REJECTED")


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _fraction(report: dict[str, Any], category: str) -> str:
    entry = report["by_category"].get(category)
    return "N/A" if entry is None else f"{entry['correct']}/{entry['total']}"


def build_comparison(phase1: dict[str, Any], phase2: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side metrics. Refuses to compare runs made on different datasets."""
    if phase1["dataset"]["sha256"] != phase2["dataset"]["sha256"]:
        raise ComparisonError(
            f"Datasets differ (phase 1 sha256 {phase1['dataset']['sha256'][:12]}..., phase 2 {phase2['dataset']['sha256'][:12]}...). "
            "Runs are only comparable on the identical benchmark."
        )
    if phase1["summary"]["total_claims"] != phase2["summary"]["total_claims"]:
        raise ComparisonError("Runs evaluated a different number of claims; comparison would be misleading.")

    s1, s2 = phase1["summary"], phase2["summary"]
    p2 = phase2["phase2"]
    retrieval = p2.get("retrieval")
    rec = retrieval["recall_at_k"] if retrieval else {}

    rows: list[tuple[str, str, str]] = [
        ("Decision accuracy", _pct(s1["accuracy"]), _pct(s2["accuracy"])),
        ("Automation rate", _pct(s1["automation_rate"]), _pct(s2["automation_rate"])),
        ("Human-review rate", _pct(s1["human_review_rate"]), _pct(s2["human_review_rate"])),
        ("Automated-decision accuracy", _pct(s1["automated_decision_accuracy"]), _pct(s2["automated_decision_accuracy"])),
        ("False approvals", str(s1["false_approval_count"]), str(s2["false_approval_count"])),
        ("False rejections", str(false_rejection_count(phase1)), str(false_rejection_count(phase2))),
        ("Workflow failure rate", _pct(s1["workflow_failure_rate"]), _pct(s2["workflow_failure_rate"])),
    ]
    rows += [(label, _fraction(phase1, cat), _fraction(phase2, cat)) for label, cat in FOCUS]
    rows += [
        ("Latency mean", _ms(phase1["latency_ms"]["mean"]), _ms(phase2["latency_ms"]["mean"])),
        ("Latency median", _ms(phase1["latency_ms"]["median"]), _ms(phase2["latency_ms"]["median"])),
        ("Latency P95", _ms(phase1["latency_ms"]["p95"]), _ms(phase2["latency_ms"]["p95"])),
        ("Retrieval Recall@1", "N/A", _pct(rec.get("1"))),
        ("Retrieval Recall@3", "N/A", _pct(rec.get("3"))),
        ("Retrieval Recall@5", "N/A", _pct(rec.get("5"))),
        ("Retrieval MRR", "N/A", "N/A" if not retrieval else f"{retrieval['mrr']:.3f}"),
        ("LLM invocation rate", "0.0%", _pct(p2["llm_invocation"]["invocation_rate"])),
        ("Avg LLM calls per claim", "0.00", f"{p2['llm_calls']['avg_calls_per_claim']:.2f}"),
        ("Retry rate (LLM claims)", "N/A", _pct(p2["llm_calls"]["retry_rate"])),
        ("LLM failure rate (LLM claims)", "N/A", _pct(p2["reliability"]["llm_failure_rate"])),
        ("Unsupported citation rate (responses)", "N/A", _pct(p2["grounding"]["unsupported_citation_rate_per_response"])),
        ("Total tokens", "0", f"{p2['tokens']['total_tokens']:,}"),
    ]

    categories = []
    for name, entry in phase2["by_category"].items():
        before = phase1["by_category"].get(name)
        categories.append(
            {
                "category": name,
                "expected_outcome": entry["expected_outcome"],
                "phase1_correct": before["correct"] if before else None,
                "phase2_correct": entry["correct"],
                "total": entry["total"],
            }
        )
    return {
        "dataset": phase2["dataset"],
        "phase1": {"label": phase1.get("label"), "engine_version": phase1.get("engine_version")},
        "phase2": {"label": phase2.get("label"), "engine_version": phase2.get("engine_version")},
        "rows": [{"metric": m, "phase1": a, "phase2": b} for m, a, b in rows],
        "by_category": categories,
    }


def format_comparison(comparison: dict[str, Any]) -> str:
    rows = comparison["rows"]
    width = max(len(r["metric"]) for r in rows) + 2
    lines = [
        "PHASE 1 (deterministic baseline) vs PHASE 2 (RAG + LLM)",
        f"Same benchmark: {comparison['dataset']['name']} (sha256 {comparison['dataset']['sha256'][:12]}...)",
        "",
        f"{'Metric':<{width}}{'Phase 1':>14}{'Phase 2':>14}",
        "-" * (width + 28),
    ]
    lines += [f"{r['metric']:<{width}}{r['phase1']:>14}{r['phase2']:>14}" for r in rows]
    lines += ["", "Accuracy by category (correct / total)", f"  {'category':<30}{'expected':>14}{'Phase 1':>10}{'Phase 2':>10}"]
    for c in comparison["by_category"]:
        p1 = "N/A" if c["phase1_correct"] is None else f"{c['phase1_correct']}/{c['total']}"
        lines.append(f"  {c['category']:<30}{c['expected_outcome']:>14}{p1:>10}{str(c['phase2_correct']) + '/' + str(c['total']):>10}")
    return "\n".join(lines)
