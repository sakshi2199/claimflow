"""Retrieval metrics over ranked policy lists. Pure functions; formulas in docs/rag_evaluation.md.

For a query with relevant set R and a ranked list of retrieved policy ids:
  Recall@K   = |R ∩ top_K| / |R|              (share of the relevant policies found in the top K)
  Hit@K      = 1 if R ∩ top_K is non-empty    (was at least one relevant policy found in the top K)
  RR         = 1 / rank of the first relevant policy, 0 if none was retrieved
Reported numbers are the mean over queries; MRR is the mean of RR.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from app.evaluation.metrics import percentile

DEFAULT_KS = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalRecord:
    claim_number: str
    category: str
    relevant: list[str]  # ground truth, from the benchmark definition
    retrieved: list[str]  # ranked policy ids, best first
    latency_ms: float


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    return len(set(relevant) & set(retrieved[:k])) / len(set(relevant))


def hit_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def reciprocal_rank(relevant: list[str], retrieved: list[str]) -> float:
    for rank, policy_id in enumerate(retrieved, start=1):
        if policy_id in relevant:
            return 1.0 / rank
    return 0.0


def _summarize(records: list[RetrievalRecord], ks: tuple[int, ...]) -> dict[str, Any]:
    n = len(records)
    return {
        "queries": n,
        "recall_at_k": {str(k): sum(recall_at_k(r.relevant, r.retrieved, k) for r in records) / n for k in ks},
        "hit_rate_at_k": {str(k): sum(hit_at_k(r.relevant, r.retrieved, k) for r in records) / n for k in ks},
        "mrr": sum(reciprocal_rank(r.relevant, r.retrieved) for r in records) / n,
    }


def compute_retrieval_metrics(records: list[RetrievalRecord], ks: tuple[int, ...] = DEFAULT_KS) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot compute retrieval metrics for zero queries.")

    report = _summarize(records, ks)
    latencies = [r.latency_ms for r in records]
    report["latency_ms"] = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p95": percentile(latencies, 95),
    }

    by_category: dict[str, list[RetrievalRecord]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    report["by_category"] = {c: _summarize(group, ks) for c, group in sorted(by_category.items())}

    # Per relevant policy: how often was it supposed to be found, and how often was it found in the top 5?
    max_k = max(ks)
    per_policy: dict[str, dict[str, int]] = {}
    for record in records:
        for policy_id in record.relevant:
            stats = per_policy.setdefault(policy_id, {"expected": 0, f"found_in_top_{max_k}": 0})
            stats["expected"] += 1
            stats[f"found_in_top_{max_k}"] += policy_id in record.retrieved[:max_k]
    report["by_relevant_policy"] = dict(sorted(per_policy.items()))
    return report
