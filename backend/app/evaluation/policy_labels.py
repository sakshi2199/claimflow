"""Ground-truth policy relevance for the retrieval benchmark.

Relevance is defined by the benchmark, from each claim's category and procedure code. It is never derived
from what any retriever returned, so it cannot be tuned to retrieval results.

A claim needs a policy lookup when the deterministic rules cannot decide it without interpreting its
clinical notes. Those are exactly the claims that reach the RAG + LLM step. For such a claim the relevant
policies are:
  1. the procedure-coverage policy that lists the claim's procedure code, plus
  2. the cross-cutting policy that governs how its notes should be read (by category, below).

The labels live in a sidecar file so the Phase 1 dataset (and its SHA-256) stays unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.documents import PolicyDocument

# category -> cross-cutting policy ids that are relevant for claims of that category
CROSS_CUTTING_POLICIES: dict[str, tuple[str, ...]] = {
    "valid_routine": ("POL-DOC-001",),
    "normal_valid": ("POL-DOC-001",),
    "similar_but_distinct": ("POL-DOC-001",),
    "negated_ambiguity": ("POL-AMB-002",),
    "ambiguous_notes_explicit": ("POL-AMB-001",),
    "ambiguous_notes_subtle": ("POL-AMB-001",),
}

POLICY_LOOKUP_CATEGORIES = tuple(CROSS_CUTTING_POLICIES)


def labels_path_for(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".policy_labels.jsonl")


def relevant_policy_ids(category: str, procedure_code: str | None, documents: list[PolicyDocument]) -> list[str] | None:
    """Relevant policy ids for a claim, or None if the claim does not need a policy lookup."""
    if category not in CROSS_CUTTING_POLICIES:
        return None
    coverage = [
        d.policy_id
        for d in documents
        if d.category == "procedure_coverage" and procedure_code in d.procedure_codes
    ]
    known = {d.policy_id for d in documents}
    cross_cutting = list(CROSS_CUTTING_POLICIES[category])
    missing = [p for p in cross_cutting if p not in known]
    if missing or not coverage:
        raise ValueError(f"cannot label {category}/{procedure_code}: missing policy documents {missing or coverage}")
    return sorted(coverage) + cross_cutting


def build_policy_labels(claims: list[dict[str, Any]], documents: list[PolicyDocument]) -> list[dict[str, Any]]:
    labels = []
    for claim in claims:
        relevant = relevant_policy_ids(claim["category"], claim["procedure_code"], documents)
        if relevant is not None:
            labels.append(
                {
                    "claim_number": claim["claim_number"],
                    "category": claim["category"],
                    "procedure_code": claim["procedure_code"],
                    "relevant_policy_ids": relevant,
                }
            )
    return labels


def serialize_labels(labels: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(label) + "\n" for label in labels)


def write_policy_labels(claims: list[dict[str, Any]], documents: list[PolicyDocument], dataset_path: Path) -> Path:
    path = labels_path_for(dataset_path)
    path.write_text(serialize_labels(build_policy_labels(claims, documents)), encoding="utf-8", newline="\n")
    return path


def load_policy_labels(path: Path) -> dict[str, list[str]]:
    """claim_number -> relevant policy ids."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["claim_number"]] = row["relevant_policy_ids"]
    return labels
