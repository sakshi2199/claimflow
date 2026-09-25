"""Retrieval configurations compared in the retrieval experiments (see docs/rag_evaluation.md).

The presets form an ablation: each one changes one thing relative to the previous, so the effect of each
change can be read from the measured metrics rather than assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    description: str
    chunk_size: int  # characters
    chunk_overlap: int  # characters
    include_title: bool  # prefix each chunk with its policy title before embedding
    query_mode: Literal["notes", "structured"]  # what the query text is built from
    metadata_filter: bool  # restrict candidates to policies for the claim's procedure + cross-cutting policies
    fetch_chunks: int = 20  # chunks requested from the vector store before grouping into policies
    context_policies: int = 5  # top policies (best chunk each) shown to the LLM

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, RetrievalConfig] = {
    "baseline": RetrievalConfig(
        name="baseline",
        description="Basic semantic search: ~1200-char chunks (one chunk per policy), query = clinical notes only",
        chunk_size=1200, chunk_overlap=0, include_title=False, query_mode="notes", metadata_filter=False,
    ),  # fmt: skip
    "chunking": RetrievalConfig(
        name="chunking",
        description="Baseline + 400-char chunks with 80-char overlap and the policy title prefixed to each chunk",
        chunk_size=400, chunk_overlap=80, include_title=True, query_mode="notes", metadata_filter=False,
    ),  # fmt: skip
    "query": RetrievalConfig(
        name="query",
        description="Chunking preset + query built from the procedure description and the clinical notes",
        chunk_size=400, chunk_overlap=80, include_title=True, query_mode="structured", metadata_filter=False,
    ),  # fmt: skip
    "filtered": RetrievalConfig(
        name="filtered",
        description="Query preset + metadata filter: policies for the claim's procedure code plus cross-cutting policies",
        chunk_size=400, chunk_overlap=80, include_title=True, query_mode="structured", metadata_filter=True,
    ),  # fmt: skip
}


def get_preset(name: str) -> RetrievalConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown retrieval config {name!r}; choose from {sorted(PRESETS)}") from None
