"""Retrieval service: builds the query, searches the vector store, groups chunks into ranked policies.

Everything that decides *what is retrieved* lives here (not in API routes or the workflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.core.config import Settings
from app.rag.config import RetrievalConfig, get_preset
from app.rag.documents import PolicyDocument, load_policy_documents
from app.rag.embeddings import Embedder, get_embedder
from app.rag.store import PolicyStore, get_client


@dataclass(frozen=True)
class RetrievalQuery:
    notes: str
    procedure_code: str | None = None
    procedure_description: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int  # 1-based rank among chunks
    chunk_id: str
    policy_id: str
    score: float
    text: str


@dataclass(frozen=True)
class RetrievedPolicy:
    rank: int  # 1-based rank among policies (a policy is ranked by its best-scoring chunk)
    policy_id: str
    title: str
    score: float
    chunk_id: str  # the best chunk
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    config_name: str
    query: str
    chunks: list[RetrievedChunk]
    policies: list[RetrievedPolicy]
    context_size: int
    latency_ms: float

    @property
    def policy_ids(self) -> list[str]:
        return [p.policy_id for p in self.policies]

    @property
    def context(self) -> list[RetrievedPolicy]:
        """The policies actually shown to the LLM. Citations must come from this set."""
        return self.policies[: self.context_size]

    def to_log(self) -> dict[str, Any]:
        return {
            "config": self.config_name,
            "query": self.query,
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "chunk_scores": [round(c.score, 4) for c in self.chunks],
            "policy_ids": self.policy_ids,
            "policy_scores": [round(p.score, 4) for p in self.policies],
            "context_policy_ids": [p.policy_id for p in self.context],
            "latency_ms": self.latency_ms,
        }


def build_query_text(config: RetrievalConfig, query: RetrievalQuery) -> str:
    if config.query_mode == "structured" and query.procedure_description:
        return f"{query.procedure_description}. {query.notes}"
    return query.notes


class Retriever:
    def __init__(self, store: PolicyStore, config: RetrievalConfig, documents: list[PolicyDocument]) -> None:
        self.store = store
        self.config = config
        self.documents = documents

    def ensure_ready(self) -> None:
        if not self.store.has_collection(self.config):
            self.store.ingest(self.documents, self.config)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        text = build_query_text(self.config, query)
        start = perf_counter()
        hits = self.store.query(self.config, text, query.procedure_code)
        latency_ms = (perf_counter() - start) * 1000

        chunks = [RetrievedChunk(i, h.chunk_id, h.policy_id, h.score, h.text) for i, h in enumerate(hits, start=1)]
        policies: list[RetrievedPolicy] = []
        seen: set[str] = set()
        for hit in hits:  # hits arrive best-first, so the first chunk seen for a policy is its best
            if hit.policy_id not in seen:
                seen.add(hit.policy_id)
                policies.append(RetrievedPolicy(len(policies) + 1, hit.policy_id, hit.title, hit.score, hit.chunk_id, hit.text))
        return RetrievalResult(self.config.name, text, chunks, policies, self.config.context_policies, latency_ms)


def build_retriever(
    settings: Settings, config_name: str | None = None, embedder: Embedder | None = None
) -> Retriever:
    config = get_preset(config_name or settings.retrieval_config)
    store = PolicyStore(get_client(settings.chroma_dir), embedder or get_embedder(settings.embedder))
    retriever = Retriever(store, config, load_policy_documents(settings.policy_documents_dir))
    retriever.ensure_ready()
    return retriever
