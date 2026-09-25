"""ChromaDB wrapper: one collection per retrieval config, so configs can be compared side by side."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.rag.chunking import chunk_policy
from app.rag.config import RetrievalConfig
from app.rag.documents import PolicyDocument
from app.rag.embeddings import Embedder


@dataclass(frozen=True)
class RawHit:
    chunk_id: str
    policy_id: str
    title: str
    text: str
    score: float  # cosine similarity (1 - cosine distance); higher is more similar


@dataclass(frozen=True)
class IngestSummary:
    config: str
    policies: int
    chunks: int
    embedder: str


def procedure_key(code: str) -> str:
    """Chroma metadata keys cannot express 'list contains', so each covered procedure becomes a boolean key."""
    return "proc_" + code.replace("-", "_")


def get_client(path: Path) -> chromadb.api.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


class PolicyStore:
    def __init__(self, client: chromadb.api.ClientAPI, embedder: Embedder) -> None:
        self.client = client
        self.embedder = embedder

    @staticmethod
    def collection_name(config: RetrievalConfig) -> str:
        return f"policies-{config.name}"

    def has_collection(self, config: RetrievalConfig) -> bool:
        return self.collection_name(config) in {c.name for c in self.client.list_collections()}

    def ingest(self, documents: list[PolicyDocument], config: RetrievalConfig) -> IngestSummary:
        """(Re)build the collection for `config`. Idempotent: any previous collection is replaced."""
        name = self.collection_name(config)
        if self.has_collection(config):
            self.client.delete_collection(name)
        collection = self.client.create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, metadata={"embedder": self.embedder.name}
        )

        ids, texts, embed_texts, metadatas = [], [], [], []
        by_id = {d.policy_id: d for d in documents}
        for document in documents:
            for chunk in chunk_policy(document, config.chunk_size, config.chunk_overlap, config.include_title):
                metadata: dict = {
                    "policy_id": document.policy_id,
                    "title": document.title,
                    "category": document.category,
                    "effective_date": document.effective_date,
                    "chunk_index": chunk.index,
                    "applies_to_all": document.applies_to_all,
                }
                metadata.update({procedure_key(code): True for code in by_id[chunk.policy_id].procedure_codes})
                ids.append(chunk.chunk_id)
                texts.append(chunk.text)
                embed_texts.append(chunk.embed_text)
                metadatas.append(metadata)

        collection.add(ids=ids, documents=texts, embeddings=self.embedder.embed(embed_texts), metadatas=metadatas)
        return IngestSummary(config.name, len(documents), len(ids), self.embedder.name)

    def query(
        self, config: RetrievalConfig, query_text: str, procedure_code: str | None = None
    ) -> list[RawHit]:
        collection = self.client.get_collection(self.collection_name(config))
        where = None
        if config.metadata_filter and procedure_code:
            where = {"$or": [{"applies_to_all": True}, {procedure_key(procedure_code): True}]}

        result = collection.query(
            query_embeddings=self.embedder.embed([query_text]),
            n_results=config.fetch_chunks,
            where=where,
        )
        hits = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            hits.append(RawHit(chunk_id, metadata["policy_id"], metadata["title"], text, 1.0 - float(distance)))
        return hits
