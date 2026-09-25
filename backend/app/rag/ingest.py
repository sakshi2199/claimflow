"""CLI: build the vector store from data/policies/documents.

Usage:
    python -m app.rag.ingest                    # all retrieval presets
    python -m app.rag.ingest --config filtered  # one preset
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.rag.config import PRESETS, get_preset
from app.rag.documents import load_policy_documents
from app.rag.embeddings import get_embedder
from app.rag.store import PolicyStore, get_client


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest synthetic policy documents into ChromaDB.")
    parser.add_argument("--config", default="all", help=f"'all' or one of {sorted(PRESETS)}")
    args = parser.parse_args(argv)

    configs = list(PRESETS.values()) if args.config == "all" else [get_preset(args.config)]
    documents = load_policy_documents(settings.policy_documents_dir)
    store = PolicyStore(get_client(settings.chroma_dir), get_embedder(settings.embedder))
    for config in configs:
        summary = store.ingest(documents, config)
        print(f"{summary.config:<10} {summary.policies} policies -> {summary.chunks} chunks ({summary.embedder})")
    print(f"Vector store: {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
