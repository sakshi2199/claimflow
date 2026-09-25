"""CLI: run the retrieval benchmark for one or more retrieval configurations and compare them.

Usage:
    python -m app.evaluation.run_retrieval                 # all presets
    python -m app.evaluation.run_retrieval --configs baseline filtered
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal, init_db
from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import DatasetError
from app.evaluation.policy_labels import labels_path_for
from app.evaluation.report import write_report_json
from app.evaluation.retrieval_eval import run_retrieval_evaluation
from app.rag.config import PRESETS, get_preset
from app.rag.documents import load_policy_documents
from app.rag.embeddings import get_embedder
from app.rag.retrieval import Retriever
from app.rag.store import PolicyStore, get_client


def format_retrieval_table(reports: list[dict]) -> str:
    header = f"{'config':<10}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'p50 ms':>8}"
    lines = [header, "-" * len(header)]
    for report in reports:
        r = report["retrieval"]
        rec, hit = r["recall_at_k"], r["hit_rate_at_k"]
        lines.append(
            f"{report['retrieval_config']['name']:<10}"
            f"{rec['1']:>10.3f}{rec['3']:>10.3f}{rec['5']:>10.3f}"
            f"{hit['1']:>8.3f}{hit['3']:>8.3f}{hit['5']:>8.3f}{r['mrr']:>8.3f}{r['latency_ms']['median']:>8.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ClaimFlow retrieval benchmark.")
    parser.add_argument("--dataset", type=Path, default=settings.synthetic_claims_dir / DATASET_NAME)
    parser.add_argument("--configs", nargs="+", default=list(PRESETS), help=f"any of {sorted(PRESETS)}")
    parser.add_argument("--output-dir", type=Path, default=settings.evaluation_output_dir)
    args = parser.parse_args(argv)

    labels_file = labels_path_for(args.dataset)
    for required in (args.dataset, labels_file):
        if not required.exists():
            parser.error(f"file not found: {required} (run: python -m app.evaluation.dataset_generator)")

    embedder = get_embedder(settings.embedder)
    store = PolicyStore(get_client(settings.chroma_dir), embedder)
    documents = load_policy_documents(settings.policy_documents_dir)

    init_db()
    reports = []
    try:
        with SessionLocal() as db:
            for name in args.configs:
                config = get_preset(name)
                summary = store.ingest(documents, config)  # rebuild so results never use a stale index
                print(f"Ingested {summary.policies} policies -> {summary.chunks} chunks for '{name}'")
                evaluation = run_retrieval_evaluation(db, args.dataset, Retriever(store, config, documents), embedder.name)
                reports.append(evaluation.report)
                write_report_json(evaluation.report, args.output_dir, f"retrieval_{name}.json")
    except (DatasetError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(f"\nRetrieval benchmark: {reports[0]['retrieval']['queries']} queries with ground-truth policy labels, embedder {embedder.name}\n")
    print(format_retrieval_table(reports))
    (args.output_dir / "retrieval_comparison.json").write_text(
        json.dumps({r["retrieval_config"]["name"]: r["retrieval"] for r in reports}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nJSON reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
