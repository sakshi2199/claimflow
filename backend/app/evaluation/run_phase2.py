"""CLI: run the SAME benchmark through the Phase 2 RAG + LLM workflow and compare it with the saved Phase 1 baseline.

Requires a configured LLM (see README). Without one it exits without producing any metrics.

Usage:
    LLM_PROVIDER=anthropic LLM_MODEL=<model> ANTHROPIC_API_KEY=<key> python -m app.evaluation.run_phase2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal, init_db
from app.evaluation.comparison import ComparisonError, build_comparison, format_comparison, load_report
from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import DatasetError, run_evaluation
from app.evaluation.report import format_report, write_report_json
from app.evaluation.report_phase2 import format_phase2_section
from app.evaluation.thresholds import DEFAULT_THRESHOLDS
from app.rag.config import PRESETS
from app.rag.retrieval import build_retriever
from app.services.interpretation import build_interpreter
from app.services.llm.base import LLMConfigError
from app.services.llm.factory import build_provider

NOT_CONFIGURED = """No LLM is configured, so no Phase 2 metrics were produced (none are ever estimated or invented).
To run the real evaluation, set these environment variables and re-run:

  LLM_PROVIDER=anthropic            # or: openai
  LLM_MODEL=<model id>              # e.g. claude-opus-5
  ANTHROPIC_API_KEY=<your key>      # or OPENAI_API_KEY (optionally OPENAI_BASE_URL)

  python -m app.evaluation.run_phase2
"""


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ClaimFlow Phase 2 (RAG + LLM) evaluation.")
    parser.add_argument("--dataset", type=Path, default=settings.synthetic_claims_dir / DATASET_NAME)
    parser.add_argument("--label", default="rag-llm")
    parser.add_argument("--retrieval-config", default=settings.retrieval_config, choices=sorted(PRESETS))
    parser.add_argument("--baseline", type=Path, default=settings.evaluation_output_dir / "baseline_rules-v1.json")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--output-dir", type=Path, default=settings.evaluation_output_dir)
    args = parser.parse_args(argv)

    try:
        provider = build_provider(settings)
    except LLMConfigError as exc:
        parser.error(str(exc))
    if provider is None:
        print(NOT_CONFIGURED)
        return 2
    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset}")
    if not args.baseline.exists():
        parser.error(f"Phase 1 baseline not found: {args.baseline} (run: python -m app.evaluation.run_baseline)")

    interpreter = build_interpreter(settings, provider=provider, retriever=build_retriever(settings, args.retrieval_config))
    init_db()
    try:
        with SessionLocal() as db:
            evaluation = run_evaluation(
                db, args.dataset, args.label, interpreter, tuple(args.thresholds),
                settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok,
            )  # fmt: skip
    except DatasetError as exc:
        parser.error(str(exc))

    report = evaluation.report
    print(format_report(report))
    print(format_phase2_section(report))
    path = write_report_json(report, args.output_dir, f"phase2_{args.label}.json")
    print(f"\nPhase 2 report saved to {path} (evaluation_runs.id = {evaluation.id})")

    try:
        comparison = build_comparison(load_report(args.baseline), report)
    except ComparisonError as exc:
        print(f"\nCOMPARISON NOT PRODUCED: {exc}")
        return 1
    text = format_comparison(comparison)
    print("\n" + text)
    write_report_json(comparison, args.output_dir, "comparison_phase1_vs_phase2.json")
    (args.output_dir / "comparison_phase1_vs_phase2.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
