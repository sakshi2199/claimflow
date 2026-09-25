"""CLI: run the benchmark through the deterministic workflow, print a report, save JSON.

Usage:
    python -m app.evaluation.run_baseline [--dataset PATH] [--label NAME] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db.base import utcnow
from app.db.session import SessionLocal, init_db
from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import DEFAULT_LABEL, run_evaluation
from app.evaluation.report import format_report, write_report_json


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the ClaimFlow baseline evaluation.")
    parser.add_argument("--dataset", type=Path, default=settings.synthetic_claims_dir / DATASET_NAME)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--output-dir", type=Path, default=settings.evaluation_output_dir)
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset} (run: python -m app.evaluation.dataset_generator)")

    init_db()
    with SessionLocal() as db:
        evaluation = run_evaluation(db, args.dataset, args.label)

    print(format_report(evaluation.report))
    filename = f"{args.label}_run{evaluation.id}_{utcnow():%Y%m%dT%H%M%SZ}.json"
    path = write_report_json(evaluation.report, args.output_dir, filename)
    print(f"\nJSON report saved to {path}")
    print(f"Saved in database as evaluation_runs.id = {evaluation.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
