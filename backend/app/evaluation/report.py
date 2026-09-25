"""Console and JSON output for an evaluation report (the dict produced by compute_metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import LABELS, NO_DECISION


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_report(report: dict[str, Any]) -> str:
    s, lat = report["summary"], report["latency_ms"]
    title = "CLAIMFLOW BASELINE EVALUATION" if "baseline" in report.get("label", "") else "CLAIMFLOW EVALUATION"
    lines = [
        title,
        f"Run: {report.get('label', '-')} | engine {report.get('engine_version', '-')} | dataset {report['dataset']['name']}",
        "",
        f"Claims evaluated: {s['total_claims']}",
        f"Decision accuracy: {_pct(s['accuracy'])}",
        f"Automation rate: {_pct(s['automation_rate'])}",
        f"Human review rate: {_pct(s['human_review_rate'])}",
        f"Workflow success rate: {_pct(s['workflow_success_rate'])}",
        f"Workflow failure rate: {_pct(s['workflow_failure_rate'])}",
        f"Automated decision accuracy: {_pct(s['automated_decision_accuracy'])}",
        f"False approvals: {s['false_approval_count']}",
        f"Reason-code accuracy: {_pct(s['reason_code_accuracy'])}",
        "",
        f"Average latency: {lat['mean']:.2f} ms",
        f"Median latency: {lat['median']:.2f} ms",
        f"P95 latency: {lat['p95']:.2f} ms",
        "",
        "Per-class metrics",
        f"  {'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}",
    ]
    for label in LABELS:
        m = report["per_class"][label]
        lines.append(f"  {label:<14}{_pct(m['precision']):>10}{_pct(m['recall']):>9}{_pct(m['f1']):>8}{m['support']:>9}")

    cm = report["confusion_matrix"]
    lines += ["", "Confusion matrix (rows = expected, columns = predicted)"]
    lines.append("  " + " " * 14 + "".join(f"{c:>14}" for c in cm["labels"]))
    for expected in LABELS:
        row = cm["matrix"][expected]
        lines.append(f"  {expected:<14}" + "".join(f"{row[c]:>14}" for c in cm["labels"]))

    lines += ["", "Accuracy by category", f"  {'category':<32}{'expected':>14}{'correct':>10}{'accuracy':>10}"]
    for name, m in report["by_category"].items():
        lines.append(f"  {name:<32}{m['expected_outcome']:>14}{m['correct']:>5}/{m['total']:<4}{_pct(m['accuracy']):>10}")

    lines += ["", f"Misclassified claims: {len(report['misclassified'])} (full list in the JSON report)"]
    return "\n".join(lines)


def write_report_json(report: dict[str, Any], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
