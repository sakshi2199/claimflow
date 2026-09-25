import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.core.config import get_settings
from app.evaluation.dataset_generator import DATASET_NAME, generate_claims, serialize
from app.evaluation.evaluator import DatasetError, load_dataset, run_evaluation
from app.evaluation.report import write_report_json
from app.models.claim import Claim
from app.models.evaluation_run import EvaluationRun

DATASET = get_settings().synthetic_claims_dir / DATASET_NAME


def test_full_benchmark_run_produces_consistent_report(db) -> None:
    evaluation = run_evaluation(db, DATASET)
    report = evaluation.report
    summary = report["summary"]

    assert evaluation.status == "COMPLETED"
    assert summary["total_claims"] == evaluation.total_claims == 200
    assert summary["workflow_failure_rate"] == 0.0
    # Rates and confusion matrix must agree with each other.
    matrix = report["confusion_matrix"]["matrix"]
    assert sum(sum(row.values()) for row in matrix.values()) == 200
    diagonal = sum(matrix[label][label] for label in matrix)
    assert summary["accuracy"] == pytest.approx(diagonal / 200)
    assert summary["automation_rate"] + summary["human_review_rate"] == pytest.approx(1.0)
    # The deterministic categories must be handled perfectly; the errors live in the adversarial ones.
    wrong_categories = {c for c, m in report["by_category"].items() if m["accuracy"] < 1.0}
    assert wrong_categories <= {"ambiguous_notes_subtle", "negated_ambiguity"}


def test_evaluation_is_repeatable_and_isolated(db) -> None:
    first = run_evaluation(db, DATASET)
    second = run_evaluation(db, DATASET)  # same claim numbers again: must not collide or see the first run

    strip_latency = lambda r: {k: v for k, v in r.items() if k != "latency_ms"}  # noqa: E731
    assert strip_latency(first.report) == strip_latency(second.report)
    assert first.id != second.id
    assert db.scalar(select(func.count()).select_from(Claim).where(Claim.evaluation_run_id == first.id)) == 200
    assert db.scalar(select(func.count()).select_from(Claim).where(Claim.evaluation_run_id == second.id)) == 200
    assert db.scalar(select(func.count()).select_from(Claim).where(Claim.evaluation_run_id.is_(None))) == 0


def test_evaluation_run_is_persisted_with_dataset_hash(db) -> None:
    evaluation = run_evaluation(db, DATASET, label="my-variant")
    stored = db.get(EvaluationRun, evaluation.id)
    assert stored.label == "my-variant"
    assert stored.report["label"] == "my-variant"
    assert stored.dataset_sha256 == load_dataset(DATASET)[1]
    assert stored.completed_at is not None


def test_load_dataset_rejects_bad_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(DatasetError, match="no claims"):
        load_dataset(empty)

    no_label = tmp_path / "nolabel.jsonl"
    row = generate_claims()[0]
    del row["expected_outcome"]
    no_label.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match="expected_outcome"):
        load_dataset(no_label)

    bad_label = tmp_path / "badlabel.jsonl"
    row["expected_outcome"] = "MAYBE"
    bad_label.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match="invalid expected_outcome"):
        load_dataset(bad_label)


def test_small_custom_dataset_with_known_answers(db, tmp_path: Path) -> None:
    claims = generate_claims()
    subset = [c for c in claims if c["category"] in {"valid_routine", "malformed_claim_number"}][:6]
    path = tmp_path / "tiny.jsonl"
    path.write_text(serialize(subset))

    report = run_evaluation(db, path).report
    assert report["summary"]["total_claims"] == 6
    assert report["summary"]["accuracy"] == 1.0


def test_write_report_json_roundtrip(tmp_path: Path, db) -> None:
    report = run_evaluation(db, DATASET).report
    path = write_report_json(report, tmp_path / "out", "r.json")
    assert json.loads(path.read_text()) == report


# ---- API ------------------------------------------------------------------------------------------


def test_latest_evaluation_404_when_none_exist(client: TestClient) -> None:
    response = client.get("/evaluation/latest")
    assert response.status_code == 404


def test_run_then_fetch_latest_evaluation(client: TestClient) -> None:
    created = client.post("/evaluation/run", json={"label": "api-run"})
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "COMPLETED"
    assert body["report"]["summary"]["total_claims"] == 200

    latest = client.get("/evaluation/latest")
    assert latest.status_code == 200
    assert latest.json()["id"] == body["id"]
    assert latest.json()["report"]["label"] == "api-run"


def test_run_evaluation_with_no_body_uses_defaults(client: TestClient) -> None:
    response = client.post("/evaluation/run")
    assert response.status_code == 201
    assert response.json()["label"] == "baseline-deterministic"


def test_run_evaluation_rejects_path_traversal_and_missing_dataset(client: TestClient) -> None:
    assert client.post("/evaluation/run", json={"dataset_name": "../../etc/passwd"}).status_code == 422
    assert client.post("/evaluation/run", json={"dataset_name": "..\\secret.jsonl"}).status_code == 422
    missing = client.post("/evaluation/run", json={"dataset_name": "does_not_exist.jsonl"})
    assert missing.status_code == 404
    assert "dataset_generator" in missing.json()["detail"]


def test_benchmark_claims_are_hidden_from_default_claim_list(client: TestClient) -> None:
    run_id = client.post("/evaluation/run").json()["id"]
    assert client.get("/claims").json() == []
    assert len(client.get("/claims", params={"evaluation_run_id": run_id, "limit": 500}).json()) == 200
