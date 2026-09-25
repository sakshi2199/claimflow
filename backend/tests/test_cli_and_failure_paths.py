from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.db.session import get_db
from app.evaluation import evaluator, run_baseline
from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import DatasetError, load_dataset, run_evaluation
from app.main import app
from app.models.evaluation_run import EvaluationRun

DATASET = get_settings().synthetic_claims_dir / DATASET_NAME


def test_baseline_cli_prints_report_and_writes_json(session_factory, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_baseline, "SessionLocal", session_factory)
    monkeypatch.setattr(run_baseline, "init_db", lambda: None)

    assert run_baseline.main(["--output-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "CLAIMFLOW BASELINE EVALUATION" in out
    assert "Claims evaluated: 200" in out
    assert "Median latency:" in out
    assert len(list(tmp_path.glob("baseline-deterministic_run*.json"))) == 1


def test_baseline_cli_fails_cleanly_when_dataset_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run_baseline.main(["--dataset", str(tmp_path / "nope.jsonl")])
    assert exit_info.value.code == 2


def test_health_returns_503_when_database_is_down() -> None:
    class BrokenSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[get_db] = lambda: BrokenSession()
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"] == "database unavailable"


def test_evaluation_marks_run_failed_when_metrics_crash(db, monkeypatch) -> None:
    def boom(_records):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(evaluator, "compute_metrics", boom)
    with pytest.raises(RuntimeError):
        run_evaluation(db, DATASET)

    stored = db.query(EvaluationRun).one()
    assert stored.status == "FAILED"
    assert stored.completed_at is not None


def test_load_dataset_reports_line_number_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(DatasetError, match="line 1|line 2"):
        load_dataset(path)


def test_api_returns_422_for_corrupt_dataset_file(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "synthetic_claims").mkdir()
    (tmp_path / "synthetic_claims" / "corrupt.jsonl").write_text("not json\n")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        response = client.post("/evaluation/run", json={"dataset_name": "corrupt.jsonl"})
    finally:
        monkeypatch.delenv("DATA_DIR")
        get_settings.cache_clear()
    assert response.status_code == 422
    assert "invalid JSON" in response.json()["detail"] or "missing fields" in response.json()["detail"]
