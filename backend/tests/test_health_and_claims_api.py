from fastapi.testclient import TestClient

from tests.conftest import valid_payload


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "up"}


def test_create_claim_returns_201_and_received_status(client: TestClient) -> None:
    response = client.post("/claims", json=valid_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["id"] >= 1
    assert body["claim_number"] == "CLM-2025-000001"
    assert body["processing_status"] == "RECEIVED"
    assert body["actual_outcome"] is None
    assert body["claim_amount"] == "120.50"


def test_get_claim_includes_workflow_runs(client: TestClient) -> None:
    claim_id = client.post("/claims", json=valid_payload()).json()["id"]
    body = client.get(f"/claims/{claim_id}").json()
    assert body["id"] == claim_id
    assert body["workflow_runs"] == []


def test_get_unknown_claim_returns_404(client: TestClient) -> None:
    response = client.get("/claims/9999")
    assert response.status_code == 404
    assert "9999" in response.json()["detail"]


def test_list_claims_with_pagination_and_filters(client: TestClient) -> None:
    for i in range(3):
        client.post("/claims", json=valid_payload(claim_number=f"CLM-2025-00000{i + 1}", patient_id=f"PT-10000{i}"))
    first_id = client.get("/claims").json()[0]["id"]
    client.post(f"/claims/{first_id}/process")

    assert len(client.get("/claims").json()) == 3
    assert len(client.get("/claims", params={"limit": 2}).json()) == 2
    assert len(client.get("/claims", params={"offset": 2}).json()) == 1
    assert len(client.get("/claims", params={"processing_status": "COMPLETED"}).json()) == 1
    assert len(client.get("/claims", params={"actual_outcome": "APPROVED"}).json()) == 1


def test_list_claims_rejects_bad_query_params(client: TestClient) -> None:
    assert client.get("/claims", params={"limit": 0}).status_code == 422
    assert client.get("/claims", params={"processing_status": "NOPE"}).status_code == 422


def test_create_claim_duplicate_claim_number_returns_409(client: TestClient) -> None:
    assert client.post("/claims", json=valid_payload()).status_code == 201
    response = client.post("/claims", json=valid_payload())
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_create_claim_missing_required_field_returns_422(client: TestClient) -> None:
    payload = valid_payload()
    del payload["patient_id"]
    assert client.post("/claims", json=payload).status_code == 422


def test_create_claim_bad_types_return_422(client: TestClient) -> None:
    assert client.post("/claims", json=valid_payload(claim_amount="abc")).status_code == 422
    assert client.post("/claims", json=valid_payload(submission_date="not-a-date")).status_code == 422
    assert client.post("/claims", json=valid_payload(expected_outcome="MAYBE")).status_code == 422


def test_intake_is_lenient_and_workflow_rejects_bad_business_data(client: TestClient) -> None:
    # Missing procedure code + malformed number: accepted at intake, rejected by the workflow.
    payload = valid_payload(claim_number="BAD-NUMBER", procedure_code=None)
    claim_id = client.post("/claims", json=payload).json()["id"]
    result = client.post(f"/claims/{claim_id}/process")
    assert result.status_code == 200
    assert result.json()["claim"]["actual_outcome"] == "REJECTED"
    assert result.json()["workflow_run"]["decision_reason"] == "MALFORMED_CLAIM_NUMBER"


def test_process_claim_approves_valid_claim(client: TestClient) -> None:
    claim_id = client.post("/claims", json=valid_payload()).json()["id"]
    body = client.post(f"/claims/{claim_id}/process").json()

    assert body["claim"]["processing_status"] == "COMPLETED"
    assert body["claim"]["actual_outcome"] == "APPROVED"
    run = body["workflow_run"]
    assert run["decision_reason"] == "VALID_STANDARD_CLAIM"
    assert run["final_decision"] == "APPROVED"
    assert run["current_state"] == "COMPLETED"
    assert run["latency_ms"] >= 0
    assert run["details"]["states"] == ["RECEIVED", "VALIDATING", "RULE_CHECK", "COMPLETED"]

    detail = client.get(f"/claims/{claim_id}").json()
    assert len(detail["workflow_runs"]) == 1


def test_process_unknown_claim_returns_404(client: TestClient) -> None:
    assert client.post("/claims/9999/process").status_code == 404


def test_process_already_completed_claim_returns_409(client: TestClient) -> None:
    claim_id = client.post("/claims", json=valid_payload()).json()["id"]
    assert client.post(f"/claims/{claim_id}/process").status_code == 200
    response = client.post(f"/claims/{claim_id}/process")
    assert response.status_code == 409
    assert "COMPLETED" in response.json()["detail"]


def test_duplicate_submission_is_rejected_through_the_api(client: TestClient) -> None:
    first = client.post("/claims", json=valid_payload()).json()["id"]
    second = client.post("/claims", json=valid_payload(claim_number="CLM-2025-000002")).json()["id"]

    assert client.post(f"/claims/{first}/process").json()["claim"]["actual_outcome"] == "APPROVED"
    result = client.post(f"/claims/{second}/process").json()
    assert result["claim"]["actual_outcome"] == "REJECTED"
    assert result["workflow_run"]["decision_reason"] == "DUPLICATE_CLAIM"
