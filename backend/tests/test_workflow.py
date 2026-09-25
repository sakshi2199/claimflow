import pytest
from sqlalchemy import select

from app.models.enums import ClaimOutcome, ProcessingStatus, ReasonCode
from app.models.workflow_run import WorkflowRun
from app.services import workflow
from app.services.workflow import process_claim


def test_approved_claim_persists_decision_and_run(db, saved_claim) -> None:
    claim = saved_claim()
    run = process_claim(db, claim)

    assert claim.processing_status == ProcessingStatus.COMPLETED
    assert claim.actual_outcome == ClaimOutcome.APPROVED
    assert claim.confidence_score == pytest.approx(0.9)
    assert run.final_decision == ClaimOutcome.APPROVED
    assert run.rule_triggered == "default_approve"
    assert run.retry_count == 0
    assert run.completed_at is not None and run.latency_ms >= 0
    assert run.engine_version == "rules-v1"


def test_validation_failure_skips_rule_check_state(db, saved_claim) -> None:
    run = process_claim(db, saved_claim(procedure_code=None))
    assert run.decision_reason == ReasonCode.MISSING_PROCEDURE_CODE
    assert run.details["states"] == ["RECEIVED", "VALIDATING", "COMPLETED"]


def test_rule_check_state_is_visited_for_valid_structure(db, saved_claim) -> None:
    run = process_claim(db, saved_claim(clinical_notes="N/A"))
    assert run.details["states"] == ["RECEIVED", "VALIDATING", "RULE_CHECK", "COMPLETED"]
    assert run.final_decision == ClaimOutcome.HUMAN_REVIEW


def test_unexpected_error_marks_claim_and_run_failed(db, saved_claim, monkeypatch) -> None:
    claim = saved_claim()

    def boom(*args, **kwargs):
        raise RuntimeError("rule engine exploded")

    monkeypatch.setattr(workflow, "evaluate_business_rules", boom)
    run = process_claim(db, claim)

    assert run.current_state == ProcessingStatus.FAILED
    assert run.decision_reason == ReasonCode.PROCESSING_ERROR
    assert "rule engine exploded" in run.error_message
    assert claim.processing_status == ProcessingStatus.FAILED
    assert claim.actual_outcome is None
    assert run.details["states"][-1] == "FAILED"


def test_failed_claim_can_be_reprocessed_and_retry_count_increments(db, saved_claim, monkeypatch) -> None:
    claim = saved_claim()
    with monkeypatch.context() as m:
        m.setattr(workflow, "evaluate_business_rules", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        process_claim(db, claim)

    run = process_claim(db, claim)
    assert run.current_state == ProcessingStatus.COMPLETED
    assert run.retry_count == 1
    assert claim.actual_outcome == ClaimOutcome.APPROVED
    assert len(db.scalars(select(WorkflowRun).where(WorkflowRun.claim_id == claim.id)).all()) == 2
