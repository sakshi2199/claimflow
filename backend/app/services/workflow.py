"""Claim processing workflow: RECEIVED -> VALIDATING -> RULE_CHECK -> COMPLETED (or FAILED).

This is the only place where decisions are written to the database.
"""

import logging
from time import perf_counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import ENGINE_VERSION
from app.db.base import utcnow
from app.models.claim import Claim
from app.models.enums import ProcessingStatus, ReasonCode
from app.models.workflow_run import WorkflowRun
from app.services.decision import Decision
from app.services.duplicate_detection import find_duplicate
from app.services.policies import load_policies
from app.services.rules import evaluate_business_rules
from app.services.validation import validate_claim

logger = logging.getLogger(__name__)

PROCESSABLE_STATES = (ProcessingStatus.RECEIVED, ProcessingStatus.FAILED)


def _decide(db: Session, claim: Claim, states: list[str]) -> Decision:
    states.append(ProcessingStatus.VALIDATING)
    decision = validate_claim(claim)
    if decision is not None:
        return decision

    states.append(ProcessingStatus.RULE_CHECK)
    duplicate = find_duplicate(db, claim)
    return evaluate_business_rules(claim, load_policies(), duplicate.id if duplicate else None)


def process_claim(db: Session, claim: Claim) -> WorkflowRun:
    """Run the workflow once for a claim and persist the outcome. Never raises for processing errors:
    an unexpected exception marks the run and claim as FAILED so callers can count failures."""
    timer = perf_counter()
    prior_attempts = db.scalar(select(func.count()).select_from(WorkflowRun).where(WorkflowRun.claim_id == claim.id))
    run = WorkflowRun(
        claim_id=claim.id,
        started_at=utcnow(),
        current_state=ProcessingStatus.RECEIVED,
        retry_count=prior_attempts or 0,
        engine_version=ENGINE_VERSION,
    )
    states: list[str] = [ProcessingStatus.RECEIVED]

    try:
        decision = _decide(db, claim, states)
    except Exception as exc:  # noqa: BLE001 - any failure must be recorded, not propagated
        logger.exception("Workflow failed for claim %s", claim.id)
        db.rollback()
        states.append(ProcessingStatus.FAILED)
        run.current_state = ProcessingStatus.FAILED
        run.decision_reason = ReasonCode.PROCESSING_ERROR
        run.error_message = f"{type(exc).__name__}: {exc}"
        claim.processing_status = ProcessingStatus.FAILED
        claim.actual_outcome = None
        claim.confidence_score = None
    else:
        states.append(ProcessingStatus.COMPLETED)
        run.current_state = ProcessingStatus.COMPLETED
        run.final_decision = decision.outcome
        run.decision_reason = decision.reason
        run.rule_triggered = decision.rule
        run.decision_explanation = decision.explanation
        claim.processing_status = ProcessingStatus.COMPLETED
        claim.actual_outcome = decision.outcome
        claim.confidence_score = decision.confidence

    run.details = {"states": [str(s) for s in states]}
    run.completed_at = utcnow()
    # Measured up to (not including) the final commit; includes the duplicate lookup and validation.
    run.latency_ms = (perf_counter() - timer) * 1000
    db.add(run)
    db.commit()
    return run
