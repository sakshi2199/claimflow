"""Claim processing workflow.

    RECEIVED -> VALIDATING -> RULE_CHECK -> COMPLETED                     (Phase 1, and every Phase 2 deterministic decision)
                                    \\-> INTERPRETING -> COMPLETED         (Phase 2: notes need RAG + LLM interpretation)
    any state -> FAILED on an unexpected exception

Deterministic code decides everything it can. The interpreter is consulted only when every deterministic rule
has passed and the meaning of the clinical notes is the only thing left to judge. This is the only place where
decisions are written to the database.
"""

import logging
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import ENGINE_VERSION, ENGINE_VERSION_RAG_LLM
from app.db.base import utcnow
from app.models.claim import Claim
from app.models.enums import ProcessingStatus, ReasonCode
from app.models.llm_call import LLMCall
from app.models.retrieval_log import RetrievalLog
from app.models.workflow_run import WorkflowRun
from app.services.decision import Decision
from app.services.duplicate_detection import find_duplicate
from app.services.interpretation import SOURCE_DETERMINISTIC, Interpretation, NotesInterpreter
from app.services.policies import load_policies
from app.services.rules import evaluate_business_rules, evaluate_deterministic_rules
from app.services.validation import validate_claim

logger = logging.getLogger(__name__)

PROCESSABLE_STATES = (ProcessingStatus.RECEIVED, ProcessingStatus.FAILED)


@dataclass
class _Outcome:
    decision: Decision
    interpretation: Interpretation | None  # None whenever the LLM was not consulted


def _decide(db: Session, claim: Claim, states: list[str], interpreter: NotesInterpreter | None) -> _Outcome:
    states.append(ProcessingStatus.VALIDATING)
    decision = validate_claim(claim)
    if decision is not None:
        return _Outcome(decision, None)

    states.append(ProcessingStatus.RULE_CHECK)
    duplicate = find_duplicate(db, claim)
    duplicate_id = duplicate.id if duplicate else None
    policies = load_policies()

    if interpreter is None:  # Phase 1 behaviour
        return _Outcome(evaluate_business_rules(claim, policies, duplicate_id), None)

    decision = evaluate_deterministic_rules(claim, policies, duplicate_id)
    if decision is not None:
        return _Outcome(decision, None)

    states.append(ProcessingStatus.INTERPRETING)
    interpretation = interpreter.interpret(claim)
    return _Outcome(interpretation.decision, interpretation)


def _persist_interpretation_logs(db: Session, run: WorkflowRun, interpretation: Interpretation) -> None:
    if interpretation.retrieval is not None:
        log = interpretation.retrieval.to_log()
        db.add(
            RetrievalLog(
                workflow_run_id=run.id,
                config_name=log["config"],
                query=log["query"],
                chunk_ids=log["chunk_ids"],
                chunk_scores=log["chunk_scores"],
                policy_ids=log["policy_ids"],
                policy_scores=log["policy_scores"],
                context_policy_ids=log["context_policy_ids"],
                latency_ms=log["latency_ms"],
            )
        )
    if interpretation.llm is not None:
        for attempt in interpretation.llm.attempts:
            db.add(
                LLMCall(
                    workflow_run_id=run.id,
                    attempt_number=attempt.number,
                    provider=interpretation.provider_name,
                    model=interpretation.llm.model,
                    outcome=attempt.outcome,
                    latency_ms=attempt.latency_ms,
                    input_tokens=attempt.input_tokens,
                    output_tokens=attempt.output_tokens,
                    error_message=attempt.error,
                    unsupported_citations=attempt.unsupported_citations,
                )
            )


def process_claim(db: Session, claim: Claim, interpreter: NotesInterpreter | None = None) -> WorkflowRun:
    """Run the workflow once for a claim and persist the outcome. Never raises for processing errors:
    an unexpected exception marks the run and claim as FAILED so callers can count failures."""
    timer = perf_counter()
    prior_attempts = db.scalar(select(func.count()).select_from(WorkflowRun).where(WorkflowRun.claim_id == claim.id))
    run = WorkflowRun(
        claim_id=claim.id,
        started_at=utcnow(),
        current_state=ProcessingStatus.RECEIVED,
        retry_count=prior_attempts or 0,
        engine_version=ENGINE_VERSION_RAG_LLM if interpreter else ENGINE_VERSION,
    )
    states: list[str] = [ProcessingStatus.RECEIVED]
    interpretation: Interpretation | None = None
    details: dict = {"llm_used": False, "decision_source": SOURCE_DETERMINISTIC}

    try:
        outcome = _decide(db, claim, states, interpreter)
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
        decision, interpretation = outcome.decision, outcome.interpretation
        states.append(ProcessingStatus.COMPLETED)
        run.current_state = ProcessingStatus.COMPLETED
        run.final_decision = decision.outcome
        run.decision_reason = decision.reason
        run.rule_triggered = decision.rule
        run.decision_explanation = decision.explanation
        claim.processing_status = ProcessingStatus.COMPLETED
        claim.actual_outcome = decision.outcome
        claim.confidence_score = decision.confidence
        if interpretation is not None:
            details.update(interpretation.to_details())

    details["states"] = [str(s) for s in states]
    run.details = details
    run.completed_at = utcnow()
    # Measured up to (not including) the final commit. In Phase 2 this includes retrieval and LLM time.
    run.latency_ms = (perf_counter() - timer) * 1000
    db.add(run)
    db.flush()  # assigns run.id for the log rows below
    if interpretation is not None:
        _persist_interpretation_logs(db, run, interpretation)
    db.commit()
    return run
