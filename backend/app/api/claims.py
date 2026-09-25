from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.claim import Claim
from app.models.enums import ClaimOutcome, ProcessingStatus
from app.schemas.claim import ClaimCreate, ClaimDetail, ClaimRead, ProcessResponse, WorkflowRunRead
from app.services.interpretation import NotesInterpreter, get_default_interpreter
from app.services.llm.base import LLMConfigError
from app.services.workflow import PROCESSABLE_STATES, process_claim

router = APIRouter(prefix="/claims", tags=["claims"])


def interpreter_dependency() -> NotesInterpreter | None:
    """The Phase 2 interpreter, or None when no LLM is configured (then claims use the Phase 1 rules only)."""
    try:
        return get_default_interpreter()
    except LLMConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"LLM misconfigured: {exc}") from exc


def _get_claim_or_404(db: Session, claim_id: int) -> Claim:
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Claim {claim_id} not found")
    return claim


@router.post("", response_model=ClaimRead, status_code=status.HTTP_201_CREATED)
def create_claim(payload: ClaimCreate, db: Session = Depends(get_db)) -> ClaimRead:
    existing = db.scalar(
        select(Claim.id).where(Claim.claim_number == payload.claim_number, Claim.evaluation_run_id.is_(None))
    )
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Claim number {payload.claim_number} already exists")

    claim = Claim(**payload.model_dump())
    db.add(claim)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Claim could not be stored (conflict)") from exc
    return ClaimRead.model_validate(claim)


@router.get("", response_model=list[ClaimRead])
def list_claims(
    processing_status: ProcessingStatus | None = None,
    actual_outcome: ClaimOutcome | None = None,
    evaluation_run_id: int | None = Query(default=None, description="Show benchmark claims of one evaluation run"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ClaimRead]:
    stmt = select(Claim).order_by(Claim.id).limit(limit).offset(offset)
    # By default only live claims are listed; benchmark claims are opt-in via evaluation_run_id.
    if evaluation_run_id is None:
        stmt = stmt.where(Claim.evaluation_run_id.is_(None))
    else:
        stmt = stmt.where(Claim.evaluation_run_id == evaluation_run_id)
    if processing_status is not None:
        stmt = stmt.where(Claim.processing_status == processing_status)
    if actual_outcome is not None:
        stmt = stmt.where(Claim.actual_outcome == actual_outcome)
    return [ClaimRead.model_validate(c) for c in db.scalars(stmt)]


@router.get("/{claim_id}", response_model=ClaimDetail)
def get_claim(claim_id: int, db: Session = Depends(get_db)) -> ClaimDetail:
    return ClaimDetail.model_validate(_get_claim_or_404(db, claim_id))


@router.post("/{claim_id}/process", response_model=ProcessResponse)
def process(
    claim_id: int,
    db: Session = Depends(get_db),
    interpreter: NotesInterpreter | None = Depends(interpreter_dependency),
) -> ProcessResponse:
    claim = _get_claim_or_404(db, claim_id)
    if claim.processing_status not in PROCESSABLE_STATES:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Claim {claim_id} is already {claim.processing_status.value}; only RECEIVED or FAILED claims can be processed",
        )
    run = process_claim(db, claim, interpreter)
    return ProcessResponse(claim=ClaimRead.model_validate(claim), workflow_run=WorkflowRunRead.model_validate(run))
