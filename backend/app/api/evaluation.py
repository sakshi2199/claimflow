from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.evaluation.evaluator import DatasetError, run_evaluation
from app.models.evaluation_run import EvaluationRun
from app.schemas.evaluation import EvaluationRunRead, EvaluationRunRequest

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationRunRead, status_code=status.HTTP_201_CREATED)
def run(payload: EvaluationRunRequest | None = None, db: Session = Depends(get_db)) -> EvaluationRunRead:
    """Run the benchmark synchronously (a few seconds for ~200 claims) and store the result."""
    payload = payload or EvaluationRunRequest()
    dataset_path = get_settings().synthetic_claims_dir / payload.dataset_name
    if not dataset_path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {payload.dataset_name} not found. Generate it with: python -m app.evaluation.dataset_generator",
        )
    try:
        evaluation = run_evaluation(db, dataset_path, payload.label)
    except DatasetError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return EvaluationRunRead.model_validate(evaluation)


@router.get("/latest", response_model=EvaluationRunRead)
def latest(db: Session = Depends(get_db)) -> EvaluationRunRead:
    evaluation = db.scalars(
        select(EvaluationRun).where(EvaluationRun.status == "COMPLETED").order_by(EvaluationRun.id.desc()).limit(1)
    ).first()
    if evaluation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No completed evaluation runs yet. POST /evaluation/run first.")
    return EvaluationRunRead.model_validate(evaluation)
