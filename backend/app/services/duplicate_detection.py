from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.claim import Claim


def find_duplicate(db: Session, claim: Claim) -> Claim | None:
    """Return the earliest previously received claim with the same patient, provider, procedure and date.

    "Previously received" means a lower id, so the result does not depend on the order in which claims
    are processed. Lookup is limited to the same scope: live claims only see live claims, and benchmark
    claims only see claims from the same evaluation run.
    """
    if claim.evaluation_run_id is None:
        scope = Claim.evaluation_run_id.is_(None)
    else:
        scope = Claim.evaluation_run_id == claim.evaluation_run_id

    stmt = (
        select(Claim)
        .where(
            scope,
            Claim.id < claim.id,
            Claim.patient_id == claim.patient_id,
            Claim.provider_id == claim.provider_id,
            Claim.procedure_code == claim.procedure_code,
            Claim.submission_date == claim.submission_date,
        )
        .order_by(Claim.id)
        .limit(1)
    )
    return db.scalars(stmt).first()
