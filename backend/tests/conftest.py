from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables)
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.claim import Claim


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker]:
    """Fresh in-memory SQLite database per test (PostgreSQL is the production target)."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


@pytest.fixture()
def db(session_factory: sessionmaker) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture()
def client(session_factory: sessionmaker) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    # No `with` block: skips the lifespan, so the app never touches the real database.
    yield TestClient(app)
    app.dependency_overrides.clear()


def valid_payload(**overrides) -> dict:
    """A claim that passes every rule."""
    payload = {
        "claim_number": "CLM-2025-000001",
        "patient_id": "PT-100001",
        "provider_id": "PRV-1001",
        "procedure_code": "PRC-101",
        "diagnosis_code": "J06.9",
        "claim_amount": 120.50,
        "clinical_notes": "Patient presented with sore throat. Examination consistent with upper respiratory infection.",
        "submission_date": "2025-03-01",
    }
    payload.update(overrides)
    return payload


def make_claim(**overrides) -> Claim:
    """An unsaved Claim ORM object that passes every rule; override fields to break a rule."""
    fields = {
        "claim_number": "CLM-2025-000001",
        "patient_id": "PT-100001",
        "provider_id": "PRV-1001",
        "procedure_code": "PRC-101",
        "diagnosis_code": "J06.9",
        "claim_amount": Decimal("120.50"),
        "clinical_notes": "Patient presented with sore throat. Examination consistent with upper respiratory infection.",
        "submission_date": date(2025, 3, 1),
    }
    fields.update(overrides)
    return Claim(**fields)


@pytest.fixture()
def saved_claim(db: Session):
    def _save(**overrides) -> Claim:
        claim = make_claim(**overrides)
        db.add(claim)
        db.commit()
        return claim

    return _save
