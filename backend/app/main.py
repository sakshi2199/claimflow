import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import claims, evaluation, health
from app.db.session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="ClaimFlow",
    description="Evaluation-driven healthcare claims triage (synthetic data only).",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(claims.router)
app.include_router(evaluation.router)
