"""Loads the fictional policy data (procedure catalog, provider watchlist) from data/policies."""

import json
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings


@dataclass(frozen=True)
class Procedure:
    code: str
    description: str
    typical_amount_min: Decimal
    typical_amount_max: Decimal
    max_auto_amount: Decimal
    allowed_diagnosis_prefixes: tuple[str, ...]
    requires_manual_review: bool


@dataclass(frozen=True)
class PolicyData:
    procedures: dict[str, Procedure]
    watchlist_providers: frozenset[str]


@lru_cache
def load_policies(policies_dir: Path | None = None) -> PolicyData:
    directory = policies_dir or get_settings().policies_dir
    catalog = json.loads((directory / "procedure_catalog.json").read_text(encoding="utf-8"))
    watchlist = json.loads((directory / "provider_watchlist.json").read_text(encoding="utf-8"))

    procedures = {}
    for raw in catalog["procedures"]:
        low, high = raw["typical_amount_range"]
        procedures[raw["code"]] = Procedure(
            code=raw["code"],
            description=raw["description"],
            typical_amount_min=Decimal(str(low)),
            typical_amount_max=Decimal(str(high)),
            max_auto_amount=Decimal(str(raw["max_auto_amount"])),
            allowed_diagnosis_prefixes=tuple(raw["allowed_diagnosis_prefixes"]),
            requires_manual_review=raw["requires_manual_review"],
        )
    return PolicyData(procedures=procedures, watchlist_providers=frozenset(watchlist["providers"]))
