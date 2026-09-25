"""Loads the synthetic policy knowledge base from data/policies/documents/*.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

REQUIRED_FIELDS = (
    "policy_id", "title", "payer", "category", "procedure_codes",
    "diagnosis_prefixes", "applies_to_all", "effective_date", "text",
)  # fmt: skip


class PolicyDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class PolicyDocument:
    policy_id: str
    title: str
    payer: str
    category: str
    procedure_codes: tuple[str, ...]  # procedures this policy specifically covers (may be empty)
    diagnosis_prefixes: tuple[str, ...]  # applicable diagnosis categories, if relevant
    applies_to_all: bool  # cross-cutting policy that applies to every claim
    effective_date: str
    text: str


def load_policy_documents(directory: Path | None = None) -> list[PolicyDocument]:
    directory = directory or get_settings().policy_documents_dir
    documents = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise PolicyDocumentError(f"{path.name}: missing fields {missing}")
        if not raw["text"].strip():
            raise PolicyDocumentError(f"{path.name}: empty text")
        documents.append(
            PolicyDocument(
                policy_id=raw["policy_id"],
                title=raw["title"],
                payer=raw["payer"],
                category=raw["category"],
                procedure_codes=tuple(raw["procedure_codes"]),
                diagnosis_prefixes=tuple(raw["diagnosis_prefixes"]),
                applies_to_all=bool(raw["applies_to_all"]),
                effective_date=raw["effective_date"],
                text=raw["text"],
            )
        )
    ids = [d.policy_id for d in documents]
    if len(set(ids)) != len(ids):
        raise PolicyDocumentError("duplicate policy_id in policy documents")
    if not documents:
        raise PolicyDocumentError(f"no policy documents found in {directory}")
    return documents
