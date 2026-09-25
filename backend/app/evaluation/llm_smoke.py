"""CLI: make ONE real LLM call on a synthetic claim to check credentials, model name and output format.

Usage:
    LLM_PROVIDER=anthropic LLM_MODEL=<model> ANTHROPIC_API_KEY=<key> python -m app.evaluation.llm_smoke
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from app.core.config import get_settings
from app.models.claim import Claim
from app.services.interpretation import build_interpreter
from app.services.llm.base import LLMConfigError


def main() -> int:
    settings = get_settings()
    try:
        interpreter = build_interpreter(settings)
    except LLMConfigError as exc:
        print(f"LLM misconfigured: {exc}")
        return 2
    if interpreter is None:
        print("LLM_PROVIDER is 'none'. Set LLM_PROVIDER, LLM_MODEL and the API key (see README) to run this check.")
        return 2

    claim = Claim(
        claim_number="CLM-2025-999999", patient_id="PT-000001", provider_id="PRV-1001",
        procedure_code="PRC-101", diagnosis_code="J06.9", claim_amount=Decimal("120.00"),
        clinical_notes="Diagnosis of acute upper respiratory infection confirmed by exam; there is no uncertainty about the diagnosis. Office visit completed without issue.",
        submission_date=date(2025, 3, 1),
    )  # fmt: skip
    result = interpreter.interpret(claim)
    print(json.dumps(result.to_details(), indent=2, default=str))
    print(f"\nDecision: {result.decision.outcome.value} ({result.decision.reason.value}) via {result.source}")
    return 0 if result.source == "llm" else 1


if __name__ == "__main__":
    raise SystemExit(main())
