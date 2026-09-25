"""Structural validation (VALIDATING stage). Pure functions: no database, no policy data.

Each check returns a REJECTED Decision, or None if the claim passes that check.
"""

import re
from decimal import Decimal

from app.models.claim import Claim
from app.models.enums import ReasonCode
from app.services.decision import Decision, reject

CLAIM_NUMBER_RE = re.compile(r"^CLM-\d{4}-\d{6}$")
PROCEDURE_CODE_RE = re.compile(r"^PRC-\d{3}$")
DIAGNOSIS_CODE_RE = re.compile(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$")


def is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _check_claim_number(claim: Claim) -> Decision | None:
    if not CLAIM_NUMBER_RE.fullmatch(claim.claim_number or ""):
        return reject(
            ReasonCode.MALFORMED_CLAIM_NUMBER,
            "claim_number_format",
            f"Claim number {claim.claim_number!r} does not match CLM-YYYY-NNNNNN.",
        )
    return None


def _check_procedure_code(claim: Claim) -> Decision | None:
    if is_blank(claim.procedure_code):
        return reject(ReasonCode.MISSING_PROCEDURE_CODE, "procedure_code_present", "Procedure code is missing.")
    if not PROCEDURE_CODE_RE.fullmatch(claim.procedure_code.strip()):
        return reject(
            ReasonCode.INVALID_PROCEDURE_CODE_FORMAT,
            "procedure_code_format",
            f"Procedure code {claim.procedure_code!r} does not match PRC-NNN.",
        )
    return None


def _check_diagnosis_code(claim: Claim) -> Decision | None:
    if is_blank(claim.diagnosis_code):
        return reject(ReasonCode.MISSING_DIAGNOSIS_CODE, "diagnosis_code_present", "Diagnosis code is missing.")
    if not DIAGNOSIS_CODE_RE.fullmatch(claim.diagnosis_code.strip()):
        return reject(
            ReasonCode.INVALID_DIAGNOSIS_CODE_FORMAT,
            "diagnosis_code_format",
            f"Diagnosis code {claim.diagnosis_code!r} is not a valid ICD-10-style code.",
        )
    return None


def _check_amount(claim: Claim) -> Decision | None:
    if claim.claim_amount is None or Decimal(claim.claim_amount) <= 0:
        return reject(
            ReasonCode.INVALID_CLAIM_AMOUNT,
            "claim_amount_positive",
            f"Claim amount {claim.claim_amount} must be greater than zero.",
        )
    return None


def validate_claim(claim: Claim) -> Decision | None:
    """Return a REJECTED decision for the first failed check, or None if the claim is structurally valid."""
    return (
        _check_claim_number(claim)
        or _check_procedure_code(claim)
        or _check_diagnosis_code(claim)
        or _check_amount(claim)
    )
