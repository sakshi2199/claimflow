from decimal import Decimal

import pytest

from app.models.enums import ClaimOutcome, ReasonCode
from app.services.validation import validate_claim
from tests.conftest import make_claim


def test_valid_claim_passes_validation() -> None:
    assert validate_claim(make_claim()) is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"claim_number": "INVALID"}, ReasonCode.MALFORMED_CLAIM_NUMBER),
        ({"claim_number": "CLM-25-000001"}, ReasonCode.MALFORMED_CLAIM_NUMBER),
        ({"claim_number": "clm-2025-000001"}, ReasonCode.MALFORMED_CLAIM_NUMBER),
        ({"claim_number": "CLM-2025-00001"}, ReasonCode.MALFORMED_CLAIM_NUMBER),
        ({"claim_number": "CLM-2025-000001\n"}, ReasonCode.MALFORMED_CLAIM_NUMBER),
        ({"procedure_code": None}, ReasonCode.MISSING_PROCEDURE_CODE),
        ({"procedure_code": ""}, ReasonCode.MISSING_PROCEDURE_CODE),
        ({"procedure_code": "   "}, ReasonCode.MISSING_PROCEDURE_CODE),
        ({"procedure_code": "99213"}, ReasonCode.INVALID_PROCEDURE_CODE_FORMAT),
        ({"diagnosis_code": None}, ReasonCode.MISSING_DIAGNOSIS_CODE),
        ({"diagnosis_code": ""}, ReasonCode.MISSING_DIAGNOSIS_CODE),
        ({"diagnosis_code": "6.9"}, ReasonCode.INVALID_DIAGNOSIS_CODE_FORMAT),
        ({"claim_amount": Decimal("0")}, ReasonCode.INVALID_CLAIM_AMOUNT),
        ({"claim_amount": Decimal("-10.00")}, ReasonCode.INVALID_CLAIM_AMOUNT),
    ],
)
def test_validation_rejects_with_specific_reason(overrides: dict, reason: ReasonCode) -> None:
    decision = validate_claim(make_claim(**overrides))
    assert decision is not None
    assert decision.outcome == ClaimOutcome.REJECTED
    assert decision.reason == reason
    assert decision.confidence == 1.0


def test_validation_reports_first_failure_in_fixed_order() -> None:
    claim = make_claim(claim_number="BAD", procedure_code=None, claim_amount=Decimal("-1"))
    assert validate_claim(claim).reason == ReasonCode.MALFORMED_CLAIM_NUMBER


@pytest.mark.parametrize("diagnosis", ["J06.9", "I10", "S83.6", "Z13.220", "M25.561"])
def test_valid_diagnosis_formats_pass(diagnosis: str) -> None:
    assert validate_claim(make_claim(diagnosis_code=diagnosis)) is None
