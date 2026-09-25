from decimal import Decimal

import pytest

from app.models.enums import ClaimOutcome, ReasonCode
from app.services.duplicate_detection import find_duplicate
from app.services.policies import load_policies
from app.services.rules import evaluate_business_rules
from tests.conftest import make_claim

POLICIES = load_policies()


def decide(duplicate_of_id: int | None = None, **overrides):
    return evaluate_business_rules(make_claim(**overrides), POLICIES, duplicate_of_id)


def test_clean_claim_is_approved() -> None:
    decision = decide()
    assert decision.outcome == ClaimOutcome.APPROVED
    assert decision.reason == ReasonCode.VALID_STANDARD_CLAIM


def test_duplicate_is_rejected() -> None:
    decision = decide(duplicate_of_id=7)
    assert (decision.outcome, decision.reason) == (ClaimOutcome.REJECTED, ReasonCode.DUPLICATE_CLAIM)
    assert "7" in decision.explanation


def test_incompatible_diagnosis_is_rejected() -> None:
    # Knee MRI (PRC-160) only allows musculoskeletal (M/S) diagnoses.
    decision = decide(procedure_code="PRC-160", diagnosis_code="J06.9", claim_amount=Decimal("800"))
    assert (decision.outcome, decision.reason) == (ClaimOutcome.REJECTED, ReasonCode.DIAGNOSIS_PROCEDURE_MISMATCH)


def test_compatible_diagnosis_is_approved() -> None:
    decision = decide(procedure_code="PRC-160", diagnosis_code="M25.561", claim_amount=Decimal("800"))
    assert decision.outcome == ClaimOutcome.APPROVED


def test_amount_over_limit_goes_to_human_review_but_limit_itself_is_approved() -> None:
    limit = POLICIES.procedures["PRC-101"].max_auto_amount
    assert decide(claim_amount=limit).outcome == ClaimOutcome.APPROVED
    over = decide(claim_amount=limit + Decimal("0.01"))
    assert (over.outcome, over.reason) == (ClaimOutcome.HUMAN_REVIEW, ReasonCode.HIGH_COST_CLAIM)


def test_unknown_procedure_goes_to_human_review() -> None:
    decision = decide(procedure_code="PRC-999")
    assert (decision.outcome, decision.reason) == (ClaimOutcome.HUMAN_REVIEW, ReasonCode.UNKNOWN_PROCEDURE_CODE)


def test_manual_review_procedure_goes_to_human_review() -> None:
    decision = decide(procedure_code="PRC-210", diagnosis_code="M54.50", claim_amount=Decimal("20000"))
    assert decision.reason == ReasonCode.MANUAL_REVIEW_REQUIRED_PROCEDURE


def test_watchlist_provider_goes_to_human_review() -> None:
    decision = decide(provider_id="PRV-9001")
    assert (decision.outcome, decision.reason) == (ClaimOutcome.HUMAN_REVIEW, ReasonCode.PROVIDER_ON_WATCHLIST)


@pytest.mark.parametrize("notes", [None, "", "N/A", "See attached.", "Pending documentation."])
def test_missing_or_short_notes_go_to_human_review(notes: str | None) -> None:
    decision = decide(clinical_notes=notes)
    assert (decision.outcome, decision.reason) == (ClaimOutcome.HUMAN_REVIEW, ReasonCode.INSUFFICIENT_SUPPORTING_INFO)


@pytest.mark.parametrize(
    "notes",
    [
        "Diagnosis is unclear at this time; further testing needed before treatment.",
        "Suspected bronchitis, rule out pneumonia. Chest exam performed today.",
        "Symptoms inconclusive after the exam; patient advised to return next week.",
    ],
)
def test_explicit_hedging_goes_to_human_review(notes: str) -> None:
    decision = decide(clinical_notes=notes)
    assert (decision.outcome, decision.reason) == (ClaimOutcome.HUMAN_REVIEW, ReasonCode.AMBIGUOUS_CLINICAL_NOTES)
    assert decision.confidence < 1.0


def test_known_limitation_negated_hedging_is_flagged_and_subtle_ambiguity_is_missed() -> None:
    """Documents the baseline's blind spots (the reason later phases add an LLM). If these ever start
    passing/failing differently, the benchmark's hard categories need to be revisited."""
    negated = decide(clinical_notes="Diagnosis confirmed by exam; there is no uncertainty about the cause here.")
    assert negated.outcome == ClaimOutcome.HUMAN_REVIEW  # false positive

    subtle = decide(
        clinical_notes="Findings overlap several conditions and the clinician deferred the diagnosis to later work-up."
    )
    assert subtle.outcome == ClaimOutcome.APPROVED  # false negative


def test_rule_precedence_duplicate_beats_everything_else() -> None:
    decision = decide(duplicate_of_id=1, procedure_code="PRC-160", diagnosis_code="J06.9", clinical_notes="N/A")
    assert decision.reason == ReasonCode.DUPLICATE_CLAIM


def test_rule_precedence_rejection_beats_human_review() -> None:
    # Mismatch (reject) and high cost (review) both apply: the rejection wins.
    decision = decide(procedure_code="PRC-160", diagnosis_code="J06.9", claim_amount=Decimal("99999"))
    assert decision.outcome == ClaimOutcome.REJECTED


# ---- duplicate detection against the database ---------------------------------------------------


def test_duplicate_found_for_same_patient_provider_procedure_date(db, saved_claim) -> None:
    original = saved_claim()
    resubmission = saved_claim(claim_number="CLM-2025-000002")
    assert find_duplicate(db, resubmission).id == original.id


def test_original_is_not_a_duplicate_of_its_later_resubmission(db, saved_claim) -> None:
    original = saved_claim()
    saved_claim(claim_number="CLM-2025-000002")
    assert find_duplicate(db, original) is None


@pytest.mark.parametrize(
    "different",
    [
        {"submission_date": __import__("datetime").date(2025, 3, 2)},
        {"patient_id": "PT-999999"},
        {"provider_id": "PRV-2002"},
        {"procedure_code": "PRC-102"},
    ],
)
def test_no_duplicate_when_any_key_field_differs(db, saved_claim, different: dict) -> None:
    saved_claim()
    other = saved_claim(claim_number="CLM-2025-000002", **different)
    assert find_duplicate(db, other) is None


def test_duplicate_detection_is_scoped_to_evaluation_run(db, saved_claim) -> None:
    from app.models.evaluation_run import EvaluationRun

    runs = [
        EvaluationRun(label="x", engine_version="v", dataset_name="d", dataset_sha256="0" * 64, total_claims=1)
        for _ in range(2)
    ]
    db.add_all(runs)
    db.commit()

    saved_claim(evaluation_run_id=runs[0].id)
    live = saved_claim(claim_number="CLM-2025-000002")  # live claim, same key as a benchmark claim
    other_run = saved_claim(claim_number="CLM-2025-000003", evaluation_run_id=runs[1].id)
    same_run = saved_claim(claim_number="CLM-2025-000004", evaluation_run_id=runs[0].id)

    assert find_duplicate(db, live) is None
    assert find_duplicate(db, other_run) is None
    assert find_duplicate(db, same_run) is not None
