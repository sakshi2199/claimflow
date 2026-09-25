"""Business rules (RULE_CHECK stage). Pure functions: the duplicate lookup happens in the workflow
and is passed in, so this module never touches the database.

Rules run in a fixed order and the first one that fires decides the claim. If none fires, the claim
is approved. Order matters: hard rejections come before human-review triggers.
"""

import re

from app.models.claim import Claim
from app.models.enums import ReasonCode
from app.services.decision import Decision, approve, reject, review
from app.services.policies import PolicyData, Procedure

# Notes shorter than this cannot plausibly support a claim ("See attached.", "N/A", ...).
MIN_NOTES_LENGTH = 25

# Deliberately simple keyword heuristic. It catches explicit hedging language but has known blind
# spots (paraphrased ambiguity is missed; negated hedging such as "no uncertainty" is flagged).
# Those blind spots are what the benchmark's hard categories measure, and what Phase 2 targets.
AMBIGUITY_PATTERN = re.compile(
    r"\b(possible|possibly|unclear|uncertain\w*|unknown|rule out|r/o|suspected|questionable"
    r"|inconclusive|may be|cannot determine|unable to determine|differential)\b",
    re.IGNORECASE,
)


def _check_duplicate(duplicate_of_id: int | None) -> Decision | None:
    if duplicate_of_id is not None:
        return reject(
            ReasonCode.DUPLICATE_CLAIM,
            "duplicate_check",
            f"Same patient, provider, procedure and date as earlier claim id {duplicate_of_id}.",
        )
    return None


def _check_known_procedure(claim: Claim, procedure: Procedure | None) -> Decision | None:
    if procedure is None:
        return review(
            ReasonCode.UNKNOWN_PROCEDURE_CODE,
            "procedure_catalog_lookup",
            f"Procedure code {claim.procedure_code} is not in the procedure catalog.",
        )
    return None


def _check_diagnosis_procedure_match(claim: Claim, procedure: Procedure | None) -> Decision | None:
    if procedure is None:
        return None
    prefix = claim.diagnosis_code.strip()[0]
    if prefix not in procedure.allowed_diagnosis_prefixes:
        return reject(
            ReasonCode.DIAGNOSIS_PROCEDURE_MISMATCH,
            "diagnosis_procedure_compatibility",
            f"Diagnosis {claim.diagnosis_code} is not compatible with procedure {procedure.code} "
            f"({procedure.description}).",
        )
    return None


def _check_manual_review_procedure(procedure: Procedure | None) -> Decision | None:
    if procedure is not None and procedure.requires_manual_review:
        return review(
            ReasonCode.MANUAL_REVIEW_REQUIRED_PROCEDURE,
            "manual_review_procedure",
            f"Procedure {procedure.code} ({procedure.description}) always requires manual review.",
        )
    return None


def _check_provider_watchlist(claim: Claim, policies: PolicyData) -> Decision | None:
    if claim.provider_id in policies.watchlist_providers:
        return review(
            ReasonCode.PROVIDER_ON_WATCHLIST,
            "provider_watchlist",
            f"Provider {claim.provider_id} is on the manual-review watchlist.",
        )
    return None


def _check_high_cost(claim: Claim, procedure: Procedure | None) -> Decision | None:
    if procedure is not None and claim.claim_amount > procedure.max_auto_amount:
        return review(
            ReasonCode.HIGH_COST_CLAIM,
            "high_cost_threshold",
            f"Amount {claim.claim_amount} exceeds the auto-approval limit "
            f"{procedure.max_auto_amount} for procedure {procedure.code}.",
        )
    return None


def _check_supporting_info(claim: Claim) -> Decision | None:
    notes = (claim.clinical_notes or "").strip()
    if len(notes) < MIN_NOTES_LENGTH:
        return review(
            ReasonCode.INSUFFICIENT_SUPPORTING_INFO,
            "notes_minimum_length",
            f"Clinical notes are missing or shorter than {MIN_NOTES_LENGTH} characters.",
            confidence=0.9,
        )
    return None


def _check_ambiguous_notes(claim: Claim) -> Decision | None:
    match = AMBIGUITY_PATTERN.search(claim.clinical_notes or "")
    if match:
        return review(
            ReasonCode.AMBIGUOUS_CLINICAL_NOTES,
            "ambiguity_keywords",
            f"Clinical notes contain hedging language ({match.group(0)!r}); needs human review.",
            confidence=0.6,
        )
    return None


def evaluate_deterministic_rules(claim: Claim, policies: PolicyData, duplicate_of_id: int | None) -> Decision | None:
    """Every rule that decides a claim without interpreting the meaning of its notes.

    Returns None when none fires: the claim is structurally fine and only the notes' meaning is left to judge.
    Phase 1 judges that with a keyword list; Phase 2 hands it to retrieval + LLM.
    """
    procedure = policies.procedures.get((claim.procedure_code or "").strip())
    return (
        _check_duplicate(duplicate_of_id)
        or _check_known_procedure(claim, procedure)
        or _check_diagnosis_procedure_match(claim, procedure)
        or _check_manual_review_procedure(procedure)
        or _check_provider_watchlist(claim, policies)
        or _check_high_cost(claim, procedure)
        or _check_supporting_info(claim)
    )


def keyword_notes_decision(claim: Claim) -> Decision:
    """Phase 1 notes judgement: keyword ambiguity check, otherwise approve."""
    return _check_ambiguous_notes(claim) or approve(
        ReasonCode.VALID_STANDARD_CLAIM,
        "default_approve",
        "Claim passed all validation and business rules.",
    )


def evaluate_business_rules(claim: Claim, policies: PolicyData, duplicate_of_id: int | None) -> Decision:
    """The complete Phase 1 rule chain (unchanged behaviour)."""
    return evaluate_deterministic_rules(claim, policies, duplicate_of_id) or keyword_notes_decision(claim)
