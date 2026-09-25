import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.evaluation.dataset_generator import (
    CATEGORY_COUNTS,
    DATASET_NAME,
    DEFAULT_SEED,
    generate_claims,
    main,
    serialize,
    write_dataset,
)
from app.models.enums import ClaimOutcome, ReasonCode
from app.services.validation import CLAIM_NUMBER_RE

REQUIRED_KEYS = {
    "claim_number", "patient_id", "provider_id", "procedure_code", "diagnosis_code",
    "claim_amount", "clinical_notes", "submission_date", "category", "expected_outcome", "expected_reason",
}  # fmt: skip


@pytest.fixture(scope="module")
def claims() -> list[dict]:
    return generate_claims(DEFAULT_SEED)


def test_generates_at_least_100_claims_matching_category_counts(claims) -> None:
    assert len(claims) >= 100
    assert Counter(c["category"] for c in claims) == CATEGORY_COUNTS


def test_every_claim_has_all_fields_and_a_valid_ground_truth(claims) -> None:
    valid_outcomes = {o.value for o in ClaimOutcome}
    valid_reasons = {r.value for r in ReasonCode}
    for claim in claims:
        assert set(claim) == REQUIRED_KEYS
        assert claim["expected_outcome"] in valid_outcomes
        assert claim["expected_reason"] in valid_reasons


def test_all_three_outcomes_are_represented(claims) -> None:
    outcomes = Counter(c["expected_outcome"] for c in claims)
    assert set(outcomes) == {o.value for o in ClaimOutcome}
    assert min(outcomes.values()) >= 20  # no degenerate class


def test_same_seed_is_byte_for_byte_reproducible() -> None:
    assert serialize(generate_claims(42)) == serialize(generate_claims(42))


def test_different_seed_produces_different_data() -> None:
    assert serialize(generate_claims(42)) != serialize(generate_claims(43))


def test_committed_dataset_matches_generator_output() -> None:
    """Guards against editing the generator or catalog without regenerating data/synthetic_claims."""
    path = get_settings().synthetic_claims_dir / DATASET_NAME
    assert path.read_text(encoding="utf-8") == serialize(generate_claims(DEFAULT_SEED))


def test_claim_numbers_are_unique_and_only_malformed_category_is_malformed(claims) -> None:
    numbers = [c["claim_number"] for c in claims]
    assert len(set(numbers)) == len(numbers)
    for claim in claims:
        well_formed = bool(CLAIM_NUMBER_RE.fullmatch(claim["claim_number"]))
        assert well_formed == (claim["category"] != "malformed_claim_number"), claim["claim_number"]


def test_duplicates_share_key_with_an_earlier_claim(claims) -> None:
    key = lambda c: (c["patient_id"], c["provider_id"], c["procedure_code"], c["submission_date"])  # noqa: E731
    first_seen: dict[tuple, int] = {}
    for position, claim in enumerate(claims):
        if claim["category"] == "duplicate_claim":
            assert key(claim) in first_seen and first_seen[key(claim)] < position
            assert claim["claim_number"] != claims[first_seen[key(claim)]]["claim_number"]
        else:
            assert key(claim) not in first_seen, f"unintended key collision at {claim['claim_number']}"
            first_seen[key(claim)] = position


def test_defect_categories_contain_exactly_the_intended_defect(claims) -> None:
    by_category: dict[str, list[dict]] = {}
    for c in claims:
        by_category.setdefault(c["category"], []).append(c)

    assert all(not c["procedure_code"] for c in by_category["missing_procedure_code"])
    assert all(not c["diagnosis_code"] for c in by_category["missing_diagnosis_code"])
    assert all(c["claim_amount"] <= 0 for c in by_category["invalid_amount"])
    assert all(len(c["clinical_notes"] or "") < 25 for c in by_category["incomplete_supporting_info"])
    # Valid categories have no structural defects.
    for name in ("valid_routine", "normal_valid", "similar_but_distinct", "negated_ambiguity"):
        for c in by_category[name]:
            assert c["procedure_code"] and c["diagnosis_code"] and c["claim_amount"] > 0
            assert len(c["clinical_notes"]) >= 25
            assert re.fullmatch(r"PRC-\d{3}", c["procedure_code"])


def test_write_dataset_creates_jsonl_and_manifest(tmp_path: Path, claims) -> None:
    path = tmp_path / "bench.jsonl"
    manifest = write_dataset(claims, path, DEFAULT_SEED)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(claims) == manifest["total_claims"]
    assert json.loads(lines[0])["claim_number"] == claims[0]["claim_number"]
    assert json.loads((tmp_path / "bench.manifest.json").read_text())["seed"] == DEFAULT_SEED


def test_cli_writes_requested_output(tmp_path: Path, capsys) -> None:
    out = tmp_path / "custom.jsonl"
    assert main(["--seed", "7", "--output", str(out)]) == 0
    assert out.exists()
    assert "Wrote" in capsys.readouterr().out
