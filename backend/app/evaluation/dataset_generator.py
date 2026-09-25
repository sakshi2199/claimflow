"""Deterministic synthetic claims benchmark generator. All data is fictional; there is no PHI.

Every claim carries a ground-truth `expected_outcome` and `expected_reason` that come from how the claim
was constructed (its category), NOT from running the rule engine. See docs/synthetic_data.md.

Determinism: all randomness comes from one `random.Random(seed)`; nothing reads the clock or global state.

Usage:
    python -m app.evaluation.dataset_generator --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.evaluation.policy_labels import write_policy_labels
from app.models.enums import ClaimOutcome, ReasonCode
from app.rag.documents import load_policy_documents
from app.services.policies import PolicyData, Procedure, load_policies

DEFAULT_SEED = 42
DATASET_NAME = "claims_benchmark_v1.jsonl"
START_DATE = date(2025, 1, 1)
DATE_SPAN_DAYS = 180

# category -> number of claims. Order is fixed so output is reproducible.
CATEGORY_COUNTS: dict[str, int] = {
    "valid_routine": 36,
    "normal_valid": 22,
    "similar_but_distinct": 10,
    "negated_ambiguity": 8,
    "duplicate_claim": 15,
    "malformed_claim_number": 12,
    "missing_procedure_code": 10,
    "missing_diagnosis_code": 10,
    "invalid_amount": 12,
    "diagnosis_procedure_mismatch": 15,
    "high_cost": 12,
    "ambiguous_notes_explicit": 10,
    "ambiguous_notes_subtle": 8,
    "incomplete_supporting_info": 10,
    "policy_review": 10,
}

CATEGORY_LABELS: dict[str, tuple[ClaimOutcome, ReasonCode]] = {
    "valid_routine": (ClaimOutcome.APPROVED, ReasonCode.VALID_STANDARD_CLAIM),
    "normal_valid": (ClaimOutcome.APPROVED, ReasonCode.VALID_STANDARD_CLAIM),
    "similar_but_distinct": (ClaimOutcome.APPROVED, ReasonCode.VALID_STANDARD_CLAIM),
    "negated_ambiguity": (ClaimOutcome.APPROVED, ReasonCode.VALID_STANDARD_CLAIM),
    "duplicate_claim": (ClaimOutcome.REJECTED, ReasonCode.DUPLICATE_CLAIM),
    "malformed_claim_number": (ClaimOutcome.REJECTED, ReasonCode.MALFORMED_CLAIM_NUMBER),
    "missing_procedure_code": (ClaimOutcome.REJECTED, ReasonCode.MISSING_PROCEDURE_CODE),
    "missing_diagnosis_code": (ClaimOutcome.REJECTED, ReasonCode.MISSING_DIAGNOSIS_CODE),
    "invalid_amount": (ClaimOutcome.REJECTED, ReasonCode.INVALID_CLAIM_AMOUNT),
    "diagnosis_procedure_mismatch": (ClaimOutcome.REJECTED, ReasonCode.DIAGNOSIS_PROCEDURE_MISMATCH),
    "high_cost": (ClaimOutcome.HUMAN_REVIEW, ReasonCode.HIGH_COST_CLAIM),
    "ambiguous_notes_explicit": (ClaimOutcome.HUMAN_REVIEW, ReasonCode.AMBIGUOUS_CLINICAL_NOTES),
    "ambiguous_notes_subtle": (ClaimOutcome.HUMAN_REVIEW, ReasonCode.AMBIGUOUS_CLINICAL_NOTES),
    "incomplete_supporting_info": (ClaimOutcome.HUMAN_REVIEW, ReasonCode.INSUFFICIENT_SUPPORTING_INFO),
    # policy_review overrides the reason per claim (manual-review procedure vs. watchlist provider)
    "policy_review": (ClaimOutcome.HUMAN_REVIEW, ReasonCode.MANUAL_REVIEW_REQUIRED_PROCEDURE),
}

# Categories whose claims are valid and can serve as the "original" for duplicates / near-duplicates.
ORIGINAL_CATEGORIES = ("valid_routine", "normal_valid")
ROUTINE_PROCEDURES = ("PRC-101", "PRC-102", "PRC-110", "PRC-130", "PRC-140")

# ICD-10-style diagnoses by first letter: (code, description, patient-reported symptom)
DIAGNOSES: dict[str, list[tuple[str, str, str]]] = {
    "J": [
        ("J06.9", "acute upper respiratory infection", "sore throat and nasal congestion"),
        ("J20.9", "acute bronchitis", "persistent cough and wheezing"),
        ("J45.909", "unspecified asthma", "shortness of breath and wheezing"),
    ],
    "R": [
        ("R05.9", "cough", "dry cough lasting several days"),
        ("R07.9", "chest pain, unspecified", "intermittent chest tightness"),
    ],
    "K": [
        ("K21.9", "gastro-esophageal reflux disease", "heartburn after meals"),
        ("K59.00", "constipation", "infrequent bowel movements"),
    ],
    "M": [
        ("M25.561", "pain in right knee", "knee pain with activity"),
        ("M54.50", "low back pain", "lower back pain radiating to the hip"),
        ("M17.11", "primary osteoarthritis of the right knee", "chronic knee stiffness"),
    ],
    "L": [("L30.9", "dermatitis, unspecified", "itchy red rash on the forearms")],
    "Z": [
        ("Z00.00", "general adult medical examination", "no acute complaints"),
        ("Z13.220", "screening for lipid disorders", "no acute complaints"),
    ],
    "E": [
        ("E11.9", "type 2 diabetes mellitus without complications", "elevated fasting glucose readings"),
        ("E78.5", "hyperlipidemia, unspecified", "elevated cholesterol on prior labs"),
    ],
    "I": [
        ("I10", "essential hypertension", "elevated blood pressure readings"),
        ("I25.10", "coronary artery disease", "exertional chest discomfort"),
    ],
    "S": [("S83.6", "sprain of the knee", "knee swelling after a fall")],
    "N": [("N39.0", "urinary tract infection", "burning on urination")],
    "O": [("O09.90", "supervision of high-risk pregnancy", "routine prenatal concerns")],
    "C": [("C34.90", "malignant neoplasm of the lung", "persistent cough and weight loss")],
    "D": [("D50.9", "iron deficiency anemia", "fatigue and pallor")],
}

# Clear notes: specific, no hedging language.
CLEAR_NOTES = (
    "Patient presented with {symptom}. Examination findings consistent with {dx}. "
    "{proc} performed as clinically indicated. Patient tolerated the visit well; follow-up in {weeks} weeks.",
    "Chief complaint: {symptom}. History and physical exam documented. Diagnosis: {dx}. "
    "Service provided: {proc}. No complications noted.",
    "Established patient seen for {dx}. {proc} completed and results reviewed with the patient. "
    "Treatment plan documented; return visit scheduled in {weeks} weeks.",
)

# Explicit ambiguity: contains hedging keywords the baseline rule engine looks for.
EXPLICIT_AMBIGUOUS_NOTES = (
    "Patient reports {symptom}. Diagnosis is unclear at this time; possible {dx} but cannot determine "
    "the cause without further testing. {proc} performed.",
    "Presentation is inconclusive. Suspected {dx}, rule out other causes. {proc} performed today.",
    "Symptoms of {symptom}. Differential includes {dx}; uncertain etiology. {proc} ordered for evaluation.",
)

# Subtle ambiguity: genuinely ambiguous to a human reader but uses NO hedging keywords.
# The keyword baseline is expected to miss these (they are hard cases for a later LLM phase).
SUBTLE_AMBIGUOUS_NOTES = (
    "Patient describes {symptom} that has waxed and waned for months. Findings overlap several conditions and "
    "the clinician deferred to further work-up before committing to a diagnosis. {proc} carried out.",
    "Documentation lists {symptom}, but the exam findings do not clearly point to {dx}; a reviewer should "
    "confirm the coded diagnosis matches the chart. {proc} billed.",
    "Coded as {dx}, yet the narrative describes findings that fit another condition better. "
    "Coding to be revisited by the clinician after {proc}.",
)

# Negated hedging: clearly valid notes that contain hedging keywords in a negated context.
# The keyword baseline is expected to wrongly flag these for human review.
NEGATED_AMBIGUITY_NOTES = (
    "Diagnosis of {dx} confirmed by exam and testing; there is no uncertainty about the diagnosis. "
    "{proc} completed without issue.",
    "Findings are not unclear: {symptom} with objective signs of {dx}. Nothing to rule out per the clinician. "
    "{proc} performed.",
    "Previously suspected {dx}, now confirmed on repeat testing. {proc} performed as planned.",
    "Diagnosis is not in question; no possible alternative explanations were identified. {proc} completed.",
)

INCOMPLETE_NOTES = (None, "", "See attached.", "N/A", "Pending documentation.", "Records to follow.")

MALFORMED_NUMBER_FORMATS = (
    "CLM-25-{num:06d}",
    "CLM{num:06d}",
    "clm-2025-{num:06d}",
    "CLM-2025-{short:05d}",
    "CLM-2025-{num}X",
    "{num:06d}",
    "CLM_2025_{num:06d}",
)

FIRST_CLAIM_SEQUENCE = 100001
LIVE_PROVIDERS = tuple(f"PRV-{n}" for n in range(1001, 1061))


class _Generator:
    def __init__(self, seed: int, policies: PolicyData) -> None:
        self.rng = random.Random(seed)
        self.policies = policies
        self.used_keys: set[tuple[str, str, str | None, date]] = set()
        self.watchlist = sorted(policies.watchlist_providers)
        self.standard = [p for p in policies.procedures.values() if not p.requires_manual_review]
        self.review_procs = [p for p in policies.procedures.values() if p.requires_manual_review]

    # ---- shared helpers -------------------------------------------------------------------------

    def _identity(self, procedure_code: str | None, providers: tuple[str, ...] | list[str] = LIVE_PROVIDERS):
        """Draw a (patient, provider, date) triple that has not been used with this procedure yet."""
        while True:
            patient = f"PT-{self.rng.randint(100000, 999999)}"
            provider = self.rng.choice(providers)
            day = START_DATE + timedelta(days=self.rng.randint(0, DATE_SPAN_DAYS))
            key = (patient, provider, procedure_code, day)
            if key not in self.used_keys:
                self.used_keys.add(key)
                return patient, provider, day

    def _diagnosis(self, prefixes: tuple[str, ...] | list[str]) -> tuple[str, str, str]:
        return self.rng.choice(DIAGNOSES[self.rng.choice(sorted(prefixes))])

    def _incompatible_diagnosis(self, procedure: Procedure) -> tuple[str, str, str]:
        prefixes = [p for p in sorted(DIAGNOSES) if p not in procedure.allowed_diagnosis_prefixes]
        return self._diagnosis(prefixes)

    def _amount(self, low: float, high: float) -> float:
        return round(self.rng.uniform(low, high), 2)

    def _typical_amount(self, procedure: Procedure, low_frac: float = 0.0, high_frac: float = 1.0) -> float:
        lo, hi = float(procedure.typical_amount_min), float(procedure.typical_amount_max)
        return self._amount(lo + (hi - lo) * low_frac, lo + (hi - lo) * high_frac)

    def _notes(self, templates: tuple[str, ...], procedure: Procedure, diagnosis: tuple[str, str, str]) -> str:
        return self.rng.choice(templates).format(
            symptom=diagnosis[2], dx=diagnosis[1], proc=procedure.description.capitalize(), weeks=self.rng.randint(2, 8)
        )

    def _record(
        self,
        category: str,
        *,
        patient: str,
        provider: str,
        day: date,
        procedure_code: str | None,
        diagnosis_code: str | None,
        amount: float,
        notes: str | None,
        reason: ReasonCode | None = None,
    ) -> dict[str, Any]:
        outcome, default_reason = CATEGORY_LABELS[category]
        return {
            "patient_id": patient,
            "provider_id": provider,
            "procedure_code": procedure_code,
            "diagnosis_code": diagnosis_code,
            "claim_amount": amount,
            "clinical_notes": notes,
            "submission_date": day.isoformat(),
            "category": category,
            "expected_outcome": outcome.value,
            "expected_reason": (reason or default_reason).value,
        }

    def _valid_claim(self, category: str, procedure: Procedure, notes_templates=CLEAR_NOTES, **overrides):
        """A claim that passes every rule, unless `overrides` deliberately break one."""
        patient, provider, day = self._identity(procedure.code)
        diagnosis = self._diagnosis(procedure.allowed_diagnosis_prefixes)
        fields = {
            "patient": patient,
            "provider": provider,
            "day": day,
            "procedure_code": procedure.code,
            "diagnosis_code": diagnosis[0],
            "amount": self._typical_amount(procedure),
            "notes": self._notes(notes_templates, procedure, diagnosis),
        }
        fields.update(overrides)
        return self._record(category, **fields)

    # ---- one builder per category ----------------------------------------------------------------

    def build_valid_routine(self) -> dict[str, Any]:
        procedure = self.policies.procedures[self.rng.choice(ROUTINE_PROCEDURES)]
        return self._valid_claim("valid_routine", procedure, amount=self._typical_amount(procedure, 0.0, 0.6))

    def build_normal_valid(self) -> dict[str, Any]:
        procedure = self.rng.choice(self.standard)
        return self._valid_claim("normal_valid", procedure, amount=self._typical_amount(procedure, 0.4, 1.0))

    def build_negated_ambiguity(self) -> dict[str, Any]:
        procedure = self.rng.choice(self.standard)
        return self._valid_claim("negated_ambiguity", procedure, notes_templates=NEGATED_AMBIGUITY_NOTES)

    def build_malformed_claim_number(self) -> dict[str, Any]:
        # The number itself is filled in during final numbering (needs the claim's position).
        return self._valid_claim("malformed_claim_number", self.rng.choice(self.standard))

    def build_missing_procedure_code(self) -> dict[str, Any]:
        procedure = self.rng.choice(self.standard)
        patient, provider, day = self._identity(None)
        diagnosis = self._diagnosis(procedure.allowed_diagnosis_prefixes)
        return self._record(
            "missing_procedure_code",
            patient=patient, provider=provider, day=day,
            procedure_code=self.rng.choice((None, "")),
            diagnosis_code=diagnosis[0],
            amount=self._typical_amount(procedure),
            notes=self._notes(CLEAR_NOTES, procedure, diagnosis),
        )  # fmt: skip

    def build_missing_diagnosis_code(self) -> dict[str, Any]:
        return self._valid_claim(
            "missing_diagnosis_code", self.rng.choice(self.standard), diagnosis_code=self.rng.choice((None, ""))
        )

    def build_invalid_amount(self) -> dict[str, Any]:
        amount = 0.0 if self.rng.random() < 0.4 else -self._amount(1, 500)
        return self._valid_claim("invalid_amount", self.rng.choice(self.standard), amount=amount)

    def build_diagnosis_procedure_mismatch(self) -> dict[str, Any]:
        narrow = [p for p in self.standard if len(p.allowed_diagnosis_prefixes) <= 5]
        procedure = self.rng.choice(narrow)
        diagnosis = self._incompatible_diagnosis(procedure)
        return self._valid_claim(
            "diagnosis_procedure_mismatch",
            procedure,
            diagnosis_code=diagnosis[0],
            notes=self._notes(CLEAR_NOTES, procedure, diagnosis),
        )

    def build_high_cost(self) -> dict[str, Any]:
        procedure = self.rng.choice(self.standard)
        amount = self._amount(float(procedure.max_auto_amount) * 1.15, float(procedure.max_auto_amount) * 3.0)
        return self._valid_claim("high_cost", procedure, amount=amount)

    def build_ambiguous_notes_explicit(self) -> dict[str, Any]:
        return self._valid_claim(
            "ambiguous_notes_explicit", self.rng.choice(self.standard), notes_templates=EXPLICIT_AMBIGUOUS_NOTES
        )

    def build_ambiguous_notes_subtle(self) -> dict[str, Any]:
        return self._valid_claim(
            "ambiguous_notes_subtle", self.rng.choice(self.standard), notes_templates=SUBTLE_AMBIGUOUS_NOTES
        )

    def build_incomplete_supporting_info(self) -> dict[str, Any]:
        return self._valid_claim(
            "incomplete_supporting_info", self.rng.choice(self.standard), notes=self.rng.choice(INCOMPLETE_NOTES)
        )

    def build_policy_review(self) -> dict[str, Any]:
        if self.rng.random() < 0.5:
            procedure = self.rng.choice(self.review_procs)
            return self._valid_claim("policy_review", procedure)
        # Routine claim from a watchlisted provider.
        procedure = self.policies.procedures[self.rng.choice(ROUTINE_PROCEDURES)]
        patient, provider, day = self._identity(procedure.code, self.watchlist)
        diagnosis = self._diagnosis(procedure.allowed_diagnosis_prefixes)
        return self._record(
            "policy_review",
            patient=patient, provider=provider, day=day,
            procedure_code=procedure.code,
            diagnosis_code=diagnosis[0],
            amount=self._typical_amount(procedure),
            notes=self._notes(CLEAR_NOTES, procedure, diagnosis),
            reason=ReasonCode.PROVIDER_ON_WATCHLIST,
        )  # fmt: skip

    def build_similar_but_distinct(self, original: dict[str, Any]) -> dict[str, Any]:
        """Same patient/provider/procedure as `original` but a clearly different date: a legitimate repeat visit."""
        procedure = self.policies.procedures[original["procedure_code"]]
        original_day = date.fromisoformat(original["submission_date"])
        while True:
            day = START_DATE + timedelta(days=self.rng.randint(0, DATE_SPAN_DAYS))
            key = (original["patient_id"], original["provider_id"], procedure.code, day)
            if abs((day - original_day).days) >= 14 and key not in self.used_keys:
                self.used_keys.add(key)
                break
        diagnosis = self._diagnosis(procedure.allowed_diagnosis_prefixes)
        return self._record(
            "similar_but_distinct",
            patient=original["patient_id"], provider=original["provider_id"], day=day,
            procedure_code=procedure.code,
            diagnosis_code=diagnosis[0],
            amount=self._typical_amount(procedure),
            notes=self._notes(CLEAR_NOTES, procedure, diagnosis),
        )  # fmt: skip

    def build_duplicate(self, original: dict[str, Any]) -> dict[str, Any]:
        """A resubmission of `original` under a new claim number: same patient, provider, procedure and date."""
        duplicate = dict(original)
        duplicate["category"] = "duplicate_claim"
        duplicate["expected_outcome"] = ClaimOutcome.REJECTED.value
        duplicate["expected_reason"] = ReasonCode.DUPLICATE_CLAIM.value
        return duplicate

    # ---- assembly --------------------------------------------------------------------------------

    def generate(self) -> list[dict[str, Any]]:
        derived = {"duplicate_claim", "similar_but_distinct"}
        records: list[dict[str, Any]] = []
        for category, count in CATEGORY_COUNTS.items():
            if category not in derived:
                builder = getattr(self, f"build_{category}")
                records.extend(builder() for _ in range(count))

        originals = [r for r in records if r["category"] in ORIGINAL_CATEGORIES]
        records.extend(
            self.build_similar_but_distinct(self.rng.choice(originals))
            for _ in range(CATEGORY_COUNTS["similar_but_distinct"])
        )
        self.rng.shuffle(records)

        # Duplicates must appear after their original so "earlier claim wins" is well defined.
        for original in self.rng.sample(originals, CATEGORY_COUNTS["duplicate_claim"]):
            position = next(i for i, record in enumerate(records) if record is original)
            records.insert(self.rng.randint(position + 1, len(records)), self.build_duplicate(original))

        return [self._number(sequence, record) for sequence, record in enumerate(records, FIRST_CLAIM_SEQUENCE)]

    def _number(self, sequence: int, record: dict[str, Any]) -> dict[str, Any]:
        if record["category"] == "malformed_claim_number":
            claim_number = self.rng.choice(MALFORMED_NUMBER_FORMATS).format(num=sequence, short=sequence % 100000)
        else:
            claim_number = f"CLM-2025-{sequence:06d}"
        return {"claim_number": claim_number, **record}


def generate_claims(seed: int = DEFAULT_SEED, policies: PolicyData | None = None) -> list[dict[str, Any]]:
    return _Generator(seed, policies or load_policies()).generate()


def serialize(claims: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(claim) + "\n" for claim in claims)


def build_manifest(claims: list[dict[str, Any]], seed: int, dataset_name: str, sha256: str) -> dict[str, Any]:
    by_category = Counter(c["category"] for c in claims)
    return {
        "dataset": dataset_name,
        "seed": seed,
        "total_claims": len(claims),
        "sha256": sha256,
        "outcome_distribution": dict(Counter(c["expected_outcome"] for c in claims)),
        "categories": {
            name: {
                "count": by_category[name],
                "expected_outcome": CATEGORY_LABELS[name][0].value,
            }
            for name in CATEGORY_COUNTS
        },
    }


def write_dataset(claims: list[dict[str, Any]], path: Path, seed: int) -> dict[str, Any]:
    content = serialize(claims)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    manifest = build_manifest(claims, seed, path.name, hashlib.sha256(content.encode("utf-8")).hexdigest())
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the ClaimFlow synthetic claims benchmark.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=get_settings().synthetic_claims_dir / DATASET_NAME)
    args = parser.parse_args(argv)

    claims = generate_claims(args.seed)
    manifest = write_dataset(claims, args.output, args.seed)
    labels_path = write_policy_labels(claims, load_policy_documents(), args.output)
    print(f"Wrote {manifest['total_claims']} claims to {args.output} (seed={args.seed})")
    print(f"Wrote retrieval ground-truth labels to {labels_path}")
    print(f"Outcome distribution: {manifest['outcome_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
