# Synthetic benchmark: generation rules

All data is fictional. There is no real patient data and no PHI. Names of people, addresses and dates of birth
are not generated at all; patients are opaque ids like `PT-882169`.

- Generator: `backend/app/evaluation/dataset_generator.py`
- Output: `data/synthetic_claims/claims_benchmark_v1.jsonl` (+ `.manifest.json` with counts and SHA-256)
- Seed: `42` (all randomness comes from one `random.Random(seed)`; regenerating gives byte-identical output,
  enforced by tests)
- Regenerate: `cd backend && python -m app.evaluation.dataset_generator --seed 42`

## Where the labels come from

Each claim is built **from a category recipe**, and the label is the recipe's label. The label is never
produced by running the rule engine. Every claim has:

- `expected_outcome`: `APPROVED`, `REJECTED` or `HUMAN_REVIEW`
- `expected_reason`: the machine-readable reason a correct system should give
- `category`: which recipe built it (used for per-category accuracy)

Each defect category changes exactly **one** thing about an otherwise valid claim, so the correct answer is
unambiguous and does not depend on rule precedence.

## Reference data

- **Procedure codes** are synthetic (`PRC-101`, ...), defined in `data/policies/procedure_catalog.json`, with
  a description, typical price range, auto-approval limit, the diagnosis chapters that are compatible, and
  whether the procedure always needs manual review. They are *not* real CPT codes (CPT is licensed).
- **Diagnosis codes** are ICD-10-style codes (`J06.9`, `E11.9`, ...), chosen from a small list in the generator.
- **Provider watchlist**: `data/policies/provider_watchlist.json` (fictional `PRV-9001..9003`). All other
  claims use providers `PRV-1001..1060`.
- The catalog and watchlist encode the *policy* the labels are defined by. The rule engine reads the same
  files, so on "deterministic" categories agreement is expected. See "Limits" below.

## Categories (200 claims)

| Category | n | Expected outcome | Expected reason | How it is built |
|---|---:|---|---|---|
| `valid_routine` | 36 | APPROVED | VALID_STANDARD_CLAIM | Common low-cost procedures, price in the lower 60% of the typical range, clear notes |
| `normal_valid` | 22 | APPROVED | VALID_STANDARD_CLAIM | Any non-review procedure, price in the upper 60% of the typical range, clear notes |
| `similar_but_distinct` | 10 | APPROVED | VALID_STANDARD_CLAIM | Same patient/provider/procedure as an earlier valid claim but a date at least 14 days apart: a legitimate repeat visit (guards against over-eager duplicate detection) |
| `negated_ambiguity` | 8 | APPROVED | VALID_STANDARD_CLAIM | Valid claims whose notes contain hedging words in a *negated* way ("no uncertainty", "not unclear", "previously suspected, now confirmed"). **Hard case** |
| `duplicate_claim` | 15 | REJECTED | DUPLICATE_CLAIM | Copy of an earlier valid claim under a new claim number, placed later in the file |
| `malformed_claim_number` | 12 | REJECTED | MALFORMED_CLAIM_NUMBER | Seven wrong formats (`CLM-25-...`, missing dash, lowercase, wrong digit count, ...) |
| `missing_procedure_code` | 10 | REJECTED | MISSING_PROCEDURE_CODE | `null` or `""` |
| `missing_diagnosis_code` | 10 | REJECTED | MISSING_DIAGNOSIS_CODE | `null` or `""` |
| `invalid_amount` | 12 | REJECTED | INVALID_CLAIM_AMOUNT | Zero (40%) or negative |
| `diagnosis_procedure_mismatch` | 15 | REJECTED | DIAGNOSIS_PROCEDURE_MISMATCH | A procedure with a narrow set of compatible diagnoses paired with an incompatible diagnosis chapter (e.g. knee MRI with a respiratory diagnosis) |
| `high_cost` | 12 | HUMAN_REVIEW | HIGH_COST_CLAIM | Amount 1.15x to 3x the procedure's auto-approval limit |
| `ambiguous_notes_explicit` | 10 | HUMAN_REVIEW | AMBIGUOUS_CLINICAL_NOTES | Notes with obvious hedging ("unclear", "possible", "rule out", "inconclusive", ...) |
| `ambiguous_notes_subtle` | 8 | HUMAN_REVIEW | AMBIGUOUS_CLINICAL_NOTES | Notes that are ambiguous to a human reader but use **no** hedging keywords ("findings overlap several conditions...", "coding to be revisited"). **Hard case** |
| `incomplete_supporting_info` | 10 | HUMAN_REVIEW | INSUFFICIENT_SUPPORTING_INFO | Notes are `null`, empty, or a placeholder ("See attached.", "N/A", ...) |
| `policy_review` | 10 | HUMAN_REVIEW | MANUAL_REVIEW_REQUIRED_PROCEDURE / PROVIDER_ON_WATCHLIST | Half use a procedure that always requires manual review; half are routine claims from a watchlisted provider |

Outcome distribution: 76 APPROVED, 74 REJECTED, 50 HUMAN_REVIEW.

## Ordering

Claims are shuffled with the seed, then each duplicate is inserted at a random position **after** its
original, and claim numbers are assigned sequentially (`CLM-2025-100001`...). File order is submission order,
which duplicate detection relies on (the earlier claim wins).

## Limits (read this before quoting numbers)

- The benchmark was authored by the same person who wrote the rule engine, and both read the same policy
  files. Near-perfect accuracy on the deterministic categories is therefore **expected by construction**; it
  shows the rules implement the stated policy and handle edge cases, not that they generalize to real claims.
- The two **hard categories** (`ambiguous_notes_subtle`, `negated_ambiguity`) exist on purpose. They contain
  the kind of language a keyword rule cannot handle, and they are what a later LLM/RAG phase should improve.
  Without them the baseline would score ~100% and there would be nothing to measure improvement against.
- Notes are generated from a small set of templates, so they are far less varied than real clinical text.
  Any Phase 2 result on this data should be described as "on the synthetic benchmark", not as real-world
  performance.
