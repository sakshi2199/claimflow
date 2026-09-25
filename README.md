# ClaimFlow

> An evaluation-driven healthcare claims automation system combining deterministic workflows, human escalation, and eventually RAG/LLM reasoning.

**Status: Phase 1 (deterministic baseline + evaluation harness).** No LLM, RAG or frontend yet, by design.
All data is synthetic; there is no real patient data or PHI.

## Motivation

Claims operations teams triage large volumes of claims: most are routine, some are clearly invalid, and a
minority need a person's judgment. Automating this is only valuable if you can *measure* it: how many claims
are handled without a human, and how often the automated decision is wrong (especially wrongly approved).

ClaimFlow starts by building the measuring stick first. It has a reproducible labeled benchmark, an
evaluation framework, and a deterministic baseline. Later phases (RAG, LLM reasoning) are then judged against
real numbers, not impressions.

## Baseline results (measured)

Deterministic rules engine `rules-v1` on the 200-claim synthetic benchmark (seed 42). Produced by
`python -m app.evaluation.run_baseline`; raw output in
[evaluation_results/baseline_rules-v1.json](evaluation_results/baseline_rules-v1.json).

| Metric | Value |
|---|---|
| Claims evaluated | 200 |
| Decision accuracy | **92.0%** (184 / 200) |
| Automation rate (auto approve/reject) | **75.0%** |
| Human-review rate | 25.0% |
| Accuracy of automated decisions | 94.7% (142 / 150) |
| False approvals (approved but should not be) | 8 |
| Workflow success / failure rate | 100.0% / 0.0% |
| Latency mean / median / P95 | 1.01 ms / 0.89 ms / 1.68 ms (in-process SQLite) |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPROVED | 89.5% | 89.5% | 89.5% | 76 |
| REJECTED | 100.0% | 100.0% | 100.0% | 74 |
| HUMAN_REVIEW | 84.0% | 84.0% | 84.0% | 50 |

Confusion matrix (rows = expected, columns = predicted):

| | APPROVED | REJECTED | HUMAN_REVIEW |
|---|---:|---:|---:|
| **APPROVED** | 68 | 0 | 8 |
| **REJECTED** | 0 | 74 | 0 |
| **HUMAN_REVIEW** | 8 | 0 | 42 |

**How to read these numbers honestly.** All 16 errors come from two categories that were *deliberately built to
defeat keyword rules*, and these are the room for improvement that later phases target:

- `ambiguous_notes_subtle` (0/8 correct): ambiguous notes with no hedging keywords are auto-approved (these are the 8 false approvals).
- `negated_ambiguity` (0/8 correct): valid notes such as "no uncertainty about the diagnosis" trigger the keyword rule and are sent to review unnecessarily.

Every other category is 100%, which is expected because the benchmark and the rules were written by the same
author from the same policy files. It shows the rules implement the policy and handle the edge cases, not
that they would generalize to real claims. See [docs/synthetic_data.md](docs/synthetic_data.md#limits-read-this-before-quoting-numbers).
Latency was measured on SQLite in one process, so it is only meaningful for comparing variants on the same setup.

## Architecture

```mermaid
flowchart TD
    A[Synthetic claims JSONL] --> B[FastAPI]
    B --> C[Claims workflow]
    C --> D[Validation]
    C --> E[Rule engine]
    D --> G{Decision}
    E --> G
    G --> H[APPROVED / REJECTED / HUMAN_REVIEW]
    H --> F[(PostgreSQL + Evaluation)]
```

Details, the workflow state diagram and design decisions: [docs/architecture.md](docs/architecture.md).

```
claimflow/
  backend/app/
    api/          HTTP routes only
    models/       Claim, WorkflowRun, EvaluationRun (SQLAlchemy)
    schemas/      Pydantic request/response models
    services/     validation.py, rules.py (pure), duplicate_detection.py, workflow.py
    evaluation/   dataset_generator.py, metrics.py, evaluator.py, report.py, run_baseline.py
    core/ db/     config and database session
  backend/tests/  105 tests
  data/           synthetic_claims/ (benchmark), policies/ (procedure catalog, watchlist)
  docs/           architecture, dataset rules, metric definitions
  frontend/       placeholder (later)
```

## Technology

Python 3.12, FastAPI, SQLAlchemy 2, Pydantic 2, PostgreSQL 16, pytest, Docker Compose.
Planned for later phases: ChromaDB (RAG), an LLM, React + TypeScript.

## How claims are processed

```
RECEIVED -> VALIDATING -> RULE_CHECK -> COMPLETED   (or FAILED if the software throws)
                                          |
                          APPROVED / REJECTED / HUMAN_REVIEW  + machine-readable reason
```

1. **Validating** (structural, rejects): claim number format `CLM-YYYY-NNNNNN`, procedure and diagnosis code
   present and well-formed, amount greater than zero.
2. **Rule check** (first match wins): duplicate (same patient, provider, procedure and date as an earlier claim, REJECT) ->
   unknown procedure (REVIEW) -> diagnosis incompatible with procedure (REJECT) -> procedure always needs manual
   review (REVIEW) -> provider on watchlist (REVIEW) -> amount above auto-approval limit (REVIEW) -> notes missing or
   too short (REVIEW) -> hedging language in notes (REVIEW) -> otherwise APPROVE.
3. Every decision stores a **reason code** (`DUPLICATE_CLAIM`, `MISSING_PROCEDURE_CODE`,
   `INVALID_CLAIM_AMOUNT`, `DIAGNOSIS_PROCEDURE_MISMATCH`, `AMBIGUOUS_CLINICAL_NOTES`,
   `VALID_STANDARD_CLAIM`, ...), the rule that fired, and a plain-language explanation in a `WorkflowRun` row.

The rules live in [backend/app/services/rules.py](backend/app/services/rules.py) and
[validation.py](backend/app/services/validation.py) as pure functions, not in the API routes.

## Design philosophy

- **Measure first.** The benchmark and metrics exist before any AI, so improvements are provable.
- **Deterministic where possible.** Known rules stay as code.
- **Escalate, don't guess.** When the system cannot decide, the answer is `HUMAN_REVIEW`, never a coin flip.
- **Honest evaluation.** Failures are counted separately from rejections; undefined metrics are `null`, not 0;
  the dataset is hashed so runs are comparable; known weaknesses are in the benchmark on purpose.

### Why not use an LLM everywhere?

A missing procedure code, a malformed claim number, a duplicate, a negative amount or a known
diagnosis/procedure incompatibility are questions with exact answers. Code answers them in microseconds, for
free, identically every time, and can be unit-tested and audited. An LLM would be slower, cost money, and
sometimes be wrong about things that should never be wrong, and in a regulated domain "why was this rejected?"
needs a precise answer.

LLMs earn their place on **unstructured or ambiguous information**, where rules fail: reading clinical notes
for meaning rather than keywords, checking notes against policy text (RAG), and deciding whether a claim
deserves human attention. Later phases will add an LLM *only there*, keeping the deterministic checks in front,
and will use this benchmark to show whether it actually helps (for example on the two categories the baseline
gets wrong) and what it costs in latency.

## Why synthetic data

Real claims contain PHI, which brings HIPAA obligations and makes the project impossible to share publicly. A
synthetic generator also gives **ground truth by construction** (we know why each claim is what it is),
lets us build targeted edge cases, and is fully reproducible from a seed. The trade-off is that synthetic text
is simpler than real clinical text, so results are "on the benchmark", not real-world performance.
Generation rules: [docs/synthetic_data.md](docs/synthetic_data.md).

## Evaluation methodology

- The evaluator loads the JSONL benchmark, inserts each claim (tagged with an evaluation run id so runs are
  isolated), runs it through the **real** workflow, and compares actual with expected.
- Metrics: accuracy, per-class precision/recall/F1, automation rate, human-review rate, automated-decision
  accuracy, false approvals, reason-code accuracy, workflow success/failure, latency (mean, median, P95),
  accuracy by category, misclassified-claim list, and a confusion matrix.
- Runs are stored in the `evaluation_runs` table with the dataset SHA-256 and engine version for later
  comparison. Formulas: [docs/evaluation_metrics.md](docs/evaluation_metrics.md).

## Run locally

### Option A: Docker Compose (PostgreSQL + API)

```bash
docker compose up --build          # API on http://localhost:8000 (docs at /docs)
docker compose exec api python -m app.evaluation.run_baseline
```

Works without a `.env` (development defaults). Copy [.env.example](.env.example) to `.env` to change credentials.

### Option B: local Python (PostgreSQL required, or SQLite for a quick try)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

# PostgreSQL (e.g. `docker compose up -d db`):
export DATABASE_URL=postgresql+psycopg2://claimflow:claimflow@localhost:5432/claimflow
# ...or SQLite, only for trying it out:  DATABASE_URL=sqlite:///claimflow.db
# (PowerShell:  $env:DATABASE_URL = "sqlite:///claimflow.db")

uvicorn app.main:app --reload     # http://localhost:8000/docs
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + database check (503 if DB is down) |
| POST | `/claims` | Create a claim (201; 409 duplicate claim number; 422 bad payload) |
| GET | `/claims` | List live claims (`limit`, `offset`, `processing_status`, `actual_outcome`, `evaluation_run_id`) |
| GET | `/claims/{id}` | Claim with its workflow runs (404 if unknown) |
| POST | `/claims/{id}/process` | Run the workflow (409 if already processed) |
| POST | `/evaluation/run` | Run the benchmark and store the result (201) |
| GET | `/evaluation/latest` | Latest completed evaluation (404 if none) |

## Generate the benchmark

```bash
cd backend
python -m app.evaluation.dataset_generator --seed 42
```

Writes `data/synthetic_claims/claims_benchmark_v1.jsonl` and a manifest. The output is byte-identical for the
same seed, and a test fails if the committed file drifts from the generator.

## Run the evaluation

```bash
cd backend
python -m app.evaluation.run_baseline            # prints report, saves JSON, stores the run in the DB
python -m app.evaluation.run_baseline --label my-experiment
```

Or via the API: `curl -X POST localhost:8000/evaluation/run` then `curl localhost:8000/evaluation/latest`.

## Run the tests

```bash
cd backend
pytest                                   # 105 tests, ~8 s
pytest --cov=app --cov-report=term-missing
```

Tests run against in-memory SQLite so they need no services. That means PostgreSQL-specific behavior is
not covered by the automated suite (see below).

## Known limitations

- Tests and the recorded baseline were run on SQLite; the Docker Compose / PostgreSQL path is configured but was
  not exercised in the environment this was built in (the Docker daemon was unavailable).
- No database migrations (tables via `create_all`); add Alembic before real deployment.
- Live-claim `claim_number` uniqueness is enforced in the API, not by a database constraint (SQL unique
  constraints treat NULL scopes as distinct).
- `/evaluation/run` is synchronous (about 3 s for 200 claims); it would need a background job at larger sizes.
- The `confidence_score` is a rule-assigned constant, not a calibrated probability.

## Roadmap

1. **Phase 2:** RAG over policy documents (ChromaDB) to ground decisions in policy text.
2. **Phase 3:** LLM reasoning for ambiguous notes only, with structured output and confidence, evaluated against
   this benchmark (baseline vs RAG vs LLM, on accuracy, automation rate, false approvals and latency).
3. **Phase 4:** Human-in-the-loop review queue and React + TypeScript UI.
