# ClaimFlow

> An evaluation-driven healthcare claims automation system combining deterministic workflows, human escalation, and RAG/LLM reasoning where rules are not enough.

**Status: Phase 1 complete and measured. Phase 2 (RAG + LLM) is built, tested, and its retrieval is measured; the
end-to-end LLM workflow results are not measured yet** because no LLM credentials were available when it was built
(see [Phase 2 results](#phase-2-results) and [Run Phase 2 yourself](#run-phase-2-yourself)).

> **All data is synthetic** (fictional patients, providers, codes, and a fictional payer's policies). This is a
> portfolio project. It is **not a clinical decision system**, it makes **no HIPAA or other compliance claims**, and
> **benchmark performance does not imply real-world healthcare performance**.

## Motivation

Claims operations teams triage large volumes of claims: most are routine, some are clearly invalid, and a minority
need a person's judgment. Automating this is only valuable if you can *measure* it: how many claims are handled
without a human, and how often the automated decision is wrong (especially wrongly approved).

ClaimFlow builds the measuring stick first: a reproducible labeled benchmark, an evaluation framework, and a
deterministic baseline. AI is then added only where deterministic rules run out, and judged against those numbers.

## Architecture

**Phase 1: deterministic.** Validation and business rules decide every claim.

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

**Phase 2: RAG + LLM only when the rules cannot decide.**

```mermaid
flowchart TD
    C[Claim] --> V[Deterministic validation + rules]
    V -->|clear outcome| D1[APPROVED / REJECTED / HUMAN_REVIEW<br/>no LLM]
    V -->|all rules passed; notes' meaning left to judge| RET[Policy retrieval<br/>ChromaDB]
    RET --> LLM[LLM interpretation<br/>structured output, cited policy ids]
    LLM --> S{valid, grounded,<br/>within retries?}
    S -->|no| FB[HUMAN_REVIEW fallback]
    S -->|yes| CF{confidence >= threshold?}
    CF -->|yes| D2[LLM decision]
    CF -->|no| FB2[HUMAN_REVIEW]
```

More detail, state diagram, design decisions and the `WorkflowRun` observability schema:
[docs/architecture.md](docs/architecture.md).

```
claimflow/
  backend/app/
    api/          HTTP routes only
    models/       Claim, WorkflowRun, EvaluationRun, RetrievalLog, LLMCall
    schemas/      Pydantic request/response models
    services/     validation.py, rules.py (pure), duplicate_detection.py, workflow.py,
                  interpretation.py, llm/ (providers, schema, prompts, retries)
    rag/          documents, chunking, embeddings, ChromaDB store, retrieval, presets
    evaluation/   Phase 1: dataset_generator, metrics, evaluator, run_baseline
                  Phase 2: retrieval_metrics, run_retrieval, phase2_metrics, thresholds, comparison, run_phase2
  backend/tests/  263 tests
  data/           synthetic_claims/ (benchmark + retrieval labels), policies/ (rules data + documents/ knowledge base)
  docs/           architecture, dataset rules, metric definitions, RAG evaluation
  evaluation_results/  saved baseline and retrieval results
```

## Technology

Python 3.12 target (developed and tested on 3.11), FastAPI, SQLAlchemy 2, Pydantic 2, PostgreSQL 16, ChromaDB,
all-MiniLM-L6-v2 embeddings (ONNX), pytest, Docker Compose. LLM access goes through a provider interface with
Anthropic and OpenAI-compatible adapters.

## How claims are processed

```
RECEIVED -> VALIDATING -> RULE_CHECK -> [INTERPRETING] -> COMPLETED   (FAILED if the software throws)
```

1. **Validating** (structural, rejects): claim number format, procedure/diagnosis code present and well-formed, amount above zero.
2. **Rule check** (first match wins): duplicate (REJECT) -> unknown procedure (REVIEW) -> diagnosis incompatible with
   procedure (REJECT) -> special procedure needing manual review (REVIEW) -> provider on watchlist (REVIEW) ->
   amount above auto-approval limit (REVIEW) -> notes missing or too short (REVIEW).
3. **Notes interpretation** (the only judgement left): in Phase 1 a keyword list for hedging language decides
   (REVIEW if found, else APPROVE). In Phase 2, if an LLM is configured, retrieval + LLM decide instead.
4. Every decision stores a machine-readable **reason code**, the rule that fired, and an explanation in `WorkflowRun`.

## Why deterministic rules stay deterministic

A missing procedure code, a malformed claim number, a duplicate, a negative amount or a known
diagnosis/procedure incompatibility are questions with exact answers. Code answers them in microseconds, for free,
identically every time, and can be unit-tested and audited. An LLM would be slower, cost money, and sometimes be
wrong about things that should never be wrong. Phase 2 therefore runs **all** of these first, and a test asserts the
LLM is never called for any of them.

## Where RAG and the LLM are used

Only for claims that pass every deterministic rule, where the remaining question is whether the clinical notes
adequately and unambiguously support the claim. That is a reading-comprehension problem keyword rules handle
badly: paraphrased ambiguity has no keywords, and negated hedging ("no uncertainty about the diagnosis") triggers
them wrongly. On the benchmark this is 94 of 200 claims (47%); the other 106 never touch retrieval or the LLM.

- **RAG:** the claim's procedure and notes are turned into a query; ChromaDB returns the most relevant passages from
  19 synthetic policies; the top 5 policies are the model's only permitted evidence.
- **LLM:** receives structured claim fields, the notes, and those policy excerpts. It must return JSON with
  `decision`, `confidence` (0 to 1), `reason_code`, a summary of at most two sentences, and `cited_policy_ids`.
  No reasoning is stored.
- **Grounding:** every cited id must be in the retrieved context; otherwise the reply is invalid, retried with
  feedback, and finally escalated. The unsupported-citation rate is a reported metric.

## Human-handoff strategy

Escalation is the default whenever the system is not sure or not able:

- confidence below `LLM_CONFIDENCE_THRESHOLD` on an APPROVED/REJECTED answer -> `HUMAN_REVIEW` (`LLM_LOW_CONFIDENCE`);
- provider timeout or error after the retry budget, malformed output, invalid citations, retrieval failure, or
  no policy found -> `HUMAN_REVIEW` (`LLM_PROCESSING_FALLBACK`);
- an LLM `HUMAN_REVIEW` answer is always accepted (escalating is the safe direction).

Retries are bounded (`1 + LLM_MAX_RETRIES` calls at most), with a per-call timeout and exponential backoff.
The right threshold is not assumed: `run_phase2` evaluates 0.60, 0.70, 0.80 and 0.90 by replaying the recorded
model answers, producing the automation-versus-safety trade-off table.

## Baseline results (Phase 1, measured)

Deterministic rules engine `rules-v1`, 200-claim synthetic benchmark (seed 42), from `python -m app.evaluation.run_baseline`;
raw output in [evaluation_results/baseline_rules-v1.json](evaluation_results/baseline_rules-v1.json).

| Metric | Value |
|---|---|
| Decision accuracy | **92.0%** (184 / 200) |
| Automation rate | **75.0%** |
| Human-review rate | 25.0% |
| Accuracy of automated decisions | 94.7% (142 / 150) |
| False approvals | 8 |
| Workflow success / failure | 100% / 0% |
| Latency mean / median / P95 | 1.01 / 0.89 / 1.68 ms (in-process SQLite) |

APPROVED precision/recall 89.5% / 89.5%; REJECTED 100% / 100%; HUMAN_REVIEW 84.0% / 84.0%.

All 16 errors sit in two categories built on purpose to defeat keyword rules: `ambiguous_notes_subtle` (0/8, the 8 false
approvals) and `negated_ambiguity` (0/8). Every other category is 100%, which is expected because the benchmark and
the rules were written by the same author from the same policy files. See
[docs/synthetic_data.md](docs/synthetic_data.md#limits-read-this-before-quoting-numbers).

## Phase 2 results

### Retrieval (measured)

94 labeled queries, real all-MiniLM-L6-v2 embeddings. Ground truth comes from the benchmark definition
(procedure coverage policy + one cross-cutting policy per category, so 2 relevant policies per claim; hence Recall@1
is capped at 0.5). Written before retrieval was run. Full method and caveats: [docs/rag_evaluation.md](docs/rag_evaluation.md).

| Config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| baseline (1200-char chunks, notes-only query) | 0.309 | 0.527 | 0.665 | 0.770 |
| chunking (400/80 chunks + title) | 0.324 | 0.484 | 0.596 | 0.789 |
| query (+ procedure in query) | 0.500 | 0.574 | 0.697 | 1.000 |
| **filtered** (+ metadata filter), the default | 0.500 | 0.734 | **0.947** | 1.000 |

Honest reading: the improved configuration raises Recall@5 from 0.665 to 0.947, but smaller chunks alone made it
*worse*, MRR saturates at 1.0 (the procedure policy is always relevant, so MRR stops discriminating), and part of the
filter's gain is a smaller candidate pool. Most importantly, `POL-AMB-002`, the policy that matters for the
`negated_ambiguity` category, is retrieved into the top 5 for only 4 of 8 such claims (the baseline did 6 of 8).

### RAG + LLM workflow (not measured yet)

**No Phase 2 workflow metrics are reported here because none have been measured.** No LLM credentials were available
in the environment where this was built, so the LLM path was implemented and tested with mocked providers only
(263 tests; none calls a real model). Accuracy, false approvals, the two hard categories, the threshold table, LLM
invocation rate, retries, citations and token usage will all be produced by one command; see below. This table is to be
filled in from `evaluation_results/comparison_phase1_vs_phase2.txt` after that run:

| Metric | Phase 1 | Phase 2 |
|---|---:|---:|
| Decision accuracy | 92.0% | not yet measured |
| Automation rate | 75.0% | not yet measured |
| False approvals | 8 | not yet measured |
| Human-review rate | 25.0% | not yet measured |
| `ambiguous_notes_subtle` | 0/8 | not yet measured |
| `negated_ambiguity` | 0/8 | not yet measured |
| Retrieval Recall@5 / MRR | N/A | 0.947 / 1.000 (retrieval benchmark, see above) |

## Evaluation methodology

- **Same benchmark:** Phase 2 runs the identical 200-claim file as Phase 1. The comparison refuses to run if the dataset
  SHA-256 or claim count differ.
- **Retrieval:** Recall@K, Hit@K and MRR against benchmark-defined relevant policies; four configurations compared.
- **LLM decisions:** accuracy, per-category accuracy, automation and human-review rates, automated-decision accuracy, false
  approvals, false rejections, plus LLM invocation rate, calls per claim, retry rate, LLM failure rate, structured-output
  failure rate, unsupported-citation rate, retrieval/LLM/total latency, tokens (and cost only if you configure prices).
- **Thresholds:** several confidence thresholds replayed from stored answers (exactly reproducing the live run at its own threshold).
- Formulas: [docs/evaluation_metrics.md](docs/evaluation_metrics.md). Every run is saved in the `evaluation_runs` table.

## Run locally

### Docker Compose (PostgreSQL + API)

```bash
docker compose up --build          # API on http://localhost:8000 (docs at /docs); Postgres on 127.0.0.1:5433
docker compose exec api python -m app.evaluation.run_baseline
```

Works without a `.env`. Copy [.env.example](.env.example) to `.env` to set credentials and the optional LLM variables.

### Local Python

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
export DATABASE_URL=postgresql+psycopg2://claimflow:claimflow@localhost:5432/claimflow
# ...or for a quick try:  DATABASE_URL=sqlite:///claimflow.db   (PowerShell: $env:DATABASE_URL = "sqlite:///claimflow.db")
uvicorn app.main:app --reload     # http://localhost:8000/docs
```

With no `LLM_PROVIDER` set the API behaves exactly as Phase 1. With one set, `POST /claims/{id}/process` uses RAG + LLM
for claims that need note interpretation.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + database check |
| POST / GET | `/claims`, `/claims/{id}` | Create (201/409/422), list, fetch with workflow runs |
| POST | `/claims/{id}/process` | Run the workflow (409 if already processed; 503 if the LLM is misconfigured) |
| POST / GET | `/evaluation/run`, `/evaluation/latest` | Run and fetch the Phase 1 benchmark evaluation |

### Benchmark, baseline and retrieval

```bash
cd backend
python -m app.evaluation.dataset_generator --seed 42   # benchmark + retrieval labels (byte-identical for the same seed)
python -m app.evaluation.run_baseline                  # Phase 1 baseline
python -m app.rag.ingest                               # build the vector store (downloads MiniLM once, ~80 MB)
python -m app.evaluation.run_retrieval                 # retrieval benchmark, all 4 configurations
```

### Run Phase 2 yourself

Needs a real LLM. Set the variables, check one call, then run the full evaluation (about 94 model calls; on the order of
150 to 250 thousand input tokens in total is my estimate, not a measurement). Choose the model yourself; `claude-opus-5`
is shown as an example.

```powershell
cd backend
$env:LLM_PROVIDER = "anthropic"          # or "openai" (also OPENAI_API_KEY, optional OPENAI_BASE_URL)
$env:LLM_MODEL = "claude-opus-5"         # required
$env:ANTHROPIC_API_KEY = "<your key>"    # never commit this
$env:LLM_EFFORT = "low"                  # optional, Anthropic only
$env:DATABASE_URL = "sqlite:///phase2.db"   # or your PostgreSQL URL

python -m app.evaluation.llm_smoke       # ONE call: checks credentials, model name and output format
python -m app.evaluation.run_phase2      # full run + threshold table + comparison with the Phase 1 baseline
```

Optional: `LLM_CONFIDENCE_THRESHOLD`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `RETRIEVAL_CONFIG`,
`LLM_INPUT_PRICE_PER_MTOK` and `LLM_OUTPUT_PRICE_PER_MTOK` (to estimate cost; no prices are built in).
Outputs: `evaluation_results/phase2_rag-llm.json`, `comparison_phase1_vs_phase2.json` and `.txt`.
Without credentials `run_phase2` prints these instructions and exits without producing any numbers.

### Tests

```bash
cd backend
pytest                                   # 263 tests, about 1 minute
pytest --cov=app --cov-report=term-missing   # 99% line coverage
```

Tests use in-memory SQLite, an offline hashing embedder and scripted fake LLM providers, so they need no services,
network or credentials (one test uses the real MiniLM model and skips if it is unavailable).

## Limitations

- **The Phase 2 LLM workflow has not been run against a real model.** Its results are unmeasured, and nothing about
  its expected performance should be inferred from this repository.
- **Synthetic, template-based data.** Notes come from a handful of templates and the 19 policies are short and clean;
  real notes and policies are far messier. Retrieval labels and policies were authored by the same person as the
  benchmark, so numbers are "on this benchmark".
- **Small hard categories.** 8 claims each; one or two claims of difference is noise.
- **Retrieval config chosen on the benchmark it is measured on**, and the key negation policy is retrieved for only 4/8 claims.
- **Self-reported confidence** is not calibrated; thresholds trade automation for safety but do not guarantee correctness.
- **Tests and the recorded baseline ran on SQLite;** the Docker Compose / PostgreSQL / image-build path is configured but
  was not exercised where this was built (the Docker daemon was unavailable). Developed on Python 3.11; the Dockerfile targets 3.12.
- No database migrations (`create_all`); Phase 2 adds tables rather than columns so a Phase 1 database keeps working.
  Fields such as `decision_source` live in JSON (`WorkflowRun.details`), not real columns.
- `/evaluation/run` (Phase 1 baseline) is synchronous. The Phase 2 run is a CLI, since it makes many slow model calls.
- No authentication, frontend, OCR, real EHR or payer integration. Not a clinical system.

## Roadmap

1. Run Phase 2 with a real model; fill in the results table; analyse failures (for example negated claims where `POL-AMB-002` was not retrieved).
2. Ablations: LLM without retrieval, larger context, other models and thresholds.
3. Human-in-the-loop review queue and React + TypeScript UI.
