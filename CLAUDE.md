# ClaimFlow: project handoff

Synthetic healthcare claims triage portfolio project. **All data is synthetic. Not a clinical decision system. No HIPAA or compliance claims.** Never invent metrics: every reported number must come from an actual run.

## Status (end of 2026-09-25)

- **Phase 1** (deterministic rules + evaluation framework): complete and measured.
- **Phase 2** (ChromaDB RAG + LLM note interpretation): built and tested with mocks. **Retrieval is measured. The end-to-end LLM workflow has NOT been run**, because no LLM credentials existed in the build environment. Phase 2 accuracy, false approvals, threshold table, token usage, etc. are all still unknown.
- Tests: 263 pass, 99% line coverage (none call a real model). No git commits have been made (`git init` only).

## Architecture

`RECEIVED -> VALIDATING -> RULE_CHECK -> [INTERPRETING] -> COMPLETED` (or `FAILED` on an unexpected exception).

1. `services/validation.py`, `services/rules.py`: pure functions returning a `Decision`. Deterministic rules decide everything they can (duplicates, malformed/missing fields, amounts, diagnosis/procedure compatibility, watchlist, high cost, special procedures, missing notes).
2. Only claims that pass every deterministic rule reach the notes-interpretation step (94 of 200 benchmark claims, 47%).
   - Phase 1 / no LLM configured: keyword hedging check decides.
   - Phase 2 / LLM configured: `services/interpretation.py` retrieves policies (`app/rag/`), calls the LLM (`services/llm/`), validates structured JSON and that every cited policy id was retrieved, then applies the confidence threshold.
3. Any failure (provider error, malformed output, invalid citations, retrieval error) becomes `HUMAN_REVIEW`. Retries are bounded (`1 + LLM_MAX_RETRIES` calls). An LLM `HUMAN_REVIEW` answer is always accepted; APPROVED/REJECTED must clear the threshold.
4. `services/workflow.py` is the only place decisions are persisted. Tables: `claims`, `workflow_runs` (`details` JSON holds llm_used, decision_source, retrieval, llm trace), `evaluation_runs`, `retrieval_logs`, `llm_calls`. Tables come from `create_all` (no migrations); Phase 2 added tables, not columns.
5. Evaluation runs on the **same** committed benchmark for both phases; comparison code refuses to run if dataset SHA-256 or claim count differ.

Stack: Python (built on 3.11; Dockerfile targets 3.12), FastAPI, SQLAlchemy 2, Pydantic 2, PostgreSQL, ChromaDB, all-MiniLM-L6-v2 (ONNX), pytest, Docker Compose. Docs: `docs/architecture.md`, `docs/synthetic_data.md`, `docs/evaluation_metrics.md`, `docs/rag_evaluation.md`.

## Phase 1 baseline (measured, `evaluation_results/baseline_rules-v1.json`)

200-claim benchmark (`data/synthetic_claims/claims_benchmark_v1.jsonl`, seed 42, SHA-256 `96e8b319…`).

| Metric | Value |
|---|---|
| Decision accuracy | 92.0% (184/200) |
| Automation rate / human-review rate | 75.0% / 25.0% |
| Automated-decision accuracy | 94.7% |
| False approvals | 8 |
| Workflow success | 100% |
| Latency mean / median / P95 | 1.01 / 0.89 / 1.68 ms (in-process SQLite) |
| Precision / recall | APPROVED 89.5/89.5, REJECTED 100/100, HUMAN_REVIEW 84/84 |

All 16 errors are in two deliberately hard categories: `ambiguous_notes_subtle` (0/8, the 8 false approvals) and `negated_ambiguity` (0/8). Other categories are 100% by construction (same author wrote benchmark and rules).

## Phase 2 retrieval (measured, `evaluation_results/retrieval_*.json`)

94 labeled queries, real MiniLM embeddings, 19 synthetic policies in `data/policies/documents/`. Ground truth (procedure policy + one cross-cutting policy per category) is defined in `evaluation/policy_labels.py` and was written before retrieval was run.

| Config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| baseline | 0.309 | 0.527 | 0.665 | 0.770 |
| chunking | 0.324 | 0.484 | 0.596 | 0.789 |
| query | 0.500 | 0.574 | 0.697 | 1.000 |
| **filtered** (default `RETRIEVAL_CONFIG`) | 0.500 | 0.734 | **0.947** | 1.000 |

Caveats: Recall@1 is capped at 0.5 (2 relevant policies per claim); MRR saturates; chunking alone made Recall@5 worse (cause not investigated); the key negation policy `POL-AMB-002` is in the top 5 for only 4/8 `negated_ambiguity` claims under `filtered` (baseline: 6/8).

## Known limitations

- **No real LLM run yet.** Provider adapters (Anthropic, OpenAI-compatible) are tested only against fake SDK clients; the `LLM_EFFORT` parameter and real error behaviour are unverified against a live API.
- Docker image build, Compose startup and PostgreSQL were never exercised (Docker daemon was down); all tests and recorded results used SQLite. Compose maps Postgres to `127.0.0.1:5433` (user-edited).
- Synthetic template-based data; policies, labels and benchmark share one author, so results are "on this benchmark". Hard categories have only 8 claims each. Retrieval config was chosen on the benchmark it is measured on.
- LLM confidence is self-reported, not calibrated. Phase 1 latency (SQLite, no network) is not comparable to Phase 2 latency.
- No migrations, auth, frontend, OCR, or real integrations. A git-ignored `evaluation_results/baseline-deterministic_run1_*.json` exists that was not created by the assistant (likely a user run); harmless.
- Windows/Git Bash: heredocs containing single quotes break the shell tool; write such files with the file-write tool.

## Important files

| Purpose | Path |
|---|---|
| Workflow orchestration and persistence | `backend/app/services/workflow.py` |
| LLM + retrieval + confidence handoff | `backend/app/services/interpretation.py` |
| Structured output, grounding, retries | `backend/app/services/llm/structured.py`, `schemas.py`, `prompts.py` |
| Deterministic rules | `backend/app/services/rules.py`, `validation.py` |
| RAG | `backend/app/rag/` (`retrieval.py`, `store.py`, `chunking.py`, `config.py` presets) |
| Benchmark and labels | `backend/app/evaluation/dataset_generator.py`, `policy_labels.py` |
| Metrics and comparison | `backend/app/evaluation/metrics.py`, `phase2_metrics.py`, `thresholds.py`, `comparison.py` |
| CLIs | `evaluation/run_baseline.py`, `run_retrieval.py`, `llm_smoke.py`, `run_phase2.py`; `rag/ingest.py` |
| Config / env vars | `backend/app/core/config.py`, `.env.example` |
| Test doubles | `backend/tests/fakes.py`, `backend/tests/conftest.py` |

## Exact next steps for resuming

1. Provide LLM credentials, then run the one-call smoke check (`llm_smoke`). Fix any provider/parameter issue it reveals (for example rejected `effort` value or model id).
2. Run the full Phase 2 evaluation (`run_phase2`). It writes `evaluation_results/phase2_rag-llm.json` and `comparison_phase1_vs_phase2.{json,txt}`.
3. Fill the README "Phase 2 results" table (currently "not yet measured") from those files only. Update this file's status and add the Phase 2 metrics.
4. Analyse results honestly: accuracy and false approvals vs the Phase 1 baseline; the two hard categories; the threshold table (0.60 to 0.90); invocation rate, retries, citation failures, tokens. Check negated claims where `POL-AMB-002` was not retrieved (context is stored in `workflow_runs.details.retrieval`).
5. Optional follow-ups: ablations (LLM without retrieval, larger `context_policies`, other models/thresholds); exercise Docker/PostgreSQL; add Alembic; make the first git commit; then human-in-the-loop queue and React UI (not started).

## Commands to run next (PowerShell, from `claimflow\backend`)

```powershell
.venv\Scripts\activate
pytest --cov=app --cov-report=term-missing      # expect 263 passed
python -m app.evaluation.run_retrieval          # optional: reproduces retrieval table

$env:LLM_PROVIDER = "anthropic"                 # or "openai" (+ OPENAI_API_KEY, optional OPENAI_BASE_URL)
$env:LLM_MODEL = "<model id>"                   # your choice, e.g. claude-opus-5
$env:ANTHROPIC_API_KEY = "<key>"                # never commit
$env:LLM_EFFORT = "low"                         # optional, Anthropic only
$env:DATABASE_URL = "sqlite:///phase2.db"       # or PostgreSQL URL

python -m app.evaluation.llm_smoke              # ONE call
python -m app.evaluation.run_phase2             # ~94 calls; needs the network and costs money
```

Optional env: `LLM_CONFIDENCE_THRESHOLD` (default 0.80), `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `RETRIEVAL_CONFIG`, `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` (cost estimated only if both set). Docker path: `docker compose up --build`.
