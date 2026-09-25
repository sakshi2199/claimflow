# Evaluation metrics: definitions

Implemented in `backend/app/evaluation/metrics.py` as pure functions and verified against hand-computed
examples in `backend/tests/test_metrics.py`.

Notation: `N` = claims evaluated. For each claim there is an **expected** outcome (ground truth) and an
**actual** outcome (what the workflow decided; none if the workflow failed).

| Metric | Formula | Notes |
|---|---|---|
| Decision accuracy | `#(actual == expected) / N` | A workflow failure has no outcome and counts as incorrect. |
| Automation rate | `#(actual in {APPROVED, REJECTED}) / N` | Share of claims decided without a human. Says nothing about correctness. |
| Human-review rate | `#(actual == HUMAN_REVIEW) / N` | Automation rate + human-review rate = 1 when no workflow fails. |
| Automated-decision accuracy | `#(actual in {A,R} and actual == expected) / #(actual in {A,R})` | Of the claims the system decided alone, how many were right. Pairs with automation rate: high automation is only good if this stays high. |
| False approvals | `#(actual == APPROVED and expected != APPROVED)` | The costly error in claims operations: paying something that should not be paid or should have been reviewed. |
| Workflow success rate | `#(workflow COMPLETED) / N` | `COMPLETED` = reached a decision, whatever it was. A correct rejection is a success. |
| Workflow failure rate | `#(workflow FAILED) / N` | Unexpected exceptions only. |
| Reason-code accuracy | `#(actual_reason == expected_reason) / N` | Whether the *why* is right, not just the outcome. |
| Precision (class c) | `TP / (TP + FP)` | `TP`: expected c and actual c. `FP`: expected not c but actual c. |
| Recall (class c) | `TP / (TP + FN)` | `FN`: expected c but actual not c (includes workflow failures). |
| F1 (class c) | `2PR / (P + R)` | |
| Accuracy by category | per `category`: `#correct / #claims in category` | Shows *where* errors are. |
| Latency mean / median | arithmetic mean / `statistics.median` | Per-claim workflow latency in ms. |
| Latency P95 | nearest-rank: sorted value at position `ceil(0.95 * N)` | No interpolation; always an observed value. |

**Undefined values are `null`, not 0.** If a class is never predicted, its precision is `0/0`. It is reported
as `n/a` (null in JSON) so "undefined" is never confused with "bad".

## Confusion matrix

Rows are the expected outcome, columns the predicted outcome. The columns are `APPROVED`, `REJECTED`,
`HUMAN_REVIEW`, plus `NO_DECISION` for claims where the workflow failed, so every claim is accounted for and
the matrix always sums to `N`.

## Latency: what is and is not measured

Measured from workflow start to the decision being written to the session (validation, the duplicate query
and the `WorkflowRun` insert), **excluding** the final commit. Numbers depend on the database: the committed
baseline was measured on in-process SQLite, so it is much lower than a networked PostgreSQL would give. Treat
latency as a way to compare variants **on the same setup** (e.g. how much an LLM call adds), not as an absolute
production figure.

## Comparing runs

Every run is stored in `evaluation_runs` with `label`, `engine_version`, `dataset_sha256` and the whole
report. Two runs are comparable only if their `dataset_sha256` matches.

---

# Phase 2 metrics

Implemented in `evaluation/phase2_metrics.py`, `retrieval_metrics.py`, `thresholds.py` and `comparison.py`.
Everything is computed from the stored `WorkflowRun.details` of an actual run; no metric calls a model.
"LLM claims" means claims for which the LLM path was taken.

| Metric | Formula |
|---|---|
| LLM invocation rate | `#(LLM claims) / N` |
| Avg LLM calls per claim | `total LLM calls / N` (retries count as calls); also reported per LLM claim |
| Retry rate | `#(LLM claims with at least one retry) / #(LLM claims)` |
| LLM failure rate | `#(LLM claims with no accepted decision after all retries) / #(LLM claims)` |
| Provider error rate | as above, restricted to transient or permanent provider errors |
| Structured-output failure rate | per call: `#(calls whose output failed schema validation) / #(calls)`; per claim: share of LLM claims that ended with malformed output |
| Unsupported citation rate | per response: `#(parseable responses citing an id outside the retrieved context) / #(parseable responses)`; per id: `#(unsupported cited ids) / #(cited ids)`. Measured on every attempt, before retries can hide it |
| Avg retrieval latency | mean of `retrieval.latency_ms` over LLM claims |
| Avg LLM latency | mean over LLM claims of the sum of that claim's call latencies (retries included) |
| Total latency (mean / median / P95) | as in Phase 1: workflow start to decision, now including retrieval and LLM time |
| Tokens | sums of provider-reported input and output tokens |
| Estimated cost | `(input_tokens * input_price + output_tokens * output_price) / 1e6`, **only** if both `LLM_INPUT_PRICE_PER_MTOK` and `LLM_OUTPUT_PRICE_PER_MTOK` are set. No price is built in, because prices change |
| False rejection | `#(actual == REJECTED and expected != REJECTED)`, from the confusion matrix |
| Fixed / broken by the LLM path | on LLM claims: keyword-baseline outcome wrong and final outcome right / keyword right and final wrong |
| Retrieval Recall@K, Hit@K, MRR | defined in [rag_evaluation.md](rag_evaluation.md); in a Phase 2 run they are computed from the retrievals the workflow actually made |

## Confidence threshold experiment

The confidence is the model's self-reported estimate, **not a calibrated probability**. For each threshold `t`, using
the stored raw answer of every LLM claim: an `APPROVED`/`REJECTED` answer with `confidence >= t` is accepted, otherwise
the claim becomes `HUMAN_REVIEW`; a `HUMAN_REVIEW` answer is always kept; claims where the LLM failed and
deterministic claims are unchanged. Accuracy, automation rate, human-review rate, false approvals, false rejections
and automated-decision accuracy are then recomputed (with the same formulas as above). Replaying the run's own
threshold reproduces the run exactly (tested).

## Phase 1 vs Phase 2 comparison

`run_phase2` loads `evaluation_results/baseline_rules-v1.json` and refuses to compare unless the dataset SHA-256 and
the number of claims are identical. Phase 1 latency was measured on SQLite with no network; Phase 2 latency includes
model calls, so the two latency columns are not like for like.

