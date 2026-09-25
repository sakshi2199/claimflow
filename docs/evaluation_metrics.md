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
