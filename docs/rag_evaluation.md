# RAG and retrieval evaluation

Everything on this page uses synthetic data only. The policies are fictional ("Northwind Health Plan"), not
copied from any real payer. Nothing here is a clinical or coverage decision system.

## What retrieval is for

When a claim passes every deterministic rule, the only thing left to judge is whether its clinical notes
adequately and unambiguously support it. That judgement needs policy text (for example "resolved uncertainty is
not ambiguity"), so the workflow retrieves policy passages and gives them to the LLM as its only allowed evidence.
Claims decided by deterministic rules do no retrieval.

## Knowledge base

`data/policies/documents/*.json`: 19 synthetic policies, each with `policy_id`, `title`, `payer`, `category`,
`procedure_codes`, `diagnosis_prefixes`, `applies_to_all`, `effective_date` and `text` (290 to 743 characters).

| Group | Policies |
|---|---|
| Procedure coverage (9) | `POL-EM-001` (office visits), `POL-PREV-001`, `POL-IMG-001` (chest X-ray, echo), `POL-IMG-002` (knee MRI), `POL-LAB-001` (metabolic and lipid panels), `POL-PT-001`, `POL-SCR-001`, `POL-DM-001`, `POL-OB-001` |
| Special procedures | `POL-SPL-001` |
| Documentation | `POL-DOC-001` (minimum documentation), `POL-INS-001` (placeholders), `POL-DX-001` (narrative vs coded diagnosis) |
| Ambiguity handling | `POL-AMB-001` (ambiguous or unresolved notes), `POL-AMB-002` (resolved or negated hedging language) |
| Other cross-cutting | `POL-DXPX-001`, `POL-HC-001`, `POL-PRV-001`, `POL-DUP-001` |

Several coverage policies cover more than one procedure code, and the cross-cutting policies apply to every
claim, so retrieval has to choose among genuinely overlapping documents.

**Written before retrieval was run.** The 19 documents and the ground-truth labels below were authored before any
retrieval result existed and have not been edited since. Retrieval configurations were the only thing varied.
They were, however, written to correspond to the benchmark (as intended), so results are "on this benchmark".

## Chunking

`app/rag/chunking.py`: sentence-aware. Sentences are packed into a chunk until adding one more would exceed
`chunk_size` characters; the next chunk starts with the trailing sentences of the previous one, up to `overlap`
characters. A single sentence longer than `chunk_size` is kept whole. Chunk ids are `<policy_id>::<index>`.
Optionally the policy title is prefixed to the text that is *embedded* (the LLM still sees the raw chunk text).

## Retrieval strategy

1. Build the query text from the claim (config-dependent, see below).
2. Embed it (all-MiniLM-L6-v2 via ONNX, 384 dimensions; cosine similarity) and ask ChromaDB for the top 20 chunks,
   optionally with a metadata filter.
3. Group chunks by policy; a policy is ranked by its best chunk. This gives a ranked list of distinct policy ids.
4. The top 5 policies (best chunk each) form the LLM's context. Citations must come from that set.

Every retrieval is logged (`retrieval_logs` table and `WorkflowRun.details["retrieval"]`): query, config, ranked
chunk ids and scores, ranked policy ids and scores, context policy ids, and latency.

## Ground-truth relevance

`app/evaluation/policy_labels.py`, stored in `data/synthetic_claims/claims_benchmark_v1.policy_labels.jsonl`
(a sidecar file, so the Phase 1 dataset and its SHA-256 are unchanged).

Relevance is defined from the claim's **category and procedure code only**, never from retrieval output. A claim
needs a policy lookup exactly when it reaches the LLM (94 of 200 claims). Its relevant set is:

1. the procedure-coverage policy that lists its procedure code, plus
2. one cross-cutting policy by category:

| Category | Cross-cutting relevant policy |
|---|---|
| `valid_routine`, `normal_valid`, `similar_but_distinct` | `POL-DOC-001` |
| `ambiguous_notes_explicit`, `ambiguous_notes_subtle` | `POL-AMB-001` |
| `negated_ambiguity` | `POL-AMB-002` |

So every labeled claim has exactly 2 relevant policies. A test regenerates the labels from the benchmark
definition and fails if the committed file differs.

## Metrics

For a query with relevant set `R` and a ranked list of retrieved policy ids:

- **Recall@K** = `|R ∩ top-K| / |R|`, averaged over queries. Because `|R| = 2` here, **Recall@1 can never exceed 0.5**.
- **Hit@K** = 1 if `R ∩ top-K` is non-empty, else 0, averaged (the "at least one relevant policy found" view).
- **MRR** = mean of `1 / rank of the first relevant policy` (0 if none retrieved).

Also reported: per-category metrics, per-relevant-policy "found in top 5" counts, and retrieval latency.

## Retrieval experiments

Four configurations, each changing one thing relative to the previous (`app/rag/config.py`):

| Config | Chunks | Query text | Filter |
|---|---|---|---|
| `baseline` | 1200 chars, no overlap (one chunk per policy) | clinical notes | none |
| `chunking` | 400 chars, 80 overlap, title prefixed | clinical notes | none |
| `query` | as `chunking` | procedure description + notes | none |
| `filtered` | as `query` | procedure description + notes | policies for the claim's procedure + cross-cutting policies |

### Measured results (94 labeled queries, all-MiniLM-L6-v2)

Reproduce with `python -m app.evaluation.run_retrieval`. Raw output: `evaluation_results/retrieval_<config>.json`.

| Config | Recall@1 | Recall@3 | Recall@5 | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.309 | 0.527 | 0.665 | 0.617 | 0.904 | 1.000 | 0.770 |
| chunking | 0.324 | 0.484 | 0.596 | 0.649 | 0.904 | 0.989 | 0.789 |
| query | 0.500 | 0.574 | 0.697 | 1.000 | 1.000 | 1.000 | 1.000 |
| filtered | 0.500 | 0.734 | **0.947** | 1.000 | 1.000 | 1.000 | 1.000 |

How often each kind of relevant policy was found in the top 5:

| Config | Procedure policy (of 94) | `POL-DOC-001` (of 68) | `POL-AMB-001` (of 18) | `POL-AMB-002` (of 8) |
|---|---:|---:|---:|---:|
| baseline | 81 | 23 | 15 | 6 |
| chunking | 83 | 14 | 13 | 2 |
| query | 94 | 21 | 15 | 1 |
| filtered | 94 | 62 | 18 | 4 |

### What the numbers do and do not say

- **`filtered` has the best overall Recall@5 (0.947 vs 0.665 for the baseline)**, and it is the default
  (`RETRIEVAL_CONFIG=filtered`). The rule "highest overall Recall@5" was fixed before choosing, but the choice is made
  on the same benchmark it is measured on, which is mildly optimistic.
- **Smaller chunks alone made Recall@5 worse** (0.596 vs 0.665). I did not investigate why, so no explanation is claimed.
- **The gains come from the structured query and the filter, not chunking.** Putting the procedure description in the
  query makes the procedure policy always rank first (94/94).
- **MRR and Hit@K saturate at 1.0** for `query` and `filtered` because the procedure policy is relevant to every
  claim and the query names the procedure. They stop discriminating; Recall@K on the cross-cutting policies is the
  informative number.
- **Part of the filter's gain is a smaller haystack.** With the filter, candidates are about 10 policies (the
  procedure's own plus 9 cross-cutting), so the top 5 is drawn from roughly half of the candidates.
- **The most important policy for one Phase 1 failure category is the one retrieval finds least reliably.**
  `POL-AMB-002` (which explains that negated hedging is not ambiguity) is in the LLM's context for only 4 of 8
  `negated_ambiguity` claims under `filtered`, and the baseline actually retrieved it more often (6/8). Whether
  the LLM still gets those claims right without it is exactly what the Phase 2 run will show, since the retrieved
  context is stored per claim.
- Median retrieval latency was about 21 to 24 ms per query on this machine (CPU; embedding plus vector search, not broken down further).
- Only 8 claims per hard category: differences of one or two claims are not statistically meaningful.

## What is not tested

Retrieval quality on real policy documents, on real clinical text, or against a larger and noisier knowledge base.
There are 19 short, clean documents; real retrieval is harder.
