"""Retrieval metrics against hand-computed values, plus checks that the ground truth comes from the benchmark."""

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.evaluation import retrieval_eval, run_retrieval
from app.evaluation.dataset_generator import DATASET_NAME, DEFAULT_SEED, generate_claims
from app.evaluation.evaluator import load_dataset
from app.evaluation.policy_labels import (
    CROSS_CUTTING_POLICIES,
    POLICY_LOOKUP_CATEGORIES,
    build_policy_labels,
    labels_path_for,
    load_policy_labels,
    relevant_policy_ids,
    serialize_labels,
)
from app.evaluation.retrieval_metrics import (
    RetrievalRecord,
    compute_retrieval_metrics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.models.evaluation_run import EvaluationRun
from app.rag.embeddings import HashingEmbedder
from app.rag.store import PolicyStore, get_client

DATASET = get_settings().synthetic_claims_dir / DATASET_NAME


def rec(n: str, category: str, relevant: list[str], retrieved: list[str], latency: float = 1.0) -> RetrievalRecord:
    return RetrievalRecord(n, category, relevant, retrieved, latency)


# Q1: relevant {A,B}, retrieved [A,X,B,Y,Z]  -> R@1=.5 R@3=1 R@5=1 | hit 1,1,1 | RR=1
# Q2: relevant {C},   retrieved [X,Y,C]      -> R@1=0  R@3=1 R@5=1 | hit 0,1,1 | RR=1/3
# Q3: relevant {D},   retrieved [X,Y,Z]      -> all 0                             | RR=0
RECORDS = [
    rec("q1", "cat1", ["A", "B"], ["A", "X", "B", "Y", "Z"], 1.0),
    rec("q2", "cat1", ["C"], ["X", "Y", "C"], 2.0),
    rec("q3", "cat2", ["D"], ["X", "Y", "Z"], 3.0),
]


def test_single_query_metric_functions() -> None:
    assert recall_at_k(["A", "B"], ["A", "X", "B"], 1) == 0.5
    assert recall_at_k(["A", "B"], ["A", "X", "B"], 3) == 1.0
    assert hit_at_k(["A", "B"], ["X", "B"], 1) == 0.0
    assert hit_at_k(["A", "B"], ["X", "B"], 2) == 1.0
    assert reciprocal_rank(["C"], ["X", "Y", "C"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["D"], ["X"]) == 0.0
    assert reciprocal_rank(["A"], []) == 0.0


def test_aggregate_recall_hit_rate_and_mrr() -> None:
    report = compute_retrieval_metrics(RECORDS)
    assert report["queries"] == 3
    assert report["recall_at_k"]["1"] == pytest.approx((0.5 + 0 + 0) / 3)
    assert report["recall_at_k"]["3"] == pytest.approx((1 + 1 + 0) / 3)
    assert report["recall_at_k"]["5"] == pytest.approx((1 + 1 + 0) / 3)
    assert report["hit_rate_at_k"]["1"] == pytest.approx(1 / 3)
    assert report["hit_rate_at_k"]["3"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report["latency_ms"] == {"mean": 2.0, "median": 2.0, "p95": 3.0}


def test_metrics_by_category_and_by_relevant_policy() -> None:
    report = compute_retrieval_metrics(RECORDS)
    assert report["by_category"]["cat1"]["queries"] == 2
    assert report["by_category"]["cat1"]["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert report["by_category"]["cat2"]["recall_at_k"]["5"] == 0.0
    assert report["by_relevant_policy"]["A"] == {"expected": 1, "found_in_top_5": 1}
    assert report["by_relevant_policy"]["D"] == {"expected": 1, "found_in_top_5": 0}


def test_recall_is_bounded_by_the_relevant_set_size() -> None:
    """With 2 relevant policies, Recall@1 can never exceed 0.5: an inherent property, not a defect."""
    report = compute_retrieval_metrics([rec("q", "c", ["A", "B"], ["A", "B"])])
    assert report["recall_at_k"]["1"] == 0.5 and report["hit_rate_at_k"]["1"] == 1.0


def test_retrieval_metrics_need_at_least_one_query() -> None:
    with pytest.raises(ValueError):
        compute_retrieval_metrics([])


# ---- ground truth comes from the benchmark definition -------------------------------------------------


def test_relevance_rules_per_category(policy_documents) -> None:
    assert relevant_policy_ids("valid_routine", "PRC-101", policy_documents) == ["POL-EM-001", "POL-DOC-001"]
    assert relevant_policy_ids("negated_ambiguity", "PRC-130", policy_documents) == ["POL-LAB-001", "POL-AMB-002"]
    assert relevant_policy_ids("ambiguous_notes_explicit", "PRC-150", policy_documents) == ["POL-PT-001", "POL-AMB-001"]
    assert relevant_policy_ids("ambiguous_notes_subtle", "PRC-160", policy_documents) == ["POL-IMG-002", "POL-AMB-001"]
    for category in ("duplicate_claim", "high_cost", "invalid_amount", "incomplete_supporting_info"):
        assert relevant_policy_ids(category, "PRC-101", policy_documents) is None  # no policy lookup needed


def test_grouped_procedures_share_a_coverage_policy(policy_documents) -> None:
    assert relevant_policy_ids("normal_valid", "PRC-120", policy_documents)[0] == "POL-IMG-001"
    assert relevant_policy_ids("normal_valid", "PRC-180", policy_documents)[0] == "POL-IMG-001"


def test_labeling_fails_loudly_when_a_policy_is_missing(policy_documents) -> None:
    with pytest.raises(ValueError, match="cannot label"):
        relevant_policy_ids("valid_routine", "PRC-210", policy_documents)  # special procedure: no coverage policy
    with pytest.raises(ValueError, match="cannot label"):
        relevant_policy_ids("negated_ambiguity", "PRC-101", [d for d in policy_documents if d.policy_id != "POL-AMB-002"])


def test_committed_labels_match_generator_and_cover_exactly_the_lookup_categories(policy_documents) -> None:
    claims = generate_claims(DEFAULT_SEED)
    expected = serialize_labels(build_policy_labels(claims, policy_documents))
    assert labels_path_for(DATASET).read_text(encoding="utf-8") == expected

    labels = load_policy_labels(labels_path_for(DATASET))
    lookup_claims = [c for c in claims if c["category"] in POLICY_LOOKUP_CATEGORIES]
    assert set(labels) == {c["claim_number"] for c in lookup_claims} and len(labels) == 94
    known = {d.policy_id for d in policy_documents}
    assert all(set(ids) <= known and 2 <= len(ids) for ids in labels.values())


def test_every_cross_cutting_policy_exists(policy_documents) -> None:
    known = {d.policy_id for d in policy_documents}
    assert {p for ids in CROSS_CUTTING_POLICIES.values() for p in ids} <= known


def test_labels_do_not_change_the_phase1_dataset() -> None:
    """The benchmark file itself is untouched, so Phase 1 and Phase 2 provably use the same claims."""
    _rows, sha256 = load_dataset(DATASET)
    assert sha256 == "96e8b319928021e7403a4ed9725cb98ee7beb5306cf035169877526d62b3d538"


# ---- retrieval evaluation harness ---------------------------------------------------------------------


def test_collect_records_covers_every_labeled_claim(retriever) -> None:
    rows, _ = load_dataset(DATASET)
    labels = load_policy_labels(labels_path_for(DATASET))
    records = retrieval_eval.collect_retrieval_records(retriever, rows, labels)
    assert len(records) == 94
    assert all(r.retrieved and r.latency_ms > 0 and r.relevant == labels[r.claim_number] for r in records)


def test_run_retrieval_evaluation_is_saved_as_an_evaluation_run(db, retriever) -> None:
    evaluation = retrieval_eval.run_retrieval_evaluation(db, DATASET, retriever, "hashing-384")
    stored = db.get(EvaluationRun, evaluation.id)
    assert stored.label == "retrieval-filtered" and stored.status == "COMPLETED" and stored.total_claims == 94
    report = stored.report
    assert report["retrieval_config"]["name"] == "filtered" and report["embedder"] == "hashing-384"
    assert set(report["retrieval"]["recall_at_k"]) == {"1", "3", "5"}
    assert report["dataset"]["sha256"] == stored.dataset_sha256


def test_retrieval_cli_evaluates_two_configs_and_saves_separate_reports(
    session_factory, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    monkeypatch.setattr(run_retrieval, "SessionLocal", session_factory)
    monkeypatch.setattr(run_retrieval, "init_db", lambda: None)
    try:
        assert run_retrieval.main(["--configs", "baseline", "filtered", "--output-dir", str(tmp_path / "out")]) == 0
    finally:
        get_settings.cache_clear()

    out = capsys.readouterr().out
    assert "baseline" in out and "filtered" in out and "Recall@5" in out and "MRR" in out
    assert (tmp_path / "out" / "retrieval_baseline.json").exists()
    assert (tmp_path / "out" / "retrieval_filtered.json").exists()
    assert (tmp_path / "out" / "retrieval_comparison.json").exists()


def test_retrieval_cli_rejects_missing_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as exit_info:
            run_retrieval.main(["--dataset", str(tmp_path / "missing.jsonl"), "--configs", "baseline"])
    finally:
        get_settings.cache_clear()
    assert exit_info.value.code == 2


def test_store_used_by_evaluation_is_isolated_per_call(tmp_path: Path, policy_documents) -> None:
    store = PolicyStore(get_client(tmp_path), HashingEmbedder())
    from app.rag.config import get_preset

    store.ingest(policy_documents, get_preset("baseline"))
    assert store.has_collection(get_preset("baseline")) and not store.has_collection(get_preset("filtered"))
