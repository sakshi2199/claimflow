import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag import ingest as ingest_cli
from app.rag.chunking import chunk_policy, chunk_text, split_sentences
from app.rag.config import PRESETS, get_preset
from app.rag.documents import PolicyDocumentError, load_policy_documents
from app.rag.embeddings import HashingEmbedder, get_embedder
from app.rag.retrieval import RetrievalQuery, Retriever, build_query_text, build_retriever
from app.rag.store import PolicyStore, get_client, procedure_key
from app.services.policies import load_policies

# ---- policy documents ---------------------------------------------------------------------------------


def test_policy_knowledge_base_has_expected_shape(policy_documents) -> None:
    assert 10 <= len(policy_documents) <= 20
    assert len({d.policy_id for d in policy_documents}) == len(policy_documents)
    for doc in policy_documents:
        assert doc.policy_id.startswith("POL-") and doc.title and doc.category and doc.effective_date
        assert doc.text.strip() and "fictional" in doc.payer.lower()
        if doc.applies_to_all:
            assert doc.procedure_codes == ()


def test_procedure_coverage_policies_only_reference_catalog_procedures(policy_documents) -> None:
    catalog = set(load_policies().procedures)
    for doc in policy_documents:
        assert set(doc.procedure_codes) <= catalog, doc.policy_id
    covered = {code for d in policy_documents for code in d.procedure_codes}
    assert covered == catalog  # every catalog procedure has a policy


def test_policy_categories_cover_the_required_areas(policy_documents) -> None:
    categories = {d.category for d in policy_documents}
    assert {
        "procedure_coverage", "documentation", "ambiguity_handling", "diagnosis_procedure",
        "high_cost_review", "special_procedures", "provider_review",
    } <= categories  # fmt: skip


def _write_policy(directory: Path, **overrides) -> None:
    doc = {
        "policy_id": "POL-X-001", "title": "T", "payer": "Fictional", "category": "c", "procedure_codes": [],
        "diagnosis_prefixes": [], "applies_to_all": True, "effective_date": "2024-01-01", "text": "Some text.",
    }  # fmt: skip
    doc.update(overrides)
    (directory / f"{doc['policy_id']}.json").write_text(json.dumps(doc))


def test_loader_rejects_bad_policy_files(tmp_path: Path) -> None:
    with pytest.raises(PolicyDocumentError, match="no policy documents"):
        load_policy_documents(tmp_path)

    _write_policy(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({"policy_id": "POL-BAD"}))
    with pytest.raises(PolicyDocumentError, match="missing fields"):
        load_policy_documents(tmp_path)
    (tmp_path / "bad.json").unlink()

    _write_policy(tmp_path, policy_id="POL-X-002", text="   ")
    with pytest.raises(PolicyDocumentError, match="empty text"):
        load_policy_documents(tmp_path)
    (tmp_path / "POL-X-002.json").unlink()

    (tmp_path / "copy.json").write_text((tmp_path / "POL-X-001.json").read_text())
    with pytest.raises(PolicyDocumentError, match="duplicate"):
        load_policy_documents(tmp_path)


# ---- chunking -----------------------------------------------------------------------------------------

TEXT = "Alpha one is here. Beta two is here. Gamma three is here. Delta four is here. Epsilon five is here."


def test_split_sentences_normalizes_whitespace() -> None:
    assert split_sentences("One.  Two?\n\nThree!") == ["One.", "Two?", "Three!"]


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text(TEXT, 1000, 0) == [TEXT]


def test_chunks_respect_size_and_cover_all_sentences_without_overlap() -> None:
    chunks = chunk_text(TEXT, 45, 0)
    assert len(chunks) > 1
    assert all(len(c) <= 45 for c in chunks)
    assert [s for c in chunks for s in split_sentences(c)] == split_sentences(TEXT)


def test_overlap_repeats_trailing_sentence_in_next_chunk() -> None:
    chunks = chunk_text(TEXT, 45, 25)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert split_sentences(previous)[-1] == split_sentences(following)[0]
    covered = {s for c in chunks for s in split_sentences(c)}
    assert covered == set(split_sentences(TEXT))


def test_oversized_single_sentence_is_kept_whole() -> None:
    long_sentence = "word " * 30 + "end."
    assert chunk_text(long_sentence, 20, 0) == [long_sentence.strip()]


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (50, -1), (50, 50), (50, 80)])
def test_invalid_chunking_parameters_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text(TEXT, size, overlap)


def test_chunk_policy_builds_ids_and_optional_title_prefix(policy_documents) -> None:
    doc = next(d for d in policy_documents if d.policy_id == "POL-AMB-001")
    plain = chunk_policy(doc, 400, 80, include_title=False)
    titled = chunk_policy(doc, 400, 80, include_title=True)
    assert [c.chunk_id for c in plain] == [f"POL-AMB-001::{i}" for i in range(len(plain))]
    assert len(plain) >= 2
    assert all(c.embed_text == c.text for c in plain)
    assert all(c.embed_text.startswith(doc.title) and c.text == p.text for c, p in zip(titled, plain, strict=True))


def test_presets_form_the_documented_ablation() -> None:
    assert list(PRESETS) == ["baseline", "chunking", "query", "filtered"]
    assert not PRESETS["baseline"].metadata_filter and PRESETS["filtered"].metadata_filter
    assert PRESETS["baseline"].chunk_size > PRESETS["chunking"].chunk_size
    with pytest.raises(ValueError, match="unknown retrieval config"):
        get_preset("nope")


# ---- embeddings ---------------------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    a, b = embedder.embed(["physical therapy session", "physical therapy session"])
    assert a == b and len(a) == 384
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert embedder.embed([""])[0] == [0.0] * 384  # empty text must not divide by zero


def test_get_embedder_selects_by_name() -> None:
    assert get_embedder("hashing").name == "hashing-384"
    assert get_embedder("minilm").name.startswith("all-MiniLM")
    with pytest.raises(ValueError):
        get_embedder("word2vec")


# ---- ingestion ----------------------------------------------------------------------------------------


def test_ingest_counts_and_is_idempotent(tmp_path: Path, policy_documents) -> None:
    store = PolicyStore(get_client(tmp_path), HashingEmbedder())
    config = get_preset("chunking")
    assert not store.has_collection(config)

    first = store.ingest(policy_documents, config)
    second = store.ingest(policy_documents, config)  # replaces, must not duplicate
    assert first.policies == len(policy_documents) and first.chunks == second.chunks > len(policy_documents)
    assert store.client.get_collection(store.collection_name(config)).count() == first.chunks


def test_ingest_stores_metadata_for_filtering(hashing_store) -> None:
    collection = hashing_store.client.get_collection(hashing_store.collection_name(get_preset("filtered")))
    metadata = collection.get(ids=["POL-PT-001::0"], include=["metadatas"])["metadatas"][0]
    assert metadata["policy_id"] == "POL-PT-001" and metadata["applies_to_all"] is False
    assert metadata[procedure_key("PRC-150")] is True and metadata["category"] == "procedure_coverage"
    general = collection.get(ids=["POL-DOC-001::0"], include=["metadatas"])["metadatas"][0]
    assert general["applies_to_all"] is True


def test_each_config_gets_its_own_collection(hashing_store) -> None:
    names = {c.name for c in hashing_store.client.list_collections()}
    assert {f"policies-{name}" for name in PRESETS} <= names


# ---- retrieval ----------------------------------------------------------------------------------------

PT_NOTES = "Physical therapy session performed for knee pain with activity; functional limitation documented."
PT_QUERY = RetrievalQuery(PT_NOTES, "PRC-150", "physical therapy session")


def test_retrieval_returns_ranked_deduplicated_policies_with_full_trace(retriever) -> None:
    result = retriever.retrieve(PT_QUERY)

    assert result.policy_ids[0] == "POL-PT-001"
    assert len(result.policy_ids) == len(set(result.policy_ids))  # one entry per policy
    assert [p.rank for p in result.policies] == list(range(1, len(result.policies) + 1))
    scores = [p.score for p in result.policies]
    assert scores == sorted(scores, reverse=True)
    assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))
    assert all("::" in c.chunk_id for c in result.chunks) and result.latency_ms > 0

    log = result.to_log()
    assert set(log) == {"config", "query", "chunk_ids", "chunk_scores", "policy_ids", "policy_scores", "context_policy_ids", "latency_ms"}
    assert log["query"] == result.query and log["config"] == "filtered"
    assert len(log["chunk_ids"]) == len(log["chunk_scores"])


def test_context_is_limited_to_the_configured_number_of_policies(retriever) -> None:
    result = retriever.retrieve(PT_QUERY)
    assert len(result.context) == retriever.config.context_policies == 5
    assert [p.policy_id for p in result.context] == result.policy_ids[:5]


def test_query_construction_depends_on_config() -> None:
    assert build_query_text(PRESETS["baseline"], PT_QUERY) == PT_NOTES
    structured = build_query_text(PRESETS["query"], PT_QUERY)
    assert structured.startswith("physical therapy session.") and PT_NOTES in structured
    no_description = RetrievalQuery(PT_NOTES, "PRC-150", None)
    assert build_query_text(PRESETS["query"], no_description) == PT_NOTES


def test_metadata_filter_limits_candidates_to_procedure_and_general_policies(hashing_store, policy_documents) -> None:
    filtered = Retriever(hashing_store, get_preset("filtered"), policy_documents).retrieve(PT_QUERY)
    general = {d.policy_id for d in policy_documents if d.applies_to_all}
    assert set(filtered.policy_ids) <= general | {"POL-PT-001"}

    unfiltered = Retriever(hashing_store, get_preset("query"), policy_documents).retrieve(PT_QUERY)
    assert set(unfiltered.policy_ids) - (general | {"POL-PT-001"})  # other procedures' policies do appear


def test_baseline_config_retrieves_from_large_chunks(hashing_store, policy_documents) -> None:
    result = Retriever(hashing_store, get_preset("baseline"), policy_documents).retrieve(PT_QUERY)
    assert result.policy_ids[0] == "POL-PT-001" and all(c.chunk_id.endswith("::0") for c in result.chunks[:5])


def test_ensure_ready_ingests_missing_collection(tmp_path: Path, policy_documents) -> None:
    store = PolicyStore(get_client(tmp_path), HashingEmbedder())
    retriever = Retriever(store, get_preset("baseline"), policy_documents)
    assert not store.has_collection(retriever.config)
    retriever.ensure_ready()
    assert store.has_collection(retriever.config)
    retriever.ensure_ready()  # second call is a no-op


def test_build_retriever_from_settings(tmp_path: Path) -> None:
    settings = Settings(chroma_dir=tmp_path, embedder="hashing", retrieval_config="filtered", _env_file=None)
    retriever = build_retriever(settings)
    assert retriever.config.name == "filtered"
    assert retriever.retrieve(PT_QUERY).policy_ids[0] == "POL-PT-001"


def test_ingest_cli_builds_requested_configs(tmp_path: Path, monkeypatch, capsys) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDER", "hashing")
    get_settings.cache_clear()
    try:
        assert ingest_cli.main(["--config", "baseline"]) == 0
        assert ingest_cli.main([]) == 0
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "baseline" in out and "filtered" in out
    store = PolicyStore(get_client(tmp_path), HashingEmbedder())
    assert all(store.has_collection(c) for c in PRESETS.values())


def test_real_minilm_embedder_is_semantic_when_the_model_is_available() -> None:
    """Uses the real ONNX model; skipped (not failed) if it cannot be loaded, e.g. offline with no cached copy."""
    embedder = get_embedder("minilm")
    try:
        vectors = embedder.embed(["physical therapy for a knee injury", "rehabilitation exercises after a knee sprain", "quarterly tax filing deadline"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MiniLM model unavailable: {exc}")

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))

    assert len(vectors[0]) == 384
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2]) + 0.2  # paraphrase is much closer than unrelated text
