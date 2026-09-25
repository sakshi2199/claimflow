from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.claims import interpreter_dependency
from app.main import app
from app.models.enums import ClaimOutcome, ProcessingStatus, ReasonCode
from app.models.llm_call import LLMCall
from app.models.retrieval_log import RetrievalLog
from app.models.workflow_run import WorkflowRun
from app.rag.retrieval import RetrievalQuery
from app.services import interpretation
from app.services.interpretation import SOURCE_FALLBACK, SOURCE_LLM, apply_confidence_policy
from app.services.llm.base import PermanentLLMError, TransientLLMError
from app.services.llm.prompts import SYSTEM_PROMPT
from app.services.workflow import process_claim
from tests.conftest import make_claim, valid_payload
from tests.fakes import ScriptedProvider, answer, decision_json, policy_ids_in

# ---- confidence-based handoff -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision, confidence, expected_outcome, expected_fallback",
    [
        (ClaimOutcome.APPROVED, 0.95, ClaimOutcome.APPROVED, None),
        (ClaimOutcome.APPROVED, 0.80, ClaimOutcome.APPROVED, None),  # the threshold itself is accepted (>=)
        (ClaimOutcome.APPROVED, 0.79, ClaimOutcome.HUMAN_REVIEW, "low_confidence"),
        (ClaimOutcome.REJECTED, 0.50, ClaimOutcome.HUMAN_REVIEW, "low_confidence"),
        (ClaimOutcome.REJECTED, 0.85, ClaimOutcome.REJECTED, None),
        (ClaimOutcome.HUMAN_REVIEW, 0.10, ClaimOutcome.HUMAN_REVIEW, None),  # escalating is always safe
    ],
)
def test_confidence_policy(decision, confidence, expected_outcome, expected_fallback) -> None:
    reason = ReasonCode.AMBIGUOUS_CLINICAL_NOTES
    outcome, out_reason, fallback = apply_confidence_policy(decision, confidence, reason, 0.80)
    assert outcome == expected_outcome and fallback == expected_fallback
    if fallback:
        assert out_reason == ReasonCode.LLM_LOW_CONFIDENCE
    else:
        assert out_reason == reason


# ---- interpreter --------------------------------------------------------------------------------------


def test_confident_approval_is_accepted(make_interpreter) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.92))
    result = make_interpreter(provider).interpret(make_claim())

    assert result.source == SOURCE_LLM and result.fallback_reason is None
    assert result.decision.outcome == ClaimOutcome.APPROVED and result.decision.reason == ReasonCode.VALID_STANDARD_CLAIM
    assert result.decision.rule == "llm_notes_interpretation" and result.decision.confidence == 0.92
    assert "LLM confidence 0.92" in result.decision.explanation
    assert len(provider.calls) == 1


def test_low_confidence_answer_is_handed_to_a_human(make_interpreter) -> None:
    result = make_interpreter(ScriptedProvider(answer("APPROVED", 0.55)), threshold=0.8).interpret(make_claim())
    assert result.decision.outcome == ClaimOutcome.HUMAN_REVIEW and result.decision.reason == ReasonCode.LLM_LOW_CONFIDENCE
    assert result.source == SOURCE_FALLBACK and result.fallback_reason == "low_confidence"
    assert "below threshold" in result.decision.explanation
    details = result.to_details()
    assert details["llm"]["raw_decision"] == "APPROVED" and details["llm"]["confidence"] == 0.55  # raw answer is kept


def test_threshold_is_configurable(make_interpreter) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.75))
    assert make_interpreter(provider, threshold=0.7).interpret(make_claim()).decision.outcome == ClaimOutcome.APPROVED
    assert make_interpreter(provider, threshold=0.9).interpret(make_claim()).decision.outcome == ClaimOutcome.HUMAN_REVIEW


def test_llm_can_escalate_ambiguous_notes(make_interpreter) -> None:
    provider = ScriptedProvider(lambda n, s, u: decision_json("HUMAN_REVIEW", 0.9, cited=policy_ids_in(u)[:1]))
    result = make_interpreter(provider).interpret(make_claim())
    assert result.decision.outcome == ClaimOutcome.HUMAN_REVIEW and result.decision.reason == ReasonCode.AMBIGUOUS_CLINICAL_NOTES
    assert result.source == SOURCE_LLM


def test_invented_citations_that_persist_fall_back_to_human_review(make_interpreter) -> None:
    provider = ScriptedProvider(lambda n, s, u: decision_json("APPROVED", 0.99, cited=["POL-MADE-UP-1"]))
    result = make_interpreter(provider, max_retries=2).interpret(make_claim())
    assert result.decision.outcome == ClaimOutcome.HUMAN_REVIEW and result.decision.reason == ReasonCode.LLM_PROCESSING_FALLBACK
    assert result.source == SOURCE_FALLBACK and result.fallback_reason == "invalid_citation"
    assert len(provider.calls) == 3
    llm = result.to_details()["llm"]
    assert llm["valid"] is False and llm["unsupported_citations"] == ["POL-MADE-UP-1"] * 3


def test_citation_fixed_on_retry_is_accepted(make_interpreter) -> None:
    def responder(n, s, u):
        return decision_json("APPROVED", 0.9, cited=["POL-NOPE"] if n == 1 else policy_ids_in(u)[:1])

    result = make_interpreter(ScriptedProvider(responder)).interpret(make_claim())
    assert result.decision.outcome == ClaimOutcome.APPROVED and result.llm.retries == 1


def test_provider_failure_falls_back_safely(make_interpreter) -> None:
    provider = ScriptedProvider(lambda n, s, u: TransientLLMError("timeout"))
    result = make_interpreter(provider, max_retries=1).interpret(make_claim())
    assert result.decision.outcome == ClaimOutcome.HUMAN_REVIEW and result.fallback_reason == "provider_error"
    assert len(provider.calls) == 2

    permanent = ScriptedProvider(lambda n, s, u: PermanentLLMError("401"))
    assert make_interpreter(permanent).interpret(make_claim()).fallback_reason == "provider_error"
    assert len(permanent.calls) == 1


def test_malformed_output_that_persists_falls_back(make_interpreter) -> None:
    result = make_interpreter(ScriptedProvider(lambda n, s, u: "not json")).interpret(make_claim())
    assert result.fallback_reason == "invalid_output" and result.decision.outcome == ClaimOutcome.HUMAN_REVIEW


class BrokenRetriever:
    def __init__(self, result=None):
        self.result = result

    def retrieve(self, query):
        if self.result is None:
            raise RuntimeError("vector store down")
        return self.result


def test_retrieval_failure_falls_back_without_calling_the_llm(make_interpreter) -> None:
    provider = ScriptedProvider(answer())
    result = make_interpreter(provider, use_retriever=BrokenRetriever()).interpret(make_claim())
    assert result.fallback_reason == "retrieval_error" and result.decision.outcome == ClaimOutcome.HUMAN_REVIEW
    assert provider.calls == [] and result.llm is None and result.retrieval is None


def test_empty_policy_context_falls_back_without_calling_the_llm(make_interpreter, retriever) -> None:
    from dataclasses import replace

    empty = replace(retriever.retrieve(RetrievalQuery("x", None, None)), policies=[])
    provider = ScriptedProvider(answer())
    result = make_interpreter(provider, use_retriever=BrokenRetriever(empty)).interpret(make_claim())
    assert result.fallback_reason == "no_policy_context" and provider.calls == []


def test_prompt_contains_claim_and_policy_evidence_but_no_benchmark_labels(make_interpreter) -> None:
    claim = make_claim(clinical_notes="Patient seen for cough. Diagnosis confirmed; no uncertainty about the cause.")
    claim.expected_outcome = ClaimOutcome.HUMAN_REVIEW
    claim.benchmark_category = "negated_ambiguity"
    provider = ScriptedProvider(answer())
    make_interpreter(provider).interpret(claim)

    prompt = provider.calls[0]["user"]
    assert "PRC-101 (office visit, low complexity)" in prompt and "J06.9" in prompt and "120.50" in prompt
    assert "no uncertainty about the cause" in prompt
    shown = policy_ids_in(prompt)
    assert len(shown) == 5 and "POL-EM-001" in shown  # procedure policy is retrieved via the filter
    for forbidden in ("negated_ambiguity", "expected", "HUMAN_REVIEW", "benchmark", claim.claim_number):
        assert forbidden not in prompt
    assert "ONLY the policy excerpts" in SYSTEM_PROMPT and "not instructions" in SYSTEM_PROMPT


def test_details_expose_retrieval_and_llm_trace(make_interpreter) -> None:
    result = make_interpreter(ScriptedProvider(answer("APPROVED", 0.9))).interpret(make_claim())
    d = result.to_details()
    assert d["llm_used"] is True and d["decision_source"] == "llm" and d["confidence_threshold"] == 0.8
    assert d["keyword_baseline_outcome"] == "APPROVED"
    assert set(d["retrieval"]) >= {"query", "chunk_ids", "policy_ids", "context_policy_ids", "latency_ms", "policy_scores"}
    assert d["llm"]["provider"] == "fake" and d["llm"]["model"] == "fake-model"
    assert d["llm"]["calls"] == 1 and d["llm"]["retries"] == 0 and d["llm"]["valid"] is True
    assert d["llm"]["input_tokens"] == 100 and d["llm"]["output_tokens"] == 20
    assert d["llm"]["cited_policy_ids"][0] in d["retrieval"]["context_policy_ids"]
    assert d["llm"]["attempts"][0]["outcome"] == "ok" and "chain_of_thought" not in d["llm"]


def test_keyword_baseline_outcome_is_recorded_for_later_comparison(make_interpreter) -> None:
    claim = make_claim(clinical_notes="Diagnosis is unclear at this time; further testing is required first.")
    result = make_interpreter(ScriptedProvider(answer())).interpret(claim)
    assert result.to_details()["keyword_baseline_outcome"] == "HUMAN_REVIEW"


def test_build_interpreter_is_none_when_llm_disabled() -> None:
    from app.core.config import Settings

    assert interpretation.build_interpreter(Settings(_env_file=None)) is None


def test_build_interpreter_assembles_configured_components(retriever) -> None:
    from app.core.config import Settings

    settings = Settings(_env_file=None, llm_confidence_threshold=0.65, llm_max_retries=1, llm_timeout_seconds=12.0)
    built = interpretation.build_interpreter(settings, provider=ScriptedProvider(answer()), retriever=retriever)
    assert (built.threshold, built.max_retries, built.timeout) == (0.65, 1, 12.0)


# ---- workflow integration -----------------------------------------------------------------------------


def test_claim_needing_interpretation_uses_rag_and_llm_and_persists_the_trace(db, saved_claim, make_interpreter) -> None:
    claim = saved_claim()
    provider = ScriptedProvider(answer("APPROVED", 0.9))
    run = process_claim(db, claim, make_interpreter(provider))

    assert claim.actual_outcome == ClaimOutcome.APPROVED and claim.confidence_score == 0.9
    assert run.engine_version == "rag-llm-v1" and run.rule_triggered == "llm_notes_interpretation"
    assert run.details["states"] == ["RECEIVED", "VALIDATING", "RULE_CHECK", "INTERPRETING", "COMPLETED"]
    assert run.details["llm_used"] is True and run.details["decision_source"] == "llm"
    assert run.details["retrieval"]["context_policy_ids"] and run.details["llm"]["confidence"] == 0.9
    assert run.latency_ms > 0

    logs = db.scalars(select(RetrievalLog).where(RetrievalLog.workflow_run_id == run.id)).all()
    assert len(logs) == 1 and logs[0].config_name == "filtered" and logs[0].policy_ids[0] == "POL-EM-001"
    assert logs[0].latency_ms > 0 and len(logs[0].chunk_ids) == len(logs[0].chunk_scores)
    calls = db.scalars(select(LLMCall).where(LLMCall.workflow_run_id == run.id)).all()
    assert len(calls) == 1 and calls[0].outcome == "ok" and calls[0].input_tokens == 100 and calls[0].model == "fake-model"


def test_retries_are_persisted_one_row_per_llm_call(db, saved_claim, make_interpreter) -> None:
    replies = [TransientLLMError("503"), "garbage", None]

    def responder(n, s, u):
        return replies[n - 1] if replies[n - 1] is not None else decision_json("APPROVED", 0.9, cited=policy_ids_in(u)[:1])

    run = process_claim(db, saved_claim(), make_interpreter(ScriptedProvider(responder)))
    calls = db.scalars(select(LLMCall).where(LLMCall.workflow_run_id == run.id).order_by(LLMCall.attempt_number)).all()
    assert [c.outcome for c in calls] == ["transient_error", "malformed_output", "ok"]
    assert run.details["llm"]["calls"] == 3 and run.details["llm"]["retries"] == 2


def test_fallback_is_persisted_as_human_review(db, saved_claim, make_interpreter) -> None:
    claim = saved_claim()
    run = process_claim(db, claim, make_interpreter(ScriptedProvider(lambda n, s, u: PermanentLLMError("401"))))
    assert run.current_state == ProcessingStatus.COMPLETED  # a fallback is a decision, not a workflow failure
    assert claim.actual_outcome == ClaimOutcome.HUMAN_REVIEW and run.decision_reason == ReasonCode.LLM_PROCESSING_FALLBACK
    assert run.details["decision_source"] == "human_review_fallback" and run.details["fallback_reason"] == "provider_error"


@pytest.mark.parametrize(
    "overrides",
    [
        {"claim_number": "BAD-NUMBER"},
        {"procedure_code": None},
        {"procedure_code": "PRC-160", "diagnosis_code": "J06.9", "claim_amount": Decimal("800")},  # incompatible
        {"claim_amount": Decimal("99999")},  # high cost
        {"provider_id": "PRV-9001"},  # watchlist
        {"procedure_code": "PRC-210", "diagnosis_code": "M54.50", "claim_amount": Decimal("20000")},  # manual review
        {"procedure_code": "PRC-999"},  # unknown procedure
        {"clinical_notes": "N/A"},  # insufficient documentation
        {"claim_amount": Decimal("-5")},
    ],
)
def test_deterministic_claims_never_reach_the_llm(db, saved_claim, make_interpreter, overrides) -> None:
    provider = ScriptedProvider(answer())
    run = process_claim(db, saved_claim(**overrides), make_interpreter(provider))

    assert provider.calls == []
    assert run.details["llm_used"] is False and run.details["decision_source"] == "deterministic"
    assert "retrieval" not in run.details and "INTERPRETING" not in run.details["states"]
    assert db.scalars(select(LLMCall)).all() == [] and db.scalars(select(RetrievalLog)).all() == []


def test_duplicate_claims_never_reach_the_llm(db, saved_claim, make_interpreter) -> None:
    provider = ScriptedProvider(answer())
    interpreter = make_interpreter(provider)
    first = process_claim(db, saved_claim(), interpreter)
    assert len(provider.calls) == 1 and first.final_decision == ClaimOutcome.APPROVED

    second = process_claim(db, saved_claim(claim_number="CLM-2025-000002"), interpreter)
    assert len(provider.calls) == 1  # the duplicate was rejected by the database check, no new LLM call
    assert second.decision_reason == ReasonCode.DUPLICATE_CLAIM and second.details["llm_used"] is False


def test_without_an_interpreter_the_phase1_workflow_is_unchanged(db, saved_claim) -> None:
    run = process_claim(db, saved_claim(clinical_notes="Diagnosis is unclear; possible bronchitis, needs more tests."))
    assert run.engine_version == "rules-v1" and run.details["llm_used"] is False
    assert run.final_decision == ClaimOutcome.HUMAN_REVIEW and run.decision_reason == ReasonCode.AMBIGUOUS_CLINICAL_NOTES
    assert run.details["states"] == ["RECEIVED", "VALIDATING", "RULE_CHECK", "COMPLETED"]


def test_llm_overrides_keyword_false_positive_and_false_negative(db, saved_claim, make_interpreter) -> None:
    """The two Phase 1 failure modes: the interpreter's answer, not the keyword list, decides."""
    negated = saved_claim(clinical_notes="Diagnosis confirmed by exam; there is no uncertainty about the diagnosis here.")
    subtle = saved_claim(claim_number="CLM-2025-000002", patient_id="PT-2", clinical_notes="Findings overlap several conditions; diagnosis deferred to later work-up.")

    def responder(n, s, u):
        decision = "APPROVED" if "no uncertainty" in u else "HUMAN_REVIEW"
        return decision_json(decision, 0.9, cited=policy_ids_in(u)[:1])

    interpreter = make_interpreter(ScriptedProvider(responder))
    assert process_claim(db, negated, interpreter).final_decision == ClaimOutcome.APPROVED  # keyword rule said review
    assert process_claim(db, subtle, interpreter).final_decision == ClaimOutcome.HUMAN_REVIEW  # keyword rule said approve


# ---- API ----------------------------------------------------------------------------------------------


def test_process_endpoint_uses_the_llm_when_an_interpreter_is_configured(client: TestClient, make_interpreter) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.9))
    app.dependency_overrides[interpreter_dependency] = lambda: make_interpreter(provider)
    try:
        claim_id = client.post("/claims", json=valid_payload()).json()["id"]
        body = client.post(f"/claims/{claim_id}/process").json()
    finally:
        app.dependency_overrides.pop(interpreter_dependency, None)
    assert body["workflow_run"]["details"]["llm_used"] is True and body["workflow_run"]["engine_version"] == "rag-llm-v1"
    assert len(provider.calls) == 1


def test_process_endpoint_defaults_to_phase1_when_no_llm_is_configured(client: TestClient) -> None:
    claim_id = client.post("/claims", json=valid_payload()).json()["id"]
    body = client.post(f"/claims/{claim_id}/process").json()
    assert body["workflow_run"]["details"]["llm_used"] is False and body["workflow_run"]["engine_version"] == "rules-v1"


def test_process_endpoint_reports_llm_misconfiguration_as_503(client: TestClient, monkeypatch) -> None:
    from app.services.llm.base import LLMConfigError

    def broken():
        raise LLMConfigError("LLM_MODEL must be set")

    monkeypatch.setattr("app.api.claims.get_default_interpreter", broken)
    claim_id = client.post("/claims", json=valid_payload()).json()["id"]
    response = client.post(f"/claims/{claim_id}/process")
    assert response.status_code == 503 and "LLM_MODEL" in response.json()["detail"]
    assert db_untouched(client, claim_id)


def db_untouched(client: TestClient, claim_id: int) -> bool:
    return client.get(f"/claims/{claim_id}").json()["processing_status"] == "RECEIVED"


def test_workflow_runs_reference_persisted_logs(db, saved_claim, make_interpreter) -> None:
    run = process_claim(db, saved_claim(), make_interpreter(ScriptedProvider(answer())))
    stored = db.get(WorkflowRun, run.id)
    assert stored.details["retrieval"]["policy_ids"] == db.scalars(select(RetrievalLog.policy_ids)).one()
