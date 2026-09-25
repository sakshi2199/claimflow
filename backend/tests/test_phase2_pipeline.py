"""The whole 200-claim benchmark through the Phase 2 workflow, with a scripted (fake) LLM.

These tests verify pipeline mechanics and safety properties (what reaches the LLM, what happens when it
fails, that thresholds replay exactly). They say nothing about how good a real model is: the fake answers
are canned. No result here may be quoted as a Phase 2 metric.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.evaluation import llm_smoke, run_phase2
from app.evaluation.dataset_generator import DATASET_NAME
from app.evaluation.evaluator import load_dataset, run_evaluation
from app.evaluation.policy_labels import POLICY_LOOKUP_CATEGORIES
from app.models.llm_call import LLMCall
from app.models.retrieval_log import RetrievalLog
from app.services.llm.base import LLMConfigError, TransientLLMError
from tests.fakes import ScriptedProvider, answer, decision_json, policy_ids_in

DATASET = get_settings().synthetic_claims_dir / DATASET_NAME
BASELINE = get_settings().evaluation_output_dir / "baseline_rules-v1.json"
CATEGORY_TOTALS = Counter(row["category"] for row in load_dataset(DATASET)[0])
LOOKUP_CLAIMS = sum(n for c, n in CATEGORY_TOTALS.items() if c in POLICY_LOOKUP_CATEGORIES)


def evaluate(db, interpreter):
    return run_evaluation(db, DATASET, "test-run", interpreter).report


def test_only_claims_needing_interpretation_reach_the_llm(db, make_interpreter) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.9))
    report = evaluate(db, make_interpreter(provider))

    assert LOOKUP_CLAIMS == 94
    assert len(provider.calls) == LOOKUP_CLAIMS  # exactly one call per interpretation-bound claim, none for the other 106
    inv = report["phase2"]["llm_invocation"]
    assert inv["claims_with_llm"] == LOOKUP_CLAIMS and inv["deterministic_decisions"] == 200 - LOOKUP_CLAIMS
    assert inv["invocation_rate"] == pytest.approx(LOOKUP_CLAIMS / 200) and inv["invocation_rate"] < 0.5
    assert report["phase2"]["llm_calls"]["avg_calls_per_claim"] == pytest.approx(LOOKUP_CLAIMS / 200)
    assert db.scalar(select(func.count()).select_from(LLMCall)) == LOOKUP_CLAIMS
    assert db.scalar(select(func.count()).select_from(RetrievalLog)) == LOOKUP_CLAIMS


def test_deterministic_categories_are_unaffected_by_phase_2(db, make_interpreter) -> None:
    """Adding RAG + LLM must not degrade anything the deterministic rules already decided correctly."""
    report = evaluate(db, make_interpreter(ScriptedProvider(answer("APPROVED", 0.9))))
    for category, entry in report["by_category"].items():
        if category not in POLICY_LOOKUP_CATEGORIES:
            assert entry["accuracy"] == 1.0, category
    assert report["summary"]["workflow_failure_rate"] == 0.0


def test_canned_approve_everything_answers_produce_the_expected_mechanical_result(db, make_interpreter) -> None:
    """A model that approves everything with high confidence is wrong on exactly the 18 claims that should be
    escalated (explicit + subtle ambiguity) and right on the 8 negated ones the keyword rule got wrong."""
    report = evaluate(db, make_interpreter(ScriptedProvider(answer("APPROVED", 0.9))))
    wrong = CATEGORY_TOTALS["ambiguous_notes_explicit"] + CATEGORY_TOTALS["ambiguous_notes_subtle"]
    assert report["summary"]["false_approval_count"] == wrong == 18
    assert report["summary"]["correct_decisions"] == 200 - wrong
    assert report["by_category"]["negated_ambiguity"]["accuracy"] == 1.0
    assert report["by_category"]["ambiguous_notes_subtle"]["accuracy"] == 0.0


def test_report_has_all_phase2_sections_and_records_provenance(db, make_interpreter) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.9), input_tokens=100, output_tokens=20)
    report = evaluate(db, make_interpreter(provider))

    assert report["engine_version"] == "rag-llm-v1" and report["llm"]["model"] == "fake-model"
    assert report["retrieval_config"]["name"] == "filtered"
    p = report["phase2"]
    assert p["retrieval"]["queries"] == LOOKUP_CLAIMS  # ground-truth labels were found next to the dataset
    assert set(p["retrieval"]["recall_at_k"]) == {"1", "3", "5"}
    assert p["tokens"]["input_tokens"] == 100 * LOOKUP_CLAIMS and p["tokens"]["total_tokens"] == 120 * LOOKUP_CLAIMS
    assert p["latency_ms"]["avg_retrieval"] > 0 and p["latency_ms"]["avg_llm_per_llm_claim"] >= 0
    assert [row["threshold"] for row in report["threshold_analysis"]] == [0.6, 0.7, 0.8, 0.9]
    assert report["dataset"]["sha256"] == load_dataset(DATASET)[1]


def test_threshold_replay_reproduces_the_actual_run(db, make_interpreter) -> None:
    """Re-applying the run's own threshold offline must give exactly the outcomes the live workflow produced."""
    confidences = [0.55, 0.75, 0.85, 0.95]

    def responder(n, system, user):
        decision = "APPROVED" if n % 3 else "HUMAN_REVIEW"
        return decision_json(decision, confidences[n % 4], cited=policy_ids_in(user)[:1])

    report = evaluate(db, make_interpreter(ScriptedProvider(responder), threshold=0.8))
    row = next(r for r in report["threshold_analysis"] if r["threshold"] == 0.8)
    summary = report["summary"]
    assert row["accuracy"] == summary["accuracy"]
    assert row["automation_rate"] == summary["automation_rate"]
    assert row["human_review_rate"] == summary["human_review_rate"]
    assert row["false_approvals"] == summary["false_approval_count"]
    assert row["automated_decision_accuracy"] == summary["automated_decision_accuracy"]

    # ...and the trade-off is monotonic: a stricter bar never automates more or approves more wrongly.
    rows = report["threshold_analysis"]
    assert [r["automation_rate"] for r in rows] == sorted((r["automation_rate"] for r in rows), reverse=True)
    assert [r["false_approvals"] for r in rows] == sorted((r["false_approvals"] for r in rows), reverse=True)


def test_total_provider_outage_degrades_to_human_review_never_to_a_wrong_approval(db, make_interpreter) -> None:
    provider = ScriptedProvider(lambda n, s, u: TransientLLMError("503 overloaded"))
    report = evaluate(db, make_interpreter(provider, max_retries=2))

    assert len(provider.calls) == LOOKUP_CLAIMS * 3  # 1 try + 2 retries per claim, no infinite retrying
    p = report["phase2"]
    assert p["reliability"]["llm_failure_rate"] == 1.0 and p["llm_calls"]["retry_rate"] == 1.0
    assert p["llm_invocation"]["decision_sources"] == {"deterministic": 200 - LOOKUP_CLAIMS, "human_review_fallback": LOOKUP_CLAIMS}
    assert report["summary"]["workflow_failure_rate"] == 0.0  # a fallback is a handled decision, not a crash
    assert report["summary"]["false_approval_count"] == 0
    for category in ("valid_routine", "normal_valid"):
        assert report["confusion_matrix"]["matrix"]["APPROVED"]["HUMAN_REVIEW"] >= CATEGORY_TOTALS[category]


def test_phase2_runs_are_repeatable_and_isolated(db, make_interpreter) -> None:
    first = run_evaluation(db, DATASET, "run-a", make_interpreter(ScriptedProvider(answer())))
    second = run_evaluation(db, DATASET, "run-b", make_interpreter(ScriptedProvider(answer())))
    assert first.engine_version == "rag-llm-v1" and first.id != second.id
    assert first.report["summary"] == second.report["summary"]  # same inputs, same decisions (latency is not in summary)
    assert first.report["by_category"] == second.report["by_category"]
    assert first.report["threshold_analysis"] == second.report["threshold_analysis"]


# ---- CLIs ---------------------------------------------------------------------------------------------


@pytest.fixture()
def cli_env(monkeypatch, session_factory, retriever):
    monkeypatch.setattr(run_phase2, "SessionLocal", session_factory)
    monkeypatch.setattr(run_phase2, "init_db", lambda: None)
    monkeypatch.setattr(run_phase2, "build_retriever", lambda settings, name: retriever)
    return monkeypatch


def test_phase2_cli_runs_and_writes_comparison_against_the_saved_baseline(cli_env, tmp_path: Path, capsys) -> None:
    cli_env.setattr(run_phase2, "build_provider", lambda settings: ScriptedProvider(answer("APPROVED", 0.9)))
    assert run_phase2.main(["--output-dir", str(tmp_path), "--label", "test"]) == 0

    out = capsys.readouterr().out
    assert "PHASE 2: RAG + LLM DETAILS" in out and "PHASE 1 (deterministic baseline) vs PHASE 2 (RAG + LLM)" in out
    assert "Confidence threshold experiment" in out
    comparison = json.loads((tmp_path / "comparison_phase1_vs_phase2.json").read_text())
    rows = {r["metric"]: r for r in comparison["rows"]}
    assert rows["Decision accuracy"]["phase1"] == "92.0%" and rows["False approvals"]["phase1"] == "8"
    assert (tmp_path / "phase2_test.json").exists() and (tmp_path / "comparison_phase1_vs_phase2.txt").exists()


def test_phase2_cli_refuses_to_invent_numbers_without_an_llm(cli_env, tmp_path: Path, capsys) -> None:
    cli_env.setattr(run_phase2, "build_provider", lambda settings: None)
    assert run_phase2.main(["--output-dir", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "LLM_PROVIDER" in out and "ANTHROPIC_API_KEY" in out and "no Phase 2 metrics were produced" in out
    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_phase2_cli_reports_llm_misconfiguration(cli_env) -> None:
    def broken(settings):
        raise LLMConfigError("LLM_MODEL must be set")

    cli_env.setattr(run_phase2, "build_provider", broken)
    with pytest.raises(SystemExit) as exit_info:
        run_phase2.main([])
    assert exit_info.value.code == 2


def test_phase2_cli_refuses_to_compare_against_a_different_dataset(cli_env, tmp_path: Path, capsys) -> None:
    cli_env.setattr(run_phase2, "build_provider", lambda settings: ScriptedProvider(answer()))
    other = json.loads(BASELINE.read_text())
    other["dataset"]["sha256"] = "0" * 64
    baseline = tmp_path / "other_baseline.json"
    baseline.write_text(json.dumps(other))

    assert run_phase2.main(["--baseline", str(baseline), "--output-dir", str(tmp_path / "out")]) == 1
    assert "COMPARISON NOT PRODUCED" in capsys.readouterr().out
    assert not (tmp_path / "out" / "comparison_phase1_vs_phase2.json").exists()


def test_phase2_cli_requires_the_baseline_file(cli_env, tmp_path: Path) -> None:
    cli_env.setattr(run_phase2, "build_provider", lambda settings: ScriptedProvider(answer()))
    with pytest.raises(SystemExit) as exit_info:
        run_phase2.main(["--baseline", str(tmp_path / "missing.json")])
    assert exit_info.value.code == 2


def test_committed_baseline_still_matches_the_benchmark() -> None:
    baseline = json.loads(BASELINE.read_text())
    assert baseline["dataset"]["sha256"] == load_dataset(DATASET)[1]
    assert baseline["summary"]["accuracy"] == 0.92 and baseline["summary"]["false_approval_count"] == 8


def test_smoke_check_makes_one_call_and_prints_the_trace(monkeypatch, make_interpreter, capsys) -> None:
    provider = ScriptedProvider(answer("APPROVED", 0.9))
    monkeypatch.setattr(llm_smoke, "build_interpreter", lambda settings: make_interpreter(provider))
    assert llm_smoke.main() == 0
    out = capsys.readouterr().out
    assert len(provider.calls) == 1 and '"llm_used": true' in out and "Decision: APPROVED" in out


def test_smoke_check_explains_missing_configuration(monkeypatch, capsys) -> None:
    monkeypatch.setattr(llm_smoke, "build_interpreter", lambda settings: None)
    assert llm_smoke.main() == 2 and "LLM_PROVIDER" in capsys.readouterr().out

    def broken(settings):
        raise LLMConfigError("ANTHROPIC_API_KEY must be set")

    monkeypatch.setattr(llm_smoke, "build_interpreter", broken)
    assert llm_smoke.main() == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_smoke_check_exits_nonzero_when_the_call_falls_back(monkeypatch, make_interpreter) -> None:
    provider = ScriptedProvider(lambda n, s, u: "not json")
    monkeypatch.setattr(llm_smoke, "build_interpreter", lambda settings: make_interpreter(provider))
    assert llm_smoke.main() == 1
