"""Structured output validation, citation grounding and retry behaviour. The provider is always a test double."""

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.enums import ClaimOutcome, ReasonCode
from app.services.llm import structured
from app.services.llm.base import LLMConfigError, LLMResponse, PermanentLLMError, TransientLLMError
from app.services.llm.factory import build_provider
from app.services.llm.schemas import LLMDecision
from app.services.llm.structured import extract_json, run_structured
from tests.fakes import ScriptedProvider, decision_json

ALLOWED = ["POL-DOC-001", "POL-AMB-002"]


def run(provider: ScriptedProvider, max_retries: int = 2, sleeps: list | None = None):
    return run_structured(
        provider, "system", "user prompt", ALLOWED, timeout=7.5, max_retries=max_retries,
        max_output_tokens=321, backoff_seconds=2.0, sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )  # fmt: skip


# ---- schema -------------------------------------------------------------------------------------------


def test_valid_decision_parses() -> None:
    parsed = LLMDecision.model_validate_json(decision_json("APPROVED", 0.85))
    assert parsed.decision == ClaimOutcome.APPROVED and parsed.confidence == 0.85
    assert parsed.reason_code == ReasonCode.VALID_STANDARD_CLAIM and parsed.cited_policy_ids == ["POL-DOC-001"]


@pytest.mark.parametrize("confidence", [-0.1, 1.01])
def test_confidence_must_be_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValidationError):
        LLMDecision.model_validate_json(decision_json(confidence=confidence))


def test_boundary_confidences_are_valid() -> None:
    assert LLMDecision.model_validate_json(decision_json(confidence=0.0)).confidence == 0.0
    assert LLMDecision.model_validate_json(decision_json(confidence=1.0)).confidence == 1.0


@pytest.mark.parametrize(
    "decision, reason",
    [("APPROVED", "AMBIGUOUS_CLINICAL_NOTES"), ("HUMAN_REVIEW", "VALID_STANDARD_CLAIM"), ("REJECTED", "VALID_STANDARD_CLAIM"), ("APPROVED", "DUPLICATE_CLAIM")],
)
def test_reason_code_must_be_consistent_with_decision(decision: str, reason: str) -> None:
    with pytest.raises(ValidationError):
        LLMDecision.model_validate_json(decision_json(decision, reason=reason))


def test_decision_must_be_an_allowed_value_and_citations_are_required() -> None:
    with pytest.raises(ValidationError):
        LLMDecision.model_validate_json(decision_json("MAYBE", reason="VALID_STANDARD_CLAIM"))
    with pytest.raises(ValidationError):
        LLMDecision.model_validate_json(decision_json(cited=[]))
    with pytest.raises(ValidationError):
        LLMDecision.model_validate({"decision": "APPROVED"})


def test_extra_fields_such_as_reasoning_are_ignored_not_stored() -> None:
    payload = decision_json()[:-1] + ', "chain_of_thought": "long private reasoning"}'
    parsed = LLMDecision.model_validate_json(payload)
    assert "chain_of_thought" not in parsed.model_dump()


# ---- JSON extraction ----------------------------------------------------------------------------------


def test_extract_json_handles_plain_fenced_and_wrapped_replies() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure, here is the answer:\n{"a": {"b": 2}}\nHope that helps') == {"a": {"b": 2}}


@pytest.mark.parametrize("text", ["", "no json here", '{"a": ', "[1, 2, 3]"])
def test_extract_json_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json(text)


# ---- retries and grounding ----------------------------------------------------------------------------


def test_success_on_first_attempt() -> None:
    provider = ScriptedProvider(lambda n, s, u: decision_json(cited=["POL-DOC-001"]))
    result = run(provider)
    assert result.decision is not None and result.failure is None
    assert (result.calls, result.retries) == (1, 0)
    assert result.attempts[0].outcome == "ok" and result.model == "fake-model"
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert provider.calls[0]["timeout"] == 7.5 and provider.calls[0]["max_output_tokens"] == 321


def test_malformed_output_is_retried_with_feedback_then_succeeds() -> None:
    replies = ["this is not json", '{"decision": "APPROVED"}', decision_json()]
    provider = ScriptedProvider(lambda n, s, u: replies[n - 1])
    result = run(provider)
    assert result.decision is not None
    assert [a.outcome for a in result.attempts] == ["malformed_output", "malformed_output", "ok"]
    assert (result.calls, result.retries) == (3, 2)
    assert "previous reply was rejected" in provider.calls[1]["user"] and "POL-DOC-001" in provider.calls[1]["user"]
    assert "previous reply was rejected" not in provider.calls[0]["user"]


def test_persistent_malformed_output_stops_after_the_retry_limit() -> None:
    provider = ScriptedProvider(lambda n, s, u: "still not json")
    result = run(provider, max_retries=2)
    assert result.decision is None and result.failure == "malformed_output"
    assert len(provider.calls) == 3  # 1 attempt + 2 retries, never more
    assert result.calls == 3


def test_invalid_policy_citation_is_detected_and_retried() -> None:
    replies = [decision_json(cited=["POL-INVENTED-999"]), decision_json(cited=["POL-DOC-001"])]
    provider = ScriptedProvider(lambda n, s, u: replies[n - 1])
    result = run(provider)
    assert result.decision is not None and result.decision.cited_policy_ids == ["POL-DOC-001"]
    first = result.attempts[0]
    assert first.outcome == "invalid_citation" and first.unsupported_citations == ["POL-INVENTED-999"]
    assert "POL-INVENTED-999" in provider.calls[1]["user"]  # the model is told what it got wrong


def test_partially_invalid_citations_are_still_invalid() -> None:
    provider = ScriptedProvider(lambda n, s, u: decision_json(cited=["POL-DOC-001", "POL-FAKE-1"]))
    result = run(provider, max_retries=0)
    assert result.decision is None and result.failure == "invalid_citation"
    assert result.attempts[0].unsupported_citations == ["POL-FAKE-1"] and result.attempts[0].cited_count == 2


def test_citing_a_real_policy_that_was_not_retrieved_is_invalid() -> None:
    """POL-HC-001 exists in the knowledge base but was not in this claim's context, so it is unsupported."""
    provider = ScriptedProvider(lambda n, s, u: decision_json(cited=["POL-HC-001"]))
    assert run(provider, max_retries=0).failure == "invalid_citation"


def test_transient_errors_are_retried_with_exponential_backoff() -> None:
    def responder(n, s, u):
        return TransientLLMError("rate limited") if n < 3 else decision_json()

    sleeps: list[float] = []
    result = run(ScriptedProvider(responder), sleeps=sleeps)
    assert result.decision is not None
    assert [a.outcome for a in result.attempts] == ["transient_error", "transient_error", "ok"]
    assert sleeps == [2.0, 4.0]  # backoff * 2**(attempt-1), no sleep after the final attempt


def test_transient_errors_exhaust_retries_without_sleeping_after_the_last_attempt() -> None:
    sleeps: list[float] = []
    provider = ScriptedProvider(lambda n, s, u: TransientLLMError("503"))
    result = run(provider, max_retries=2, sleeps=sleeps)
    assert result.decision is None and result.failure == "transient_error" and len(provider.calls) == 3
    assert sleeps == [2.0, 4.0]


def test_permanent_errors_are_not_retried() -> None:
    provider = ScriptedProvider(lambda n, s, u: PermanentLLMError("401 bad key"))
    result = run(provider)
    assert result.decision is None and result.failure == "provider_error"
    assert len(provider.calls) == 1 and "401" in result.attempts[0].error


def test_unexpected_provider_exceptions_do_not_escape() -> None:
    provider = ScriptedProvider(lambda n, s, u: KeyError("boom"))
    result = run(provider)
    assert result.decision is None and result.failure == "provider_error" and len(provider.calls) == 1


def test_zero_retries_means_exactly_one_call() -> None:
    provider = ScriptedProvider(lambda n, s, u: "garbage")
    run(provider, max_retries=0)
    assert len(provider.calls) == 1


def test_negative_retries_still_make_one_call() -> None:
    provider = ScriptedProvider(lambda n, s, u: "garbage")
    run(provider, max_retries=-5)
    assert len(provider.calls) == 1


def test_mixed_failures_then_success_accumulate_tokens_and_latency() -> None:
    replies = [TransientLLMError("x"), "garbage", decision_json()]
    result = run(ScriptedProvider(lambda n, s, u: replies[n - 1]), sleeps=[])
    assert result.decision is not None and result.calls == 3
    assert result.input_tokens == 200 and result.latency_ms >= 0  # only calls that returned tokens are counted


# ---- provider adapters (SDKs replaced by fakes) --------------------------------------------------------


class FakeAnthropicClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def anthropic_response(text="{}", stop="end_turn"):
    return Obj(content=[Obj(type="thinking", text="ignored"), Obj(type="text", text=text)], usage=Obj(input_tokens=11, output_tokens=7), model="claude-x", stop_reason=stop)


def test_anthropic_provider_maps_request_and_response() -> None:
    from app.services.llm.anthropic_provider import AnthropicProvider

    client = FakeAnthropicClient(anthropic_response('{"ok": true}'))
    provider = AnthropicProvider("key", "claude-x", effort="low", client=client)
    response = provider.complete("sys", "usr", timeout=9.0, max_output_tokens=123)

    assert response == LLMResponse('{"ok": true}', 11, 7, "claude-x", "end_turn")
    request = client.requests[0]
    assert request["model"] == "claude-x" and request["max_tokens"] == 123 and request["timeout"] == 9.0
    assert request["system"] == "sys" and request["messages"] == [{"role": "user", "content": "usr"}]
    assert request["output_config"] == {"effort": "low"} and "temperature" not in request


def test_anthropic_provider_omits_effort_unless_configured() -> None:
    from app.services.llm.anthropic_provider import AnthropicProvider

    client = FakeAnthropicClient(anthropic_response())
    AnthropicProvider("key", "m", client=client).complete("s", "u", timeout=1, max_output_tokens=1)
    assert "output_config" not in client.requests[0]


def _anthropic_status_error(cls, status: int):
    request = httpx.Request("POST", "https://api.example.test/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def test_anthropic_errors_are_classified_as_transient_or_permanent() -> None:
    import anthropic

    from app.services.llm.anthropic_provider import AnthropicProvider

    request = httpx.Request("POST", "https://api.example.test/v1/messages")
    cases = [
        (anthropic.APITimeoutError(request=request), TransientLLMError),
        (anthropic.APIConnectionError(request=request), TransientLLMError),
        (_anthropic_status_error(anthropic.RateLimitError, 429), TransientLLMError),
        (_anthropic_status_error(anthropic.InternalServerError, 500), TransientLLMError),
        (_anthropic_status_error(anthropic.InternalServerError, 529), TransientLLMError),
        (_anthropic_status_error(anthropic.AuthenticationError, 401), PermanentLLMError),
        (_anthropic_status_error(anthropic.BadRequestError, 400), PermanentLLMError),
    ]
    for error, expected in cases:
        provider = AnthropicProvider("key", "m", client=FakeAnthropicClient(error=error))
        with pytest.raises(expected):
            provider.complete("s", "u", timeout=1, max_output_tokens=1)


def test_anthropic_refusal_is_a_permanent_error() -> None:
    from app.services.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider("key", "m", client=FakeAnthropicClient(anthropic_response(stop="refusal")))
    with pytest.raises(PermanentLLMError, match="refused"):
        provider.complete("s", "u", timeout=1, max_output_tokens=1)


class FakeOpenAIClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []
        self.chat = Obj(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def openai_response(text="{}"):
    return Obj(choices=[Obj(message=Obj(content=text), finish_reason="stop")], usage=Obj(prompt_tokens=5, completion_tokens=3), model="gpt-x")


def test_openai_provider_maps_request_and_response() -> None:
    from app.services.llm.openai_provider import OpenAIProvider

    client = FakeOpenAIClient(openai_response('{"ok": 1}'))
    response = OpenAIProvider("key", "gpt-x", client=client).complete("sys", "usr", timeout=4.0, max_output_tokens=50)
    assert response == LLMResponse('{"ok": 1}', 5, 3, "gpt-x", "stop")
    request = client.requests[0]
    assert request["max_completion_tokens"] == 50 and request["timeout"] == 4.0
    assert [m["role"] for m in request["messages"]] == ["system", "user"]

    compatible = FakeOpenAIClient(openai_response())
    OpenAIProvider("key", "local", base_url="http://localhost:8000/v1", client=compatible).complete("s", "u", timeout=1, max_output_tokens=9)
    assert compatible.requests[0]["max_tokens"] == 9 and "max_completion_tokens" not in compatible.requests[0]


def test_openai_errors_are_classified() -> None:
    import openai

    from app.services.llm.openai_provider import OpenAIProvider

    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    for error, expected in [
        (openai.APITimeoutError(request=request), TransientLLMError),
        (openai.RateLimitError("x", response=httpx.Response(429, request=request), body=None), TransientLLMError),
        (openai.AuthenticationError("x", response=httpx.Response(401, request=request), body=None), PermanentLLMError),
    ]:
        with pytest.raises(expected):
            OpenAIProvider("key", "m", client=FakeOpenAIClient(error=error)).complete("s", "u", timeout=1, max_output_tokens=1)


# ---- provider factory ---------------------------------------------------------------------------------


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_factory_returns_none_when_llm_is_disabled() -> None:
    assert build_provider(settings()) is None
    assert build_provider(settings(llm_provider="none", llm_model="x")) is None


def test_factory_requires_model_and_key() -> None:
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        build_provider(settings(llm_provider="anthropic", anthropic_api_key="k"))
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        build_provider(settings(llm_provider="anthropic", llm_model="m"))
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_provider(settings(llm_provider="openai", llm_model="m"))


def test_factory_treats_empty_credentials_as_missing() -> None:
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        build_provider(settings(llm_provider="anthropic", llm_model="m", anthropic_api_key="  "))
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_provider(settings(llm_provider="openai", llm_model="m", openai_api_key=""))


def test_factory_builds_providers_without_touching_the_network() -> None:
    anthropic_provider = build_provider(settings(llm_provider="anthropic", llm_model="m", anthropic_api_key="secret", llm_effort="low"))
    assert (anthropic_provider.name, anthropic_provider.model, anthropic_provider.effort) == ("anthropic", "m", "low")
    openai_provider = build_provider(settings(llm_provider="openai", llm_model="g", openai_api_key="secret", openai_base_url="http://x/v1"))
    assert (openai_provider.name, openai_provider.model) == ("openai", "g")


def test_api_keys_are_not_exposed_in_settings_repr() -> None:
    assert "super-secret" not in repr(settings(anthropic_api_key="super-secret"))


def test_structured_module_never_retries_beyond_limit_even_with_huge_config() -> None:
    provider = ScriptedProvider(lambda n, s, u: TransientLLMError("x"))
    result = run_structured(provider, "s", "u", ALLOWED, timeout=1, max_retries=4, max_output_tokens=1, sleep=lambda _s: None)
    assert len(provider.calls) == 5 and structured.TRANSIENT_ERROR == result.failure
