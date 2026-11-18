from __future__ import annotations

import json

import httpx
import pytest

from helpdesk.config import Config
from helpdesk.llm import DeepSeekLLM, LLMAuthError, LLMError, Usage


def completion(content: str, *, prompt: int = 100, out: int = 20) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out},
    }


def make_config(**overrides) -> Config:
    return Config(
        api_key="helpdesk-key",
        candidate_email="dev@example.com",
        deepseek_api_key=overrides.pop("deepseek_api_key", "deepseek-key"),
        **overrides,
    )


def test_returns_a_parsed_decision_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion('{"action": "reply"}'))

    llm = DeepSeekLLM(make_config(), transport=httpx.MockTransport(handler))
    decision, usage = llm.complete_json(system="s", user="u")

    assert decision == {"action": "reply"}
    assert usage == Usage(prompt_tokens=100, completion_tokens=20)
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0


def test_missing_key_fails_before_any_request() -> None:
    with pytest.raises(LLMAuthError):
        DeepSeekLLM(make_config(deepseek_api_key=""))


def test_401_is_fatal() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    llm = DeepSeekLLM(make_config(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMAuthError):
        llm.complete_json(system="s", user="u")
    assert calls["n"] == 1


def test_503_is_retried_then_succeeds() -> None:
    responses = [
        httpx.Response(503),
        httpx.Response(200, json=completion('{"action": "reply"}')),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    sleeps: list[float] = []
    llm = DeepSeekLLM(
        make_config(), transport=httpx.MockTransport(handler), sleep=sleeps.append
    )

    assert llm.complete_json(system="s", user="u")[0] == {"action": "reply"}
    assert sleeps == [0.5]


def test_non_json_content_is_an_error_not_a_guess() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("Sure! Here is your JSON:"))

    llm = DeepSeekLLM(make_config(), transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError, match="not JSON"):
        llm.complete_json(system="s", user="u")


def test_cost_maths() -> None:
    assert Usage(prompt_tokens=1_000_000, completion_tokens=0).cost_usd == pytest.approx(0.27)
    assert Usage(prompt_tokens=0, completion_tokens=1_000_000).cost_usd == pytest.approx(1.10)
