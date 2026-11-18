from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from helpdesk.client import (
    AlreadyHandledError,
    FatalApiError,
    HelpdeskClient,
    HelpdeskError,
    RateLimiter,
    RetriesExhaustedError,
)
from helpdesk.config import Config


class NoLimit:
    """Rate limiter stub so retry timing can be asserted exactly."""

    def acquire(self) -> None:
        pass


class Recorder:
    """MockTransport handler that replays scripted responses and records requests."""

    def __init__(self, *script: tuple[int, Any] | tuple[int, Any, dict[str, str]]) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.script:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        status, payload, *rest = self.script.pop(0)
        headers = rest[0] if rest else None
        if payload is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, json=payload, headers=headers)


def make_client(
    handler: Recorder, *, max_attempts: int = 5
) -> tuple[HelpdeskClient, list[float]]:
    config = Config(
        api_key="test-key",
        candidate_email="dev@example.com",
        max_attempts=max_attempts,
    )
    sleeps: list[float] = []
    client = HelpdeskClient(
        config,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        rng=lambda low, high: 0.0,
        jitter=0.0,
        limiter=NoLimit(),
    )
    return client, sleeps


def test_sends_required_headers_and_optional_name() -> None:
    recorder = Recorder((200, {"mode": "practice", "open": 30, "handled": 0}))
    config = Config(
        api_key="test-key",
        candidate_email="dev@example.com",
        candidate_name="Arjun",
    )
    client = HelpdeskClient(config, transport=httpx.MockTransport(recorder), limiter=NoLimit())

    assert client.status() == {"mode": "practice", "open": 30, "handled": 0}
    headers = recorder.requests[0].headers
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Candidate-Email"] == "dev@example.com"
    assert headers["X-Candidate-Name"] == "Arjun"


def test_retries_503_with_exponential_backoff() -> None:
    recorder = Recorder((503, None), (503, None), (200, {"ok": True}))
    client, sleeps = make_client(recorder)

    assert client.status() == {"ok": True}
    assert len(recorder.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_waits_for_retry_after_on_429() -> None:
    recorder = Recorder((429, None, {"Retry-After": "3"}), (200, {"ok": True}))
    client, sleeps = make_client(recorder)

    assert client.status() == {"ok": True}
    assert sleeps == [3.0]


def test_gives_up_after_max_attempts() -> None:
    recorder = Recorder((503, None), (503, None), (503, None))
    client, sleeps = make_client(recorder, max_attempts=3)

    with pytest.raises(RetriesExhaustedError) as excinfo:
        client.status()

    assert excinfo.value.attempts == 3
    assert len(recorder.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_transport_error_is_retried() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True})

    client = HelpdeskClient(
        Config(api_key="k", candidate_email="dev@example.com"),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        rng=lambda low, high: 0.0,
        jitter=0.0,
        limiter=NoLimit(),
    )

    assert client.status() == {"ok": True}
    assert attempts["count"] == 2


def test_validation_failure_is_fatal_and_keeps_problems() -> None:
    recorder = Recorder(
        (422, {"error": "validation_failed", "problems": ["priority is required"]})
    )
    client, sleeps = make_client(recorder)

    with pytest.raises(FatalApiError) as excinfo:
        client.reply("T005", {"language": "nl"})

    assert excinfo.value.status == 422
    assert excinfo.value.code == "validation_failed"
    assert excinfo.value.problems == ["priority is required"]
    assert len(recorder.requests) == 1
    assert sleeps == []


def test_already_handled_is_terminal() -> None:
    recorder = Recorder((409, {"error": "already_handled"}))
    client, sleeps = make_client(recorder)

    with pytest.raises(AlreadyHandledError):
        client.handoff("T005", {"language": "nl"})

    assert len(recorder.requests) == 1
    assert sleeps == []


def test_pagination_follows_next_cursor() -> None:
    first_page = [{"id": f"T{i:03d}"} for i in range(1, 11)]
    second_page = [{"id": "T011"}, {"id": "T012"}]
    recorder = Recorder(
        (200, {"tickets": first_page, "next_cursor": "T010"}),
        (200, {"tickets": second_page, "next_cursor": None}),
    )
    client, _ = make_client(recorder)

    tickets = client.list_tickets(status="open")

    assert [ticket["id"] for ticket in tickets] == [f"T{i:03d}" for i in range(1, 13)]
    assert "cursor=" not in str(recorder.requests[0].url)
    assert "cursor=T010" in str(recorder.requests[1].url)
    assert "limit=10" in str(recorder.requests[1].url)


def test_pagination_stops_at_limit() -> None:
    page = [{"id": f"T{i:03d}"} for i in range(1, 11)]
    recorder = Recorder((200, {"tickets": page, "next_cursor": "T010"}))
    client, _ = make_client(recorder)

    tickets: Iterator[dict[str, Any]] = client.iter_tickets(limit=3)

    assert [ticket["id"] for ticket in tickets] == ["T001", "T002", "T003"]
    assert len(recorder.requests) == 1


def test_repeated_cursor_is_an_error() -> None:
    page = [{"id": "T001"}]
    recorder = Recorder(
        (200, {"tickets": page, "next_cursor": "T001"}),
        (200, {"tickets": page, "next_cursor": "T001"}),
    )
    client, _ = make_client(recorder)

    with pytest.raises(HelpdeskError):
        client.list_tickets()


def test_rate_limiter_spaces_requests() -> None:
    now = {"value": 0.0}
    sleeps: list[float] = []

    def clock() -> float:
        return now["value"]

    def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        now["value"] += seconds

    limiter = RateLimiter(4, clock=clock, sleep=sleep)
    for _ in range(3):
        limiter.acquire()

    assert sleeps == [0.25, 0.25]
