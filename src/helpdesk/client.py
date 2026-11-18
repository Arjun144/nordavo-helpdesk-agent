"""HTTP client for the helpdesk API."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .config import Config

PAGE_LIMIT = 10
RETRYABLE_STATUS = frozenset({429, 503})
FATAL_STATUS = frozenset({400, 401, 404, 422})
ALREADY_HANDLED_STATUS = 409


class HelpdeskError(Exception):
    pass


class ApiError(HelpdeskError):
    def __init__(
        self,
        message: str,
        *,
        status: int,
        code: str | None = None,
        problems: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.problems = problems


class FatalApiError(ApiError):
    """400, 401, 404, 422: sending it again fails the same way."""


class AlreadyHandledError(ApiError):
    """409: never retry, the customer already has an answer."""


class RetriesExhaustedError(HelpdeskError):
    def __init__(self, message: str, *, attempts: int, last_error: BaseException) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class RateLimiter:
    """Spaces calls evenly to stay under the API's ceiling."""

    def __init__(
        self,
        max_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.interval = 1.0 / max_per_second
        self.clock = clock
        self.sleep = sleep
        self.next_allowed = 0.0

    def acquire(self) -> None:
        moment = self.clock()
        if moment < self.next_allowed:
            self.sleep(self.next_allowed - moment)
            moment = self.next_allowed
        self.next_allowed = moment + self.interval


class HelpdeskClient:
    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[float, float], float] = random.uniform,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        jitter: float = 0.25,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.config = config
        self.sleep = sleep
        self.rng = rng
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.jitter = jitter
        self.limiter = limiter or RateLimiter(
            config.max_requests_per_second, clock=clock, sleep=sleep
        )
        self.http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers=self.build_headers(config),
            transport=transport,
        )

    @staticmethod
    def build_headers(config: Config) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "X-Candidate-Email": config.candidate_email,
            "Accept": "application/json",
        }
        if config.candidate_name:
            headers["X-Candidate-Name"] = config.candidate_name
        return headers

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> HelpdeskClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def status(self) -> dict[str, Any]:
        return self.request("GET", "/status")

    def examples(self) -> list[dict[str, Any]]:
        payload = self.request("GET", "/examples")
        if isinstance(payload, list):
            return payload
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise HelpdeskError("/examples returned no list of examples")
        return examples

    def iter_tickets(
        self, *, status: str = "open", limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        seen: set[str] = set()
        count = 0

        while True:
            params: dict[str, Any] = {"limit": PAGE_LIMIT, "status": status}
            if cursor is not None:
                params["cursor"] = cursor

            page = self.request("GET", "/tickets", params=params)
            for ticket in page.get("tickets") or []:
                yield ticket
                count += 1
                if limit is not None and count >= limit:
                    return

            cursor = page.get("next_cursor")
            if not cursor:
                return
            if cursor in seen:
                raise HelpdeskError(f"/tickets repeated cursor {cursor!r}")
            seen.add(cursor)

    def list_tickets(
        self, *, status: str = "open", limit: int | None = None
    ) -> list[dict[str, Any]]:
        return list(self.iter_tickets(status=status, limit=limit))

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        return self.request("GET", f"/tickets/{ticket_id}")

    def reply(self, ticket_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/tickets/{ticket_id}/reply", json=payload)

    def handoff(self, ticket_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/tickets/{ticket_id}/handoff", json=payload)

    def close_ticket(self, ticket_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/tickets/{ticket_id}/close", json=payload)

    def reset_practice(self) -> dict[str, Any]:
        return self.request("POST", "/practice/reset")

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        last_error: BaseException = HelpdeskError("no attempt was made")

        for attempt in range(self.config.max_attempts):
            self.limiter.acquire()

            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt + 1 == self.config.max_attempts:
                    break
                self.sleep(self.backoff_delay(attempt))
                continue

            if response.status_code == ALREADY_HANDLED_STATUS:
                raise error_from_response(response, AlreadyHandledError)
            if response.status_code in FATAL_STATUS:
                raise error_from_response(response, FatalApiError)
            if response.status_code in RETRYABLE_STATUS:
                last_error = error_from_response(response, ApiError)
                if attempt + 1 == self.config.max_attempts:
                    break
                self.sleep(self.retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise error_from_response(response, FatalApiError)

            return decode(response)

        raise RetriesExhaustedError(
            f"{method} {path} still failing after {self.config.max_attempts} attempts: "
            f"{last_error}",
            attempts=self.config.max_attempts,
            last_error=last_error,
        )

    def backoff_delay(self, attempt: int) -> float:
        delay = min(self.backoff_base * (2**attempt), self.backoff_cap)
        return delay + self.rng(0.0, self.jitter)

    def retry_delay(self, response: httpx.Response, attempt: int) -> float:
        if response.status_code == 429:
            wait = retry_after_seconds(response)
            if wait is not None:
                return wait
        return self.backoff_delay(attempt)


def retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def decode(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise HelpdeskError(
            f"expected JSON from {response.request.method} {response.request.url}, "
            f"got {response.text[:200]!r}"
        ) from exc


def error_from_response(response: httpx.Response, error_type: type[ApiError]) -> ApiError:
    payload: Any = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    code: str | None = None
    problems: Any = None
    if isinstance(payload, dict):
        raw_code = payload.get("error")
        code = raw_code if isinstance(raw_code, str) else None
        problems = payload.get("problems")

    detail = f"{response.status_code} {code or response.reason_phrase}".strip()
    if problems:
        detail = f"{detail}: {problems}"
    return error_type(detail, status=response.status_code, code=code, problems=problems)
