"""DeepSeek call, JSON mode, temperature 0."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# US dollars per million tokens. Check these before trusting the cost line.
INPUT_USD_PER_MILLION = 0.27
OUTPUT_USD_PER_MILLION = 1.10


class LLMError(RuntimeError):
    pass


class LLMAuthError(LLMError):
    """Bad or missing key. Retrying will not help."""


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
            + self.completion_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


class DeepSeekLLM:
    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.deepseek_api_key:
            raise LLMAuthError("DEEPSEEK_API_KEY is not set")
        self.config = config
        self.sleep = sleep
        self.http = httpx.Client(
            base_url=config.deepseek_base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> DeepSeekLLM:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete_json(self, *, system: str, user: str) -> tuple[Mapping[str, Any], Usage]:
        payload = {
            "model": self.config.deepseek_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception = LLMError("no attempt was made")
        for attempt in range(self.config.max_attempts):
            try:
                response = self.http.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = LLMError(f"transport error: {exc}")
            else:
                if response.status_code in (401, 403):
                    raise LLMAuthError(
                        f"DeepSeek rejected the key: {response.status_code}"
                    )
                if response.status_code < 400:
                    return parse_choice(response)
                last_error = LLMError(
                    f"DeepSeek returned {response.status_code}: {response.text[:200]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    raise last_error

            if attempt + 1 < self.config.max_attempts:
                self.sleep(min(0.5 * 2**attempt, 8.0))

        raise LLMError(f"model call failed after {self.config.max_attempts}: {last_error}")


def parse_choice(response: httpx.Response) -> tuple[Mapping[str, Any], Usage]:
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMError("DeepSeek returned a non-JSON envelope") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected DeepSeek response: {str(body)[:200]}") from exc

    try:
        decision = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise LLMError(f"model output was not JSON: {str(content)[:200]}") from exc
    if not isinstance(decision, dict):
        raise LLMError(f"model output was not an object: {str(content)[:200]}")

    usage = body.get("usage") or {}
    return decision, Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
