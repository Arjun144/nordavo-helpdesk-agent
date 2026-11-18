"""Settings, all from the environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://nakywgkztvzkihdjnwbc.supabase.co/functions/v1/helpdesk"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_REQUESTS_PER_SECOND = 5.0
DEFAULT_CONFIDENCE_FLOOR = 0.5


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Config:
    api_key: str = field(repr=False)
    candidate_email: str
    base_url: str = DEFAULT_BASE_URL
    candidate_name: str | None = None
    max_requests_per_second: float = 4.0
    max_attempts: int = 5
    timeout_seconds: float = 30.0
    deepseek_api_key: str = field(default="", repr=False)
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    deepseek_model: str = "deepseek-chat"
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    ledger_path: Path = Path("runs/ledger.db")
    logs_dir: Path = Path("logs")
    docs_dir: Path = Path("docs")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        if env is None:
            # Shell exports win over the file.
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
            source: Mapping[str, str] = os.environ
        else:
            source = env

        api_key = source.get("HELPDESK_API_KEY", "").strip()
        email = source.get("HELPDESK_CANDIDATE_EMAIL", "").strip()
        missing = [
            name
            for name, value in (
                ("HELPDESK_API_KEY", api_key),
                ("HELPDESK_CANDIDATE_EMAIL", email),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "missing required environment variable(s): " + ", ".join(missing)
            )
        if "@" not in email:
            raise ConfigError(
                "HELPDESK_CANDIDATE_EMAIL must be the email address you applied with"
            )

        max_rps = parse_float(source, "HELPDESK_MAX_RPS", 4.0)
        if not 0 < max_rps <= MAX_REQUESTS_PER_SECOND:
            raise ConfigError(
                f"HELPDESK_MAX_RPS must be greater than 0 and at most "
                f"{MAX_REQUESTS_PER_SECOND:g}, got {max_rps:g}"
            )

        return cls(
            api_key=api_key,
            candidate_email=email,
            base_url=source.get("HELPDESK_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            candidate_name=source.get("HELPDESK_CANDIDATE_NAME", "").strip() or None,
            max_requests_per_second=max_rps,
            deepseek_api_key=source.get("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=(
                source.get("DEEPSEEK_BASE_URL", "").strip() or DEFAULT_DEEPSEEK_BASE_URL
            ),
            deepseek_model=source.get("DEEPSEEK_MODEL", "").strip() or "deepseek-chat",
            confidence_floor=parse_float(
                source, "CONFIDENCE_FLOOR", DEFAULT_CONFIDENCE_FLOOR
            ),
            ledger_path=Path(source.get("LEDGER_PATH", "").strip() or "runs/ledger.db"),
            logs_dir=Path(source.get("LOGS_DIR", "").strip() or "logs"),
            docs_dir=Path(source.get("DOCS_DIR", "").strip() or "docs"),
        )


def parse_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
