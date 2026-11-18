"""One redacted line per ticket, in logs/."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Decision
from .policy import redact

LOG_FILE = "decisions.log"


@dataclass(frozen=True, slots=True)
class LogEntry:
    ticket_id: str
    outcome: str
    decision: Decision | None
    detail: str = ""
    flags: str = ""

    def line(self) -> str:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.decision is None:
            extra = f" flags={self.flags}" if self.flags else ""
            body = f"no decision; {one_line(self.detail)}{extra}"
        else:
            d = self.decision
            confidence = "-" if d.confidence is None else f"{d.confidence:.2f}"
            body = (
                f"action={d.action.value} language={d.language.value} "
                f"category={d.category.value} priority={d.priority.value} "
                f"confidence={confidence} faq={','.join(d.faq_refs) or '-'} "
                f"reason={one_line(d.reason)}"
            )
            if self.flags:
                body = f"{body} flags={self.flags}"
            if self.detail:
                body = f"{body} detail={one_line(self.detail)}"
        return f"{stamp}\t{self.ticket_id}\t{self.outcome}\t{redact(body)}"


class Logbook:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.path = self.directory / LOG_FILE

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def write(self, entry: LogEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.line() + "\n")


def one_line(text: str) -> str:
    return " ".join(text.split())
