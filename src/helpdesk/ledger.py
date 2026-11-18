"""SQLite record of what has been claimed and sent."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

INTENT = "intent"
SENT = "sent"
FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    ticket_id  TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    action     TEXT,
    detail     TEXT,
    claimed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def claim(self, ticket_id: str) -> bool:
        """Take the ticket. False means it was already taken."""
        moment = timestamp()
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO decisions "
                "(ticket_id, state, claimed_at, updated_at) VALUES (?, ?, ?, ?)",
                (ticket_id, INTENT, moment, moment),
            )
            return cursor.rowcount == 1

    def mark(
        self,
        ticket_id: str,
        state: str,
        *,
        action: str | None = None,
        detail: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE decisions SET state = ?, action = COALESCE(?, action), "
                "detail = COALESCE(?, detail), updated_at = ? WHERE ticket_id = ?",
                (state, action, detail, timestamp(), ticket_id),
            )

    def clear(self) -> None:
        """Forget everything. Only used when the client re-opens the tickets."""
        with self.connection() as conn:
            conn.execute("DELETE FROM decisions")

    def release(self, ticket_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM decisions WHERE ticket_id = ?", (ticket_id,))

    def state_of(self, ticket_id: str) -> str | None:
        row = self.fetch_one("SELECT state FROM decisions WHERE ticket_id = ?", ticket_id)
        return row[0] if row else None

    def unfinished(self) -> list[tuple[str, str]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT ticket_id, claimed_at FROM decisions WHERE state = ? "
                "ORDER BY claimed_at",
                (INTENT,),
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) FROM decisions GROUP BY state"
            ).fetchall()
        return {str(state): int(count) for state, count in rows}

    def fetch_one(self, sql: str, *params: object) -> tuple[object, ...] | None:
        with self.connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return tuple(row) if row else None
