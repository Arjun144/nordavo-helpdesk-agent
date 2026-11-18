"""Claim, decide, send, mark."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .client import AlreadyHandledError
from .ledger import INTENT, SENT, Ledger
from .logbook import LogEntry, Logbook
from .models import Decision
from .policy import policy_flags

UNKNOWN = "unknown"


@dataclass
class RunResult:
    sent: int = 0
    skipped: int = 0
    failed: int = 0
    reconciled: int = 0
    per_ticket: list[tuple[str, str]] = field(default_factory=list)


class Helpdesk(Protocol):
    def iter_tickets(
        self, *, status: str = "open", limit: int | None = None
    ) -> Iterator[dict[str, Any]]: ...

    def get_ticket(self, ticket_id: str) -> dict[str, Any]: ...

    def reply(self, ticket_id: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def handoff(self, ticket_id: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def close_ticket(
        self, ticket_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


class Decides(Protocol):
    def decide(self, ticket: Mapping[str, Any]) -> Decision: ...


class Runner:
    def __init__(
        self,
        client: Helpdesk,
        decider: Decides,
        ledger: Ledger,
        logbook: Logbook,
        *,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.decider = decider
        self.ledger = ledger
        self.logbook = logbook
        self.dry_run = dry_run

    def reconcile(self) -> int:
        """Settle claims left behind by a crash."""
        settled = 0
        for ticket_id, _claimed_at in self.ledger.unfinished():
            try:
                ticket = self.client.get_ticket(ticket_id)
            except Exception as error:
                # The claim stays open; the next run asks again.
                self.write_log(ticket_id, "reconcile_error", None, str(error))
                continue

            if ticket.get("handled"):
                self.ledger.mark(ticket_id, SENT, detail="reconciled as already handled")
                settled += 1
            else:
                self.ledger.release(ticket_id)
        return settled

    def run(self, *, limit: int | None = None, status: str = "open") -> RunResult:
        result = RunResult()
        result.reconciled = self.reconcile()

        for ticket in self.client.iter_tickets(status=status, limit=limit):
            ticket_id = str(ticket.get("id") or UNKNOWN)
            if not self.ledger.claim(ticket_id):
                result.skipped += 1
                result.per_ticket.append((ticket_id, "already claimed"))
                continue

            try:
                decision = self.decider.decide(ticket)
            except Exception as error:
                # Nothing was sent, so give the claim back and let a later run retry.
                self.ledger.release(ticket_id)
                self.write_log(ticket_id, "failed", None, str(error))
                result.failed += 1
                result.per_ticket.append((ticket_id, "decide failed"))
                continue

            flags = ",".join(policy_flags(dict(ticket), decision))

            if self.dry_run:
                self.ledger.release(ticket_id)
                self.write_log(ticket_id, "dry_run", decision, flags=flags)
                result.per_ticket.append((ticket_id, f"dry run: {decision.action.value}"))
                continue

            try:
                self.send(ticket_id, decision)
            except AlreadyHandledError as error:
                # Someone else answered it. Terminal, and not a failure.
                self.ledger.mark(ticket_id, SENT, detail="already handled elsewhere")
                self.write_log(ticket_id, "already_handled", decision, str(error), flags)
                result.skipped += 1
                result.per_ticket.append((ticket_id, "already handled"))
                continue
            except Exception as error:
                # The request may have landed, so keep the claim for reconcile.
                self.write_log(ticket_id, "failed", decision, str(error), flags)
                result.failed += 1
                result.per_ticket.append((ticket_id, f"send failed: {error}"))
                continue

            self.ledger.mark(ticket_id, SENT, action=decision.action.value)
            self.write_log(ticket_id, "sent", decision, flags=flags)
            result.sent += 1
            result.per_ticket.append((ticket_id, decision.action.value))

        return result

    def send(self, ticket_id: str, decision: Decision) -> None:
        payload = decision.payload()
        if decision.endpoint == "reply":
            self.client.reply(ticket_id, payload)
        elif decision.endpoint == "handoff":
            self.client.handoff(ticket_id, payload)
        else:
            self.client.close_ticket(ticket_id, payload)

    def write_log(
        self,
        ticket_id: str,
        outcome: str,
        decision: Decision | None,
        detail: str = "",
        flags: str = "",
    ) -> None:
        self.logbook.write(
            LogEntry(
                ticket_id=ticket_id,
                outcome=outcome,
                decision=decision,
                detail=detail,
                flags=flags,
            )
        )


def describe_state(ledger: Ledger) -> str:
    counts = ledger.counts()
    if not counts:
        return "ledger is empty"
    return ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))


__all__ = ["INTENT", "Runner", "RunResult", "describe_state"]
