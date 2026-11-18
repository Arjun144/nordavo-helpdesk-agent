"""Model plus policy: one ticket in, one decision out."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .llm import Usage
from .models import Decision, InvalidDecision, parse_decision
from .policy import apply_confidence_floor, apply_policy
from .prompt import build_system_prompt, build_user_prompt, load_faq


class Completion(Protocol):
    def complete_json(self, *, system: str, user: str) -> tuple[Mapping[str, Any], Usage]: ...


class Decider:
    def __init__(
        self,
        llm: Completion,
        *,
        faq_path: Path | str,
        confidence_floor: float = 0.5,
    ) -> None:
        self.llm = llm
        self.system = build_system_prompt(
            load_faq(faq_path), confidence_floor=confidence_floor
        )
        self.floor = confidence_floor
        self.usage = Usage()

    def decide(self, ticket: Mapping[str, Any]) -> Decision:
        raw, usage = self.llm.complete_json(
            system=self.system, user=build_user_prompt(ticket)
        )
        self.usage = self.usage + usage

        decision = parse_decision(raw)
        decision = apply_policy(dict(ticket), decision)
        return apply_confidence_floor(decision, floor=self.floor)


def decide_or_hand_off(decider: Decider, ticket: Mapping[str, Any], *, reason: str) -> Decision:
    """Fallback when the model output is unusable."""
    decision = Decision.model_validate(
        {
            "language": guess_language(ticket),
            "category": "other",
            "priority": "P3",
            "action": "handoff",
            "reason": reason,
            "handoff_note": (
                "The agent could not produce a valid decision for this ticket. "
                "It needs a human read."
            ),
        }
    )
    return apply_policy(dict(ticket), decision)


def guess_language(ticket: Mapping[str, Any]) -> str:
    """Rough guess, used only when handing off anyway."""
    text = f"{ticket.get('subject') or ''} {ticket.get('body') or ''}".lower()
    english = ("the ", " you ", "can't", "cannot", "please", "our ", "we ")
    german = (" sie ", " wir ", " nicht", " bitte", " und ", " ist ")
    if any(marker in f" {text} " for marker in english):
        return "en"
    if any(marker in f" {text} " for marker in german):
        return "de"
    return "nl"


__all__ = ["Decider", "InvalidDecision", "decide_or_hand_off"]
