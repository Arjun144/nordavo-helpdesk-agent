"""The decision that gets sent, and its validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

FAQ_SECTION_PATTERN = re.compile(r"^F(0[1-9]|1[0-2])$")

ENDPOINT_BY_ACTION = {
    "reply": "reply",
    "handoff": "handoff",
    "close": "close",
}


class InvalidDecision(ValueError):
    def __init__(self, message: str, *, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


class Language(StrEnum):
    NL = "nl"
    EN = "en"
    DE = "de"


class Category(StrEnum):
    BUG = "bug"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    QUESTION = "question"
    OTHER = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Action(StrEnum):
    REPLY = "reply"
    HANDOFF = "handoff"
    CLOSE = "close"


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    language: Language
    category: Category
    priority: Priority
    action: Action
    reason: str = Field(min_length=1, max_length=400)
    draft_reply: str | None = None
    handoff_note: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    faq_refs: tuple[str, ...] = ()

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("draft_reply", "handoff_note", mode="before")
    @classmethod
    def blank_means_absent(cls, value: object) -> object:
        # Models send "" for the field that does not apply. Treat it as missing.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("draft_reply", "handoff_note")
    @classmethod
    def trim(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("faq_refs")
    @classmethod
    def only_known_sections(cls, refs: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [ref for ref in refs if not FAQ_SECTION_PATTERN.match(ref)]
        if unknown:
            raise ValueError(
                f"unknown help-centre section(s) {unknown}; expected F01..F12"
            )
        return refs

    @model_validator(mode="after")
    def payload_matches_action(self) -> Decision:
        if self.action is Action.REPLY:
            if not self.draft_reply:
                raise ValueError("a reply needs draft_reply")
            if self.handoff_note:
                raise ValueError("a reply must not carry a handoff_note")
        elif self.action is Action.HANDOFF:
            if not self.handoff_note:
                raise ValueError("a handoff needs handoff_note, written in English")
            if self.draft_reply:
                raise ValueError("a handoff must not carry a draft_reply")
        else:
            if self.draft_reply or self.handoff_note:
                raise ValueError("a close carries neither a reply nor a handoff note")
        return self

    @property
    def endpoint(self) -> str:
        return ENDPOINT_BY_ACTION[self.action.value]

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "language": self.language.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "reason": self.reason,
        }
        if self.action is Action.REPLY:
            body["draft_reply"] = self.draft_reply
        elif self.action is Action.HANDOFF:
            body["handoff_note"] = self.handoff_note
        if self.confidence is not None:
            body["confidence"] = round(self.confidence, 3)
        return body


def parse_decision(raw: Mapping[str, Any]) -> Decision:
    try:
        return Decision.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or 'decision'}: {error['msg']}"
            for error in exc.errors()
        ]
        raise InvalidDecision(
            "decision rejected: " + "; ".join(problems), errors=problems
        ) from exc
