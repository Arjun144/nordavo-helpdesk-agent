from __future__ import annotations

import pytest
from pydantic import ValidationError

from helpdesk.models import (
    Action,
    Category,
    Decision,
    InvalidDecision,
    Language,
    Priority,
    parse_decision,
)

BASE = {
    "language": "nl",
    "category": "question",
    "priority": "P4",
    "action": "reply",
    "reason": "How-to question covered by F02. No urgency.",
    "draft_reply": "Hoi Anouk, ga naar Team -> Medewerker toevoegen.",
}


def test_reply_payload_carries_only_reply_fields() -> None:
    decision = Decision.model_validate(BASE)

    assert decision.endpoint == "reply"
    assert decision.payload() == {
        "language": "nl",
        "category": "question",
        "priority": "P4",
        "reason": "How-to question covered by F02. No urgency.",
        "draft_reply": "Hoi Anouk, ga naar Team -> Medewerker toevoegen.",
    }


def test_handoff_payload_carries_only_the_english_note() -> None:
    decision = Decision.model_validate(
        {
            "language": "nl",
            "category": "billing",
            "priority": "P2",
            "action": "handoff",
            "reason": "Refund request; client rule sends refunds to a human.",
            "handoff_note": "Customer asks for August refund while closed for renovation.",
            "confidence": 0.8,
            "faq_refs": ("F05",),
        }
    )

    assert decision.endpoint == "handoff"
    assert decision.payload() == {
        "language": "nl",
        "category": "billing",
        "priority": "P2",
        "reason": "Refund request; client rule sends refunds to a human.",
        "handoff_note": "Customer asks for August refund while closed for renovation.",
        "confidence": 0.8,
    }


def test_close_payload_is_minimal() -> None:
    decision = Decision.model_validate(
        {
            "language": "nl",
            "category": "other",
            "priority": "P4",
            "action": "close",
            "reason": "Empty body, nothing to answer.",
        }
    )

    assert decision.endpoint == "close"
    assert set(decision.payload()) == {"language", "category", "priority", "reason"}


def test_values_are_normalised_to_api_wire_format() -> None:
    decision = Decision.model_validate({**BASE, "priority": Priority.P1})
    assert decision.payload()["priority"] == "P1"
    assert decision.language is Language.NL
    assert decision.category is Category.QUESTION
    assert decision.action is Action.REPLY


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"draft_reply": None}, "draft_reply"),
        ({"handoff_note": "note"}, "handoff_note"),
        ({"action": "handoff"}, "handoff_note"),
        ({"action": "close"}, "close"),
        ({"action": "close", "draft_reply": "hi"}, "close"),
        ({"language": "fr"}, "language"),
        ({"category": "refund"}, "category"),
        ({"priority": "P5"}, "priority"),
        ({"confidence": 1.5}, "confidence"),
        ({"faq_refs": ("F13",)}, "F01..F12"),
        ({"reason": "   "}, "reason"),
        ({"reason": "x" * 401}, "reason"),
        ({"ticket_id": "T001"}, "ticket_id"),
    ],
)
def test_invalid_decisions_are_rejected(override: dict, expected: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Decision.model_validate({**BASE, **override})

    assert expected in str(excinfo.value)


def test_parse_decision_explains_what_is_wrong() -> None:
    with pytest.raises(InvalidDecision) as excinfo:
        parse_decision({"language": "nl", "action": "reply"})

    assert excinfo.value.errors
    assert "decision rejected" in str(excinfo.value)
    assert any("category" in problem for problem in excinfo.value.errors)


def test_reason_and_reply_are_trimmed() -> None:
    decision = Decision.model_validate(
        {**BASE, "reason": "  FAQ F02.  ", "draft_reply": "  Hoi.  "}
    )

    assert decision.reason == "FAQ F02."
    assert decision.draft_reply == "Hoi."


def test_empty_strings_for_inapplicable_fields_are_treated_as_absent() -> None:
    decision = Decision.model_validate({**BASE, "handoff_note": ""})

    assert decision.handoff_note is None
    assert decision.payload()["draft_reply"] == BASE["draft_reply"]
    assert "handoff_note" not in decision.payload()


def test_blank_draft_reply_on_a_reply_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="needs draft_reply"):
        Decision.model_validate({**BASE, "draft_reply": "   "})


def test_blank_handoff_note_on_a_handoff_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="needs handoff_note"):
        Decision.model_validate(
            {**BASE, "action": "handoff", "draft_reply": None, "handoff_note": ""}
        )
