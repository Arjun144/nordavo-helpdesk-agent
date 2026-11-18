from __future__ import annotations

from helpdesk.eval import score
from helpdesk.llm import Usage
from helpdesk.models import Decision


def decision(**overrides) -> Decision:
    data = {
        "language": "nl",
        "category": "question",
        "priority": "P4",
        "action": "reply",
        "reason": "F02.",
        "draft_reply": "Ga naar Team.",
    }
    data.update(overrides)
    return Decision.model_validate(data)


EXAMPLES = [
    {
        "id": "E04",
        "subject": "Dienst ruilen",
        "label": {"language": "nl", "category": "question", "priority": "P4", "action": "reply"},
    },
    {
        "id": "E03",
        "subject": "Terugbetaling",
        "label": {"language": "nl", "category": "billing", "priority": "P2", "action": "handoff"},
    },
]


def test_scores_each_field_against_the_client_labels() -> None:
    answers = {
        "E04": decision(),
        "E03": decision(
            action="handoff",
            handoff_note="Refund request.",
            draft_reply=None,
            category="billing",
            priority="P3",
        ),
    }

    report = score(EXAMPLES, lambda e: answers[e["id"]], usage=Usage(1000, 200))

    assert report.accuracy["action"] == 1.0
    assert report.accuracy["priority"] == 0.5
    assert report.exact_rate == 0.5
    assert [row.ticket_id for row in report.failures()] == ["E03"]
    assert report.failures()[0].misses == ["priority"]


def test_cost_per_100_is_extrapolated_from_the_sample() -> None:
    report = score(EXAMPLES, lambda e: decision(), usage=Usage(1_000_000, 0))

    assert report.cost_per_100_tickets == 13.5


def test_summary_names_the_misses() -> None:
    answers = {"E04": decision(), "E03": decision()}

    summary = score(EXAMPLES, lambda e: answers[e["id"]]).summary()

    assert "examples: 2" in summary
    assert "E03" in summary
