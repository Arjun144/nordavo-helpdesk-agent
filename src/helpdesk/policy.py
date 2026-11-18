"""Client rules, applied after the model and only ever tightening it."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import Action, Category, Decision, Priority

REDACTED = "[redacted]"

PRIORITY_RULES: dict[Priority, str] = {
    Priority.P1: "Many users blocked right now, or a security or legal risk.",
    Priority.P2: "One customer blocked, or money or the contract is involved.",
    Priority.P3: "Something is wrong or unclear, but there is a workaround.",
    Priority.P4: "No urgency: how-to questions, ideas, cosmetic issues.",
}

PRIORITY_ORDER = {Priority.P1: 1, Priority.P2: 2, Priority.P3: 3, Priority.P4: 4}


@dataclass(frozen=True, slots=True)
class HandoffRule:
    """A client rule, with the words that trigger it."""

    name: str
    note: str
    phrases: tuple[str, ...]
    category: Category | None = None
    min_priority: Priority = Priority.P2
    pairs: tuple[tuple[str, str], ...] = ()


HANDOFF_RULES: tuple[HandoffRule, ...] = (
    HandoffRule(
        name="refund",
        note=(
            "Customer asks about a refund or disputes a charge. Client rule: refunds and "
            "disputed charges always go to a human."
        ),
        phrases=(
            "terugbetaling",
            "terugbetalen",
            "terugkrijgen",
            "terug storten",
            "crediteren",
            "creditnota",
            "betwist",
            "onterecht afgeschreven",
            "dubbel afgeschreven",
            "refund",
            "chargeback",
            "disputed charge",
        ),
        category=Category.BILLING,
    ),
    HandoffRule(
        name="cancellation",
        note=(
            "Customer wants to cancel or change the contract. Client rule: cancellations and "
            "contract changes always go to a human."
        ),
        phrases=(
            "abonnement opzeggen",
            "abonnement annuleren",
            "contract opzeggen",
            "contract annuleren",
            "opzeggen",
            "opgezegd",
            "annulering van het abonnement",
            "contract wijzigen",
            "contract aanpassen",
            "cancel our subscription",
            "cancel my subscription",
            "terminate the contract",
        ),
        pairs=(
            ("annuleren", "abonnement"),
            ("annuleren", "contract"),
            ("opzeggen", "abonnement"),
            ("opzeggen", "contract"),
            ("cancel", "subscription"),
        ),
        category=Category.BILLING,
    ),
    HandoffRule(
        name="personal_data",
        note=(
            "Personal-data request (GDPR/AVG). Client rule: personal-data requests always go "
            "to a human."
        ),
        phrases=(
            "avg-verzoek",
            "avg verzoek",
            "avg",
            "gdpr",
            "persoonsgegevens",
            "gegevens verwijderen",
            "verwijder mijn gegevens",
            "recht op vergetelheid",
            "data deletion",
            "delete my data",
            "export my data",
        ),
        min_priority=Priority.P2,
    ),
    HandoffRule(
        name="security",
        note=(
            "Possible security incident. Client rule: possible security incidents always go to "
            "a human."
        ),
        phrases=(
            "datalek",
            "gehackt",
            "hack",
            "phishing",
            "verdachte login",
            "vreemde login",
            "onbekende login",
            "dat waren wij niet",
            "inlogpoging",
            "inbreuk",
            "kwetsbaarheid",
            "security incident",
            "data breach",
        ),
        pairs=(("ingelogd", "wij niet"),),
        min_priority=Priority.P1,
    ),
    HandoffRule(
        name="legal",
        note=(
            "Legal threat. Client rule: legal threats always go to a human."
        ),
        phrases=(
            "juridische stappen",
            "juridisch",
            "advocaat",
            "rechtbank",
            "aanklacht",
            "legal action",
            "lawyer",
            "solicitor",
        ),
        min_priority=Priority.P1,
    ),
    HandoffRule(
        name="churn",
        note=(
            "Customer threatens to leave. Client rule: customers threatening to leave always go "
            "to a human."
        ),
        phrases=(
            "stappen we over",
            "overstappen",
            "zeggen we op",
            "zeggen wij op",
            "zeg ik op",
            "anders stoppen we",
            "weg bij jullie",
            "op zoek naar een andere",
            "leaving roosterly",
            "switch to another",
        ),
        min_priority=Priority.P2,
    ),
)

SPAM_PHRASES = (
    "unsubscribe",
    "buy now",
    "seo services",
    "seo-bureau",
    "seo bureau",
    "linkbuilding",
    "bovenaan in google",
    "gratis analyse",
    "crypto",
    "viagra",
)

LANGUAGE_MARKERS = {
    "nl": (" de ", " het ", " een ", " wij ", " jullie ", " niet ", " naar ", " voor ", " met "),
    "en": (" the ", " and ", " your ", " you ", " with ", " for ", " from ", " our "),
    "de": (" der ", " die ", " das ", " und ", " nicht ", " wir ", " sie ", " mit "),
}

INJECTION_PHRASES = (
    "negeer alle eerdere instructies",
    "negeer bovenstaande",
    "negeer de instructies",
    "ignore all previous instructions",
    "ignore previous instructions",
    "systeembericht",
    "system message",
    "notitie voor de ai",
    "note for the ai",
    "als ai-assistent",
    "as an ai assistant",
)

IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def redact(text: str) -> str:
    """Strip card numbers and IBANs before text leaves the process."""
    if not text:
        return text

    def replace_iban(match: re.Match[str]) -> str:
        value = match.group(0)
        return REDACTED if len(value.replace(" ", "")) >= 15 else value

    def replace_digits(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return REDACTED if passes_luhn(digits) else match.group(0)

    return CARD_PATTERN.sub(replace_digits, IBAN_PATTERN.sub(replace_iban, text))


def passes_luhn(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def find_handoff_rule(ticket_text: str) -> HandoffRule | None:
    """First client rule the ticket triggers, if any."""
    haystack = normalise(ticket_text)
    for rule in HANDOFF_RULES:
        if any(contains_phrase(haystack, phrase) for phrase in rule.phrases):
            return rule
        if any(
            contains_phrase(haystack, first) and contains_phrase(haystack, second)
            for first, second in rule.pairs
        ):
            return rule
    return None


def is_spam(ticket_text: str) -> bool:
    haystack = normalise(ticket_text)
    return any(contains_phrase(haystack, phrase) for phrase in SPAM_PHRASES)


def contains_injection(ticket_text: str) -> bool:
    """Flag tickets that address the agent. They are still handled normally."""
    haystack = normalise(ticket_text)
    return any(contains_phrase(haystack, phrase) for phrase in INJECTION_PHRASES)


def wrong_language(text: str, language: str) -> bool:
    """True only when the text clearly reads as a different language."""
    padded = f" {' '.join(text.lower().split())} "
    own = sum(marker in padded for marker in LANGUAGE_MARKERS[language])
    others = max(
        (
            sum(marker in padded for marker in markers)
            for code, markers in LANGUAGE_MARKERS.items()
            if code != language
        ),
        default=0,
    )
    if own:
        return False
    return others >= 2 or (others == 1 and len(padded) > 45)


def is_empty(ticket_text: str) -> bool:
    return not ticket_text.strip()


def apply_policy(ticket: dict[str, Any], decision: Decision) -> Decision:
    """Tighten the model's decision and redact it."""
    text = ticket_text(ticket)
    body_empty = not body_of(ticket).strip()

    if body_empty or is_spam(text):
        return rebuild(
            decision,
            action=Action.CLOSE,
            draft_reply=None,
            handoff_note=None,
            reason=(
                "Empty ticket; closing without a reply."
                if body_empty
                else "Spam; closing without a reply."
            ),
        )

    rule = find_handoff_rule(text)
    tightened = decision

    if (
        decision.action is Action.REPLY
        and decision.draft_reply
        and wrong_language(decision.draft_reply, decision.language.value)
    ):
        # A customer writing Dutch who gets an English answer is the failure this is for.
        tightened = rebuild(
            decision,
            action=Action.HANDOFF,
            draft_reply=None,
            handoff_note=(
                "The drafted reply was not in the customer's language, so it was not sent. "
                "The ticket needs a human answer."
            ),
            reason=(
                f"Draft reply did not read as {decision.language.value}; handed off instead."
            ),
        )

    if decision.action is Action.REPLY and not decision.faq_refs and rule is None:
        tightened = rebuild(
            decision,
            action=Action.HANDOFF,
            draft_reply=None,
            handoff_note=(
                "The agent drafted a reply without citing any help-centre section, so it "
                "could not be grounded. The ticket needs a human answer or a new help-centre "
                "entry."
            ),
            reason=(
                "Draft answer cited no help-centre section, so it was not sent. "
                "Handed off instead."
            ),
        )

    if rule is not None and tightened.action is not Action.HANDOFF:
        tightened = rebuild(
            tightened,
            action=Action.HANDOFF,
            category=rule.category or decision.category,
            priority=stronger_priority(decision.priority, rule.min_priority),
            draft_reply=None,
            handoff_note=rule.note,
            reason=f"{rule.note} (policy rule: {rule.name})",
        )
    elif rule is not None:
        tightened = rebuild(
            tightened,
            category=rule.category or tightened.category,
            priority=stronger_priority(tightened.priority, rule.min_priority),
        )

    return rebuild(
        tightened,
        reason=redact(tightened.reason),
        draft_reply=redact(tightened.draft_reply) if tightened.draft_reply else None,
        handoff_note=redact(tightened.handoff_note) if tightened.handoff_note else None,
    )


def policy_flags(ticket: dict[str, Any], decision: Decision) -> list[str]:
    """Things worth recording about a ticket without changing the decision."""
    text = ticket_text(ticket)
    flags = []
    if contains_injection(text):
        flags.append("injection")
    if is_spam(text):
        flags.append("spam")
    if decision.handoff_note and wrong_language(decision.handoff_note, "en"):
        flags.append("handoff_note_not_english")
    return flags


def apply_confidence_floor(decision: Decision, *, floor: float) -> Decision:
    """Turn a low-confidence reply into a handoff, keeping the priority."""
    if decision.action is not Action.REPLY:
        return decision
    if decision.confidence is None or decision.confidence >= floor:
        return decision

    return rebuild(
        decision,
        action=Action.HANDOFF,
        draft_reply=None,
        handoff_note=(
            f"The agent's confidence in its own answer was {decision.confidence:.2f}, "
            f"below the {floor:.2f} floor, so the draft was not sent. The ticket needs a "
            "human answer."
        ),
        reason=(
            f"Low confidence ({decision.confidence:.2f} below {floor:.2f}); "
            "handed off instead of answering."
        ),
    )


def rebuild(decision: Decision, **overrides: Any) -> Decision:
    data = decision.model_dump()
    data.update(overrides)
    return Decision.model_validate(data)


def stronger_priority(current: Priority, minimum: Priority) -> Priority:
    return current if PRIORITY_ORDER[current] <= PRIORITY_ORDER[minimum] else minimum


def ticket_text(ticket: dict[str, Any]) -> str:
    parts: Iterable[str] = (
        str(ticket.get("subject") or ""),
        body_of(ticket),
    )
    return "\n".join(parts)


def body_of(ticket: dict[str, Any]) -> str:
    return str(ticket.get("body") or "")


def normalise(text: str) -> str:
    return text.lower()


def contains_phrase(haystack: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None
