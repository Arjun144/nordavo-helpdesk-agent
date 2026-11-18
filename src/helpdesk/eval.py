"""Score against the ten labelled examples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .llm import Usage
from .models import Decision

FIELDS = ("action", "language", "category", "priority")


@dataclass(frozen=True, slots=True)
class Row:
    ticket_id: str
    subject: str
    expected: Mapping[str, str]
    actual: Decision
    matches: Mapping[str, bool]

    @property
    def exact(self) -> bool:
        return all(self.matches.values())

    @property
    def misses(self) -> list[str]:
        return [name for name, ok in self.matches.items() if not ok]


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    usage: Usage = Usage()

    @property
    def accuracy(self) -> dict[str, float]:
        if not self.rows:
            return {name: 0.0 for name in FIELDS}
        return {
            name: sum(row.matches[name] for row in self.rows) / len(self.rows)
            for name in FIELDS
        }

    @property
    def exact_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.exact for row in self.rows) / len(self.rows)

    @property
    def cost_per_100_tickets(self) -> float:
        if not self.rows:
            return 0.0
        return self.usage.cost_usd / len(self.rows) * 100

    def failures(self) -> list[Row]:
        return [row for row in self.rows if not row.exact]

    def summary(self) -> str:
        accuracy = self.accuracy
        lines = [
            f"examples: {len(self.rows)}",
            "  " + "  ".join(f"{name}={accuracy[name]:.0%}" for name in FIELDS),
            f"  all four fields: {self.exact_rate:.0%}",
            f"  tokens: {self.usage.prompt_tokens} in, {self.usage.completion_tokens} out",
            f"  cost: ${self.usage.cost_usd:.4f} for {len(self.rows)} tickets, "
            f"${self.cost_per_100_tickets:.2f} per 100",
        ]
        for row in self.failures():
            expected = ", ".join(f"{name}={row.expected[name]}" for name in row.misses)
            actual = ", ".join(
                f"{name}={getattr(row.actual, name).value}" for name in row.misses
            )
            lines.append(f"  {row.ticket_id} {row.subject[:34]}: {expected} -> {actual}")
        return "\n".join(lines)


def score(
    examples: Iterable[Mapping[str, Any]], decide: Any, usage: Usage = Usage()
) -> Report:
    report = Report(usage=usage)
    for example in examples:
        label = example["label"]
        decision = decide(example)
        report.rows.append(
            Row(
                ticket_id=str(example["id"]),
                subject=str(example.get("subject") or ""),
                expected={name: str(label[name]) for name in FIELDS},
                actual=decision,
                matches={
                    name: getattr(decision, name).value == str(label[name])
                    for name in FIELDS
                },
            )
        )
    return report
