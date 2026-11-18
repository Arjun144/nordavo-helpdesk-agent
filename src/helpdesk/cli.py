"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .client import HelpdeskClient, HelpdeskError
from .config import Config, ConfigError
from .decider import Decider, decide_or_hand_off
from .eval import score
from .ledger import SENT, Ledger
from .llm import DeepSeekLLM, LLMError
from .logbook import Logbook
from .models import InvalidDecision
from .runner import Runner, describe_state


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ConfigError as error:
        print(f"config: {error}", file=sys.stderr)
        return 2

    try:
        return dispatch(args, config)
    except (HelpdeskError, LLMError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helpdesk",
        description="Read tickets from the helpdesk and post one decision per ticket.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="show the helpdesk state and the ledger")
    commands.add_parser("eval", help="score the ten labelled examples")

    run = commands.add_parser("run", help="handle open tickets")
    run.add_argument("--limit", type=int, default=None, help="stop after N tickets")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="decide without sending, and leave the ledger untouched",
    )

    commands.add_parser("reset", help="re-open all practice tickets")
    commands.add_parser("report", help="summarise the ledger and the log")

    return parser


def dispatch(args: argparse.Namespace, config: Config) -> int:
    if args.command == "status":
        return show_status(config)
    if args.command == "eval":
        return run_eval(config)
    if args.command == "run":
        return run_tickets(config, limit=args.limit, dry_run=args.dry_run)
    if args.command == "reset":
        return reset_practice(config)
    if args.command == "report":
        return show_report(config)
    raise AssertionError(f"unknown command {args.command}")


def show_status(config: Config) -> int:
    with HelpdeskClient(config) as client:
        state = client.status()
    ledger = Ledger(config.ledger_path)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"ledger: {describe_state(ledger)} ({ledger.path})")
    return 0


def run_eval(config: Config) -> int:
    decider = make_decider(config)
    with HelpdeskClient(config) as client:
        examples = client.examples()

    def decide(example: dict) -> object:
        try:
            return decider.decide(example)
        except InvalidDecision as error:
            return decide_or_hand_off(
                decider, example, reason=f"Model output rejected: {error}"
            )

    report = score(examples, decide)
    report.usage = decider.usage
    print(report.summary())
    return 0


def run_tickets(config: Config, *, limit: int | None, dry_run: bool) -> int:
    ledger = Ledger(config.ledger_path)
    decider = make_decider(config)

    with HelpdeskClient(config) as client:
        runner = Runner(client, decider, ledger, Logbook(config.logs_dir), dry_run=dry_run)
        result = runner.run(limit=limit)

    for ticket_id, outcome in result.per_ticket:
        print(f"{ticket_id}: {outcome}")
    print(
        f"\nsent={result.sent} skipped={result.skipped} failed={result.failed} "
        f"reconciled={result.reconciled}"
    )
    print(f"ledger: {describe_state(ledger)}")
    if not dry_run:
        print(f"log: {Path(config.logs_dir) / 'decisions.log'}")
    print(f"tokens: {decider.usage.total_tokens} (${decider.usage.cost_usd:.4f})")
    return 0 if result.failed == 0 else 1


def reset_practice(config: Config) -> int:
    ledger = Ledger(config.ledger_path)
    with HelpdeskClient(config) as client:
        response = client.reset_practice()
    ledger.clear()
    Logbook(config.logs_dir).clear()
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("ledger cleared and log removed; the next run handles all tickets again")
    return 0


def show_report(config: Config) -> int:
    ledger = Ledger(config.ledger_path)
    print(f"ledger: {describe_state(ledger)}")
    print(f"sent: {ledger.counts().get(SENT, 0)}")

    log = Path(config.logs_dir) / "decisions.log"
    if not log.exists():
        print("no log yet")
        return 0

    outcomes: dict[str, int] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            outcomes[parts[2]] = outcomes.get(parts[2], 0) + 1
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count}")
    return 0


def make_decider(config: Config) -> Decider:
    return Decider(
        DeepSeekLLM(config),
        faq_path=Path(config.docs_dir) / "faq.md",
        confidence_floor=config.confidence_floor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
