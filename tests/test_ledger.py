from __future__ import annotations

from helpdesk.ledger import FAILED, INTENT, SENT, Ledger


def make_ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.db")


def test_second_claim_is_refused(tmp_path) -> None:
    ledger = make_ledger(tmp_path)

    assert ledger.claim("T001") is True
    assert ledger.claim("T001") is False
    assert ledger.state_of("T001") == INTENT


def test_claim_survives_a_new_ledger_instance(tmp_path) -> None:
    make_ledger(tmp_path).claim("T001")

    assert make_ledger(tmp_path).claim("T001") is False


def test_mark_and_counts(tmp_path) -> None:
    ledger = make_ledger(tmp_path)
    ledger.claim("T001")
    ledger.mark("T001", SENT, action="reply")
    ledger.claim("T002")

    assert ledger.counts() == {SENT: 1, INTENT: 1}
    assert ledger.state_of("T001") == SENT
    assert [ticket_id for ticket_id, _ in ledger.unfinished()] == ["T002"]


def test_unfinished_lists_only_unresolved_claims(tmp_path) -> None:
    ledger = make_ledger(tmp_path)
    ledger.claim("T001")
    ledger.claim("T002")
    ledger.mark("T002", FAILED, detail="boom")

    assert [ticket_id for ticket_id, _ in ledger.unfinished()] == ["T001"]


def test_release_allows_a_retry(tmp_path) -> None:
    ledger = make_ledger(tmp_path)
    ledger.claim("T001")
    ledger.release("T001")

    assert ledger.claim("T001") is True


def test_clear_forgets_everything(tmp_path) -> None:
    ledger = make_ledger(tmp_path)
    ledger.claim("T001")
    ledger.mark("T001", SENT)
    ledger.claim("T002")

    ledger.clear()

    assert ledger.counts() == {}
    assert ledger.claim("T001") is True


def test_reset_clears_the_log(tmp_path) -> None:
    from helpdesk.logbook import Logbook

    logbook = Logbook(tmp_path / "logs")
    assert logbook.path.exists() is False
    tmp_path.joinpath("logs").mkdir()
    logbook.path.write_text("old line\n")

    logbook.clear()

    assert logbook.path.exists() is False
