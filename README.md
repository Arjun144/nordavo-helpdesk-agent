# Roosterly helpdesk agent

Reads the open tickets from the client's helpdesk and posts one decision each. Scores 10/10 on the
labelled examples, and the final clean run over the thirty practice tickets is done: 12 replies, 16
handoffs, 2 closes, all thirty recorded by the helpdesk.

## How to run

```
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
$EDITOR .env                    # HELPDESK_API_KEY and DEEPSEEK_API_KEY
./.venv/bin/python -m pytest
./.venv/bin/python -m helpdesk.cli eval
./.venv/bin/python -m helpdesk.cli run --dry-run --limit 3
./.venv/bin/python -m helpdesk.cli run
./.venv/bin/python -m helpdesk.cli reset
```

Settings come from `.env`, which is gitignored. `--dry-run` sends nothing and leaves no claim; a real
run is resumable.

## Decisions

Python 3.11, httpx, pydantic, DeepSeek at temperature 0 in JSON mode, which keeps the assignment at
cents. `client.py` is transport only, `decider.py` turns a ticket into a decision, and `policy.py`
runs after the model and can only tighten it. Refunds, cancellations, AVG requests, security
incidents, legal threats and customers threatening to leave always go to a human, and card numbers
and IBANs are stripped from anything that leaves the process.

Below a confidence floor of 0.50 a reply becomes a handoff, keeping its priority. I chose that number
rather than measuring it, since the brief leaves "when is the agent unsure" open. Two rules are narrow
on purpose: the Dutch word for cancel alone kept catching people cancelling a shift swap, and
"contract" kept catching contract hours, so both look for a pair of words in any order.

I do not read Dutch. The help centre goes in the prompt and the model must name the F-section it used,
so "the answer is F02" is checkable without reading the reply. A draft that does not read as the
customer's language is rejected on stopword counts. That catches an English reply to a Dutch customer
but says nothing about grammar.

## Reliability

503 is retried with backoff capped at 8 seconds, 429 waits for exactly `Retry-After`, and calls are
spaced at 4 per second so it rarely comes to that. 400, 401, 404 and 422 fail at once. A 409 is
recorded as handled and never retried, because the customer already has an answer.

A ticket is claimed in SQLite before anything is sent, so a second run skips it. A claim left by a
crash is settled next run by asking `GET /tickets/{id}`: handled, the claim closes; not handled, it is
released and retried. A failed decision releases the claim, since nothing went out. A failed send
keeps it, since the request may have landed.

`reset` clears the ledger and the log, because the helpdesk has re-opened the tickets and a claim
would otherwise skip them. One redacted line per ticket goes to `logs/decisions.log`, including a note
when a ticket tried to instruct the agent.

## Eval and fix

`eval` scores action, language, category and priority against the ten client labels, exact match
only, plus cost per 100 tickets. The model never writes the labels.

The first run scored 30%. Six of ten came back as `other`/`P3`/`handoff`, the fallback in
`decider.py`, so they were rejected before reaching the policy. DeepSeek sends `"handoff_note": ""` on
a reply instead of omitting it, and I read a blank string as a blank field, discarding good answers.
Blank now means absent.

That gave 60%. The rest were judgement: two money questions filed as `question` instead of `billing`,
an old phone rated P2 instead of P3, and my policy forcing P1 on a GDPR request the client labels P2.
The prompt now defines the categories and says a plain how-to is P4 even when the help centre answers
it.

| | before | after |
|---|---|---|
| action | 40% | 100% |
| language | 90% | 100% |
| category | 50% | 100% |
| priority | 40% | 100% |
| all four | 30% | 100% |

Both runs are in `logs/`. Ten examples is ten and the prompt was tuned after seeing them, so this says
the labels are understood, not that thirty unseen tickets will all be right.

Cost is $0.0067 for ten tickets, so **$0.07 per 100**: about 1,900 prompt and 140 completion tokens
each, at $0.27 and $1.10 per million.

## What I would improve

Write the eval before the prompt. Doing it the other way makes it too easy to tune against my own
guesses instead of the client's labels. German replies also reuse the Dutch menu names from F01–F12,
so a German customer gets told to look under "Rooster → Verzoeken"; a German help centre would fix
that at the source.

One thing in the brief I would change: it wants logs in the repo and care with what customers write. A
log line that is useful and one that is safe are not the same line, and it does not say which wins.
