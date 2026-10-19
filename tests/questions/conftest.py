from datetime import UTC, datetime

import pytest

from kgrag.extract.pipeline import persist_extractions
from kgrag.store import repo
from kgrag.store.repo import EmailRow
from kgrag.tokens import count_tokens

TABLES = (
    "evidence_attempts", "questions", "relations", "mentions", "llm_cache", "documents", "jira_events", "emails",
    "threads",
)
FILLER = (
    "This message goes through the design details of the broker protocol and the rollout plan "
    "across several releases, including compatibility notes, upgrade ordering and the testing "
    "that operators should perform before enabling the change in production clusters."
)


def day(month: int, d: int) -> datetime:
    return datetime(2026, month, d, tzinfo=UTC)


# (email id, thread, date, body). Bodies mention at most one ID except where noted.
EMAILS = [
    ("e1", "t1", day(1, 1), f"Proposal for KIP-100. {FILLER}"),
    ("e2", "t2", day(1, 20), f"Update on kip 100 after review. {FILLER}"),
    ("e3", "t3", day(2, 10), f"Vote result for KIP-100. {FILLER}"),
    ("e4", "t4", day(1, 5), f"Discussion of KIP-200 design. {FILLER}"),
    ("e5", "t5", day(1, 6), f"More on KIP-200 compatibility. {FILLER}"),
    ("e6", "t6", day(1, 7), f"Regression tracked in KAFKA-500. {FILLER}"),
    ("e7", "t7", day(1, 8), f"Flaky test KAFKA-600 again. {FILLER}"),
    ("e8", "t8", day(1, 9), f"Cleaning up KAFKA-700 follow-ups. {FILLER}"),
    ("e9", "t1", day(1, 2), f"Second note on KIP-100 in the same thread. {FILLER}"),
    ("e10", "t9", day(3, 1), "Short note."),
]
THREAD_SUBJECTS = {f"t{i}": f"[DISCUSS] topic {i}" for i in range(1, 10)}
JIRA = [  # (ticket, date)
    ("KAFKA-500", day(2, 20)),
    ("KAFKA-600", day(1, 9)),
]


@pytest.fixture
def corpus(pg_conn):
    pg_conn.execute(f"truncate {', '.join(TABLES)}")
    repo.upsert_threads(pg_conn, [(t, f"<{t}@x>", s) for t, s in THREAD_SUBJECTS.items()])
    rows = [
        EmailRow(
            id=eid, message_id=f"<{eid}@x>", thread_id=tid, date=date, sender_name="Dev",
            sender_address="dev@x.org", subject=THREAD_SUBJECTS[tid], body_clean=body,
            kind="human",
        )
        for eid, tid, date, body in EMAILS
    ]
    # e9 is split into two chunks that both mention KIP-100.
    repo.upsert_emails(pg_conn, rows)
    for i, (ticket, date) in enumerate(JIRA, start=1):
        pg_conn.execute(
            "insert into emails (id, message_id, thread_id, date, sender_name, sender_address, "
            "subject, body_clean, kind) values (%s, %s, 't1', %s, 'Jira', 'j@x.org', 's', 'b', "
            "'jira')", (f"j{i}", f"<j{i}@x>", date),
        )
        pg_conn.execute(
            "insert into jira_events (id, ticket, action, actor, date, source_email_id) "
            "values (%s, %s, 'commented', 'Jane Dev', %s, %s)", (i, ticket, date, f"j{i}"),
        )
    docs = []
    for eid, _tid, date, body in EMAILS:
        chunks = [body]
        if eid == "e9":
            chunks = [body, f"Continuing: KIP-100 needs one more round. {FILLER}"]
        for n, chunk in enumerate(chunks):
            docs.append((f"email:{eid}:{n}", "email", eid, n, chunk, count_tokens(chunk), date,
                         "Dev", THREAD_SUBJECTS[_tid]))
    for i, (ticket, date) in enumerate(JIRA, start=1):
        text = f"{ticket} commented by Jane Dev. {FILLER}"
        docs.append((f"jira:{i}", "jira", str(i), 0, text, count_tokens(text), date, "Jane Dev",
                     f"{ticket} title"))
    with pg_conn.cursor() as cur:
        cur.executemany(
            "insert into documents (doc_id, source_kind, source_id, chunk_index, text, "
            "token_count, date, sender, subject) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)", docs,
        )
    pg_conn.commit()
    persist_extractions(pg_conn)
    return pg_conn
