import subprocess
import sys
import textwrap
from datetime import timedelta

from psycopg.conninfo import make_conninfo

from kgrag.extract.regex_ids import extract_ids
from kgrag.extract.pipeline import persist_extractions
from kgrag.questions.sample import _load_docs, _rng, sample_evidence_sets
from kgrag.store.repo import get_documents
from kgrag.tokens import count_tokens

ALL_TYPES = ("single_hop", "inference", "comparison", "temporal", "null")


def _docs(conn, es):
    return get_documents(conn, es.doc_ids)


def test_sampler_does_not_use_graph(corpus):
    info = corpus.info
    dsn = make_conninfo(
        host=info.host, port=info.port, dbname=info.dbname, user=info.user,
        password=info.password,
    )
    code = textwrap.dedent(
        f"""
        import sys, psycopg
        from kgrag.questions.sample import sample_evidence_sets
        conn = psycopg.connect({dsn!r})
        for t in {ALL_TYPES!r}:
            sample_evidence_sets(conn, t, 5, 1)
        print(any(m == "kgrag.graph" or m.startswith("kgrag.graph.") for m in sys.modules))
        print("neo4j" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["False", "False"]


def test_sampler_deterministic_by_seed(corpus):
    for t in ALL_TYPES:
        assert sample_evidence_sets(corpus, t, 4, 7) == sample_evidence_sets(corpus, t, 4, 7)
    variants = {
        tuple(tuple(e.doc_ids) for e in sample_evidence_sets(corpus, "single_hop", 3, s))
        for s in range(6)
    }
    assert len(variants) > 1


def test_single_hop_one_substantial_doc(corpus):
    sets = sample_evidence_sets(corpus, "single_hop", 5, 1)
    assert len(sets) == 5
    for es in sets:
        (doc,) = _docs(corpus, es)
        assert doc.token_count >= 40
    assert len({es.doc_ids[0] for es in sets}) == 5
    assert not any(es.doc_ids == ["email:e10:0"] for es in sets)  # too short


def test_inference_sets_share_id_across_threads(corpus):
    sets = sample_evidence_sets(corpus, "inference", 10, 3)
    assert sets
    for es in sets:
        assert es.type == "inference"
        assert 2 <= len(es.doc_ids) <= 3
        ident = es.link.removeprefix("shared:")
        assert es.link.startswith("shared:")
        docs = _docs(corpus, es)
        assert all(ident.lower().replace("-", " ") in d.text.lower().replace("-", " ")
                   for d in docs)
        threads = {
            "jira:" + corpus.execute(
                "select ticket from jira_events where id = %s", (int(d.source_id),)
            ).fetchone()[0] if d.source_kind == "jira" else corpus.execute(
                "select thread_id from emails where id = %s", (d.source_id,)
            ).fetchone()[0]
            for d in docs
        }
        assert len(threads) >= 2


def test_comparison_sets_two_distinct_ids(corpus):
    sets = sample_evidence_sets(corpus, "comparison", 10, 3)
    assert sets
    for es in sets:
        assert es.link.startswith("compare:")
        a, b = es.link.removeprefix("compare:").split("|")
        assert a != b
        assert a.split("-")[0] == b.split("-")[0]  # same kind
        d1, d2 = _docs(corpus, es)
        ids1 = {h.id for h in extract_ids(d1.text)}
        ids2 = {h.id for h in extract_ids(d2.text)}
        assert ids1 & {a, b} and ids2 & {a, b}
        assert len(ids1 & {a, b}) == 1 and len(ids2 & {a, b}) == 1
        assert ids1 & {a, b} != ids2 & {a, b}


def test_temporal_sets_span_a_week(corpus):
    sets = sample_evidence_sets(corpus, "temporal", 10, 3)
    assert sets
    for es in sets:
        assert es.link.startswith("temporal:")
        dates = [d.date for d in _docs(corpus, es)]
        assert max(dates) - min(dates) >= timedelta(days=7)


def test_null_sets_have_no_docs(corpus):
    sets = sample_evidence_sets(corpus, "null", 4, 3)
    assert len(sets) == 4
    for es in sets:
        assert es.doc_ids == []
        assert es.link.startswith("thread:")
    assert len({es.link for es in sets}) == 4


def test_no_set_contains_two_chunks_of_one_email(corpus):
    for t in ALL_TYPES:
        for seed in range(8):
            for es in sample_evidence_sets(corpus, t, 10, seed):
                emails = [d for d in es.doc_ids if d.startswith("email:")]
                sources = [d.rsplit(":", 1)[0] for d in emails]
                assert len(sources) == len(set(sources)), es


def test_types_use_independent_random_streams():
    assert _rng(1, "inference").random() != _rng(1, "temporal").random()
    assert _rng(1, "inference").random() == _rng(1, "inference").random()


def _add_email(conn, eid, thread, chunks):
    conn.execute("insert into threads values (%s, %s, 'subj') on conflict do nothing",
                 (thread, f"<{thread}@x>"))
    conn.execute(
        "insert into emails (id, message_id, thread_id, date, sender_name, sender_address, "
        "subject, body_clean, kind) values (%s, %s, %s, now(), 'D', 'd@x', 's', %s, 'human')",
        (eid, f"<{eid}@x>", thread, "\n\n".join(chunks)),
    )
    for n, c in enumerate(chunks):
        conn.execute(
            "insert into documents values (%s, 'email', %s, %s, %s, %s, now(), 'D', 's')",
            (f"email:{eid}:{n}", eid, n, c, count_tokens(c)),
        )


def test_surface_match_respects_word_boundaries(corpus):
    pad = "word " * 60
    _add_email(corpus, "e11", "t10", [f"{pad} KAFKA-7000 here", f"{pad} KAFKA-700 here"])
    corpus.commit()
    persist_extractions(corpus)
    ids = {d.doc_id: d.ids for d in _load_docs(corpus)}
    assert ids["email:e11:0"] == {"KAFKA-7000"}
    assert ids["email:e11:1"] == {"KAFKA-700"}


def test_events_of_one_ticket_are_one_thread(corpus):
    corpus.execute(
        "insert into emails (id, message_id, thread_id, date, sender_name, sender_address, "
        "subject, body_clean, kind) values ('j3', '<j3@x>', 't1', now(), 'J', 'j@x', 's', 'b', "
        "'jira')"
    )
    corpus.execute(
        "insert into jira_events (id, ticket, action, actor, date, source_email_id) "
        "values (3, 'KAFKA-500', 'commented', 'Jane Dev', now(), 'j3')"
    )
    text = "KAFKA-500 commented again. " + "word " * 60
    corpus.execute(
        "insert into documents values ('jira:3', 'jira', '3', 0, %s, %s, now(), 'J', 's')",
        (text, count_tokens(text)),
    )
    corpus.commit()
    persist_extractions(corpus)
    for seed in range(15):
        for es in sample_evidence_sets(corpus, "inference", 10, seed):
            assert not {"jira:1", "jira:3"} <= set(es.doc_ids), es


def test_iter_exhausts_and_never_repeats(corpus):
    from kgrag.questions.sample import iter_evidence_sets

    for t in ALL_TYPES:
        sets = list(iter_evidence_sets(corpus, t, 4))
        keys = [frozenset(e.doc_ids) or e.link for e in sets]
        assert len(keys) == len(set(keys)) and sets
