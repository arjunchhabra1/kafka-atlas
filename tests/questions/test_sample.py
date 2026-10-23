import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta

import pytest
from psycopg.conninfo import make_conninfo

from kgrag.extract.regex_ids import extract_ids
from kgrag.extract.pipeline import persist_extractions
from kgrag.questions.sample import (
    HUMAN_SHARE,
    MAX_JIRA_BRIDGE_SHARE,
    _Doc,
    _inference,
    _load_docs,
    _temporal,
    _rng,
    _single_hop,
    is_noisy,
    related_pairs,
    revision_strength,
    sample_evidence_sets,
    temporal_pair_ok,
)
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
        n = 0
        for t in {ALL_TYPES!r}:
            n += len(sample_evidence_sets(conn, t, 5, 1)) if t == "comparison" else 0
            sample_evidence_sets(conn, t, 5, 1)
        assert n >= 1, "related-pair sampler built nothing"
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
        ident = es.link.removeprefix("bridge:")
        assert es.link.startswith("bridge:")
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


def test_inference_sets_are_bridges(corpus):
    """First doc cites the id inside another discussion; the rest are about the id itself."""
    sets = [es for seed in range(6) for es in sample_evidence_sets(corpus, "inference", 10, seed)]
    assert {es.link for es in sets} == {"bridge:KIP-100", "bridge:KAFKA-500", "bridge:KAFKA-600"}
    for es in sets:
        ident = es.link.removeprefix("bridge:")
        first, *rest = _docs(corpus, es)
        assert ident.lower() not in first.subject.lower()
        assert all(ident.lower() in d.subject.lower() for d in rest)
        assert ident in es.hint and "different discussion" in es.hint


def test_comparison_sets_two_distinct_ids(corpus):
    sets = sample_evidence_sets(corpus, "comparison", 10, 3)
    assert sets
    for es in sets:
        assert es.link.startswith("related:")
        a, b = es.link.removeprefix("related:").split("|")
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


def test_comparison_only_pairs_related_items(corpus):
    # KIP-100 / KIP-200 are mentioned together (e5); KAFKA-500 / KAFKA-600 are unrelated.
    sets = [es for seed in range(6) for es in sample_evidence_sets(corpus, "comparison", 10, seed)]
    assert sets and {es.link for es in sets} == {"related:KIP-100|KIP-200"}
    assert all("mentioned together" in es.hint for es in sets)
    assert not any("email:e5:0" in es.doc_ids for es in sets)  # it names both sides


def _doc(i, kind="email", title="", about=()):
    return _Doc(f"{kind}:{i}", kind, str(i), f"t{i}", datetime(2026, 1, 1, tzinfo=UTC),
                frozenset(about), frozenset(about), title)


def test_related_pairs_by_title_words_and_release():
    docs = [
        _doc(1, "jira", "KAFKA-1 ShareConsumer acknowledgement timeout", about=["KAFKA-1"]),
        _doc(2, "jira", "KAFKA-2 share consumer acknowledgement batching", about=["KAFKA-2"]),
        _doc(3, "jira", "KAFKA-3 RocksDB metrics recording", about=["KAFKA-3"]),
        _doc(4, "jira", "KAFKA-4 metrics for remote storage", about=["KAFKA-4"]),
        _doc(5, "jira", "KAFKA-5 metrics for the controller", about=["KAFKA-5"]),
        _doc(6, "email", "[DISCUSS] KIP-7: tiered storage quotas", about=["KIP-7"]),
    ]
    pairs = related_pairs(docs, {"KAFKA-3": "4.0.0", "KAFKA-4": "4.0.0", "KAFKA-5": "3.9.0"})
    assert pairs[("KAFKA-1", "KAFKA-2")].startswith("same topic")  # share, consumer, ack...
    assert pairs[("KAFKA-3", "KAFKA-4")].startswith("both fixed in 4.0.0")  # metrics + release
    assert ("KAFKA-3", "KAFKA-5") not in pairs and ("KAFKA-4", "KAFKA-5") not in pairs
    assert not any("KIP-7" in p for p in pairs)  # never pairs across kinds


def test_noisy_chunks_detected():
    assert is_noisy("[stack trace: TimeoutException in Foo]\n[log: 40 lines]\nIt failed again.")
    assert is_noisy("ducktape.errors.TimeoutError: worker3 timed out after 600 seconds\n"
                    "  at org.apache.kafka.Foo.bar(Foo.java:1)\nsee above")
    assert not is_noisy("I've run several ducktape builds.\nAll passed.\nThe RC looks good.\n"
                        "Docs are fine.\n+1 (binding)")


def test_noisy_chunks_never_sampled(corpus):
    pad = "word " * 50
    _add_email(corpus, "e12", "t11", [f"{pad}\n[stack trace: IOError in Log]\n[log: 80 lines]"])
    corpus.commit()
    assert "email:e12:0" not in {d.doc_id for d in _load_docs(corpus)}


def test_single_hop_jira_share_is_small():
    assert HUMAN_SHARE == 0.85
    docs = [_doc(i, "email") for i in range(300)] + [_doc(i, "jira") for i in range(300, 600)]
    picked = [es.doc_ids[0] for es, _ in zip(_single_hop(docs, _rng(1, "single_hop")), range(200))]
    share = sum(p.startswith("jira:") for p in picked) / len(picked)
    assert 0.08 <= share <= 0.22


def _d(name, kind, thread, when, about=(), cites=(), title="", words="", revision=0):
    return _Doc(f"{kind}:{name}", kind, name, thread, datetime(2026, 1, when, tzinfo=UTC),
                frozenset(about) | frozenset(cites), frozenset(about), title,
                frozenset(words.split()), revision)


def test_inference_prefers_email_bridges_and_caps_jira_endings():
    docs = []
    for i in range(12):
        ident = f"KIP-{i}"
        docs.append(_d(f"c{i}", "email", f"tc{i}", 1, cites=[ident], title="[DISCUSS] other"))
        docs.append(_d(f"j{i}", "jira", f"jira:{ident}", 5, about=[ident]))
        if i < 6:  # half the ids also have a later email thread about them
            docs.append(_d(f"a{i}", "email", f"ta{i}", 9, about=[ident],
                           title=f"[VOTE] {ident}"))
    sets = [es for seed in range(5) for es in _inference(docs, _rng(seed, "inference"))]
    assert sets
    for seed in range(5):
        run = list(_inference(docs, _rng(seed, "inference")))
        jira_end = [es for es in run if any(d.startswith("jira:") for d in es.doc_ids[1:])]
        assert len(jira_end) <= MAX_JIRA_BRIDGE_SHARE * len(run)
        assert jira_end  # still allowed, as a minority
        for es in run:
            n = int(es.link.removeprefix("bridge:KIP-"))
            if n < 6:  # an email about-doc exists: it is always used
                assert f"email:a{n}" in es.doc_ids


def test_inference_prefers_later_email_threads():
    docs = [_d("c", "email", "tc", 10, cites=["KIP-1"], title="[DISCUSS] other")]
    docs += [_d(f"early{i}", "email", f"te{i}", 2, about=["KIP-1"]) for i in range(3)]
    docs += [_d("late", "email", "tl", 20, about=["KIP-1"], title="[VOTE] KIP-1")]
    firsts = []
    for seed in range(20):
        es = next(_inference(docs, _rng(seed, "inference")))
        firsts.append("email:late" in es.doc_ids)
    assert sum(firsts) >= 16


@pytest.mark.parametrize("text", [
    "I've updated the KIP to use a single epoch.",
    "We have updated the wiki page.",
    "Updated the proposal after the call.",
    "Here is the REVISED design.",
    "Based on the feedback, the config is gone.",
    "I addressed the comments from Jun.",
    "The method was renamed to close().",
    "We dropped the second RPC.",
    "In the end we decided to keep the old format.",
    "I just changed the proposal accordingly.",
])
def test_revision_strength_strong_phrases(text):
    assert revision_strength(text, "[DISCUSS] KIP-9") == 2


def test_revision_strength_weak_phrases_only_in_replies():
    text = "The flag is no longer needed, use the timeout instead of polling."
    assert revision_strength(text, "Re: [DISCUSS] KIP-9") == 1
    assert revision_strength(text, "[DISCUSS] KIP-9") == 0
    # word-bounded: no match inside other words, and plain discussion is not a revision
    assert revision_strength("An unrevised, unrenamed draft; please review.", "Re: x") == 0
    assert revision_strength("What do you think about the design?", "Re: x") == 0


def _revision_docs():
    words = "broker fencing epoch controller quorum snapshot"
    return [
        _d("root", "email", "t1", 1, about=["KIP-9"], title="[DISCUSS] KIP-9: fencing",
           words=words),
        _d("chat", "email", "t1", 3, about=["KIP-9"], title="Re: [DISCUSS] KIP-9: fencing",
           words="question about epochs"),  # no announcement: never the later doc
        # revision reply in the same thread: paired with the thread's first email
        _d("rev", "email", "t1", 12, about=["KIP-9"], title="Re: [DISCUSS] KIP-9: fencing",
           words="fencing epoch moved metadata", revision=2),
        # new thread announcing a revision: paired with the earliest [DISCUSS] about KIP-9
        _d("v2", "email", "t2", 15, about=["KIP-9"], title="[DISCUSS] KIP-9 v2",
           words="renamed epoch field", revision=2),
        # a near-verbatim repost of the original: excluded
        _d("repost", "email", "t3", 18, about=["KIP-9"], title="[DISCUSS] KIP-9 again",
           words=words + " updated", revision=2),
        # summaries and Jira events never end (or start) a revision pair
        _d("result", "email", "t4", 25, about=["KIP-9"], title="[RESULT][VOTE] KIP-9",
           words="accepted updated", revision=2),
        _d("j", "jira", "jira:KIP-9", 26, about=["KIP-9"], revision=2),
        # KIP-7: only a [DISCUSS] email MENTIONING it (regex mention) exists before the revision
        _d("road", "email", "t5", 2, cites=["KIP-7"], title="[DISCUSS] Road map",
           words="plans releases"),
        _d("j7", "jira", "jira:KIP-7", 1, about=["KIP-7"]),
        _d("rev7", "email", "t6", 20, about=["KIP-7"], title="Re: KIP-7 status",
           words="dropped second request", revision=1),
        # too soon after the original (under a week): excluded
        _d("root8", "email", "t8", 1, about=["KIP-8"], title="[DISCUSS] KIP-8", words="x y"),
        _d("rev8", "email", "t8", 4, about=["KIP-8"], title="Re: [DISCUSS] KIP-8",
           words="updated", revision=2),
    ]


def test_temporal_revision_pairs_original_and_announcement():
    docs = _revision_docs()
    sets = [es for seed in range(20) for es in _temporal(docs, _rng(seed, "temporal"))]
    pairs = {tuple(es.doc_ids) for es in sets}
    assert pairs == {("email:root", "email:rev"), ("email:root", "email:v2"),
                     ("email:road", "email:rev7")}
    by_pair = {tuple(es.doc_ids): es for es in sets}
    assert by_pair[("email:road", "email:rev7")].link == "temporal:KIP-7"
    es = by_pair[("email:root", "email:rev")]
    assert es.type == "temporal" and es.link == "temporal:KIP-9"
    assert "original" in es.hint and "revision" in es.hint
    kinds = {d.doc_id: d.kind for d in docs}
    assert all(kinds[i] == "email" for es in sets for i in es.doc_ids)


def test_temporal_pair_ok_rules():
    by = {d.doc_id.split(":")[1]: d for d in _revision_docs()}
    assert temporal_pair_ok(by["root"], by["rev"])
    assert not temporal_pair_ok(by["root"], by["chat"])  # no announcement
    assert not temporal_pair_ok(by["root"], by["repost"])  # reposts the original
    assert not temporal_pair_ok(by["root"], by["result"])  # summary
    assert not temporal_pair_ok(by["root"], by["j"])  # Jira event
    assert not temporal_pair_ok(by["j7"], by["rev7"])  # Jira event as the original
    assert not temporal_pair_ok(by["root8"], by["rev8"])  # under a week apart


def test_temporal_revision_pairs_deterministic():
    docs = _revision_docs()
    for seed in range(5):
        assert (list(_temporal(docs, _rng(seed, "temporal")))
                == list(_temporal(docs, _rng(seed, "temporal"))))


def test_temporal_corpus_pairs_revision_with_original_proposal(corpus):
    sets = [es for seed in range(5) for es in sample_evidence_sets(corpus, "temporal", 10, seed)]
    assert {tuple(es.doc_ids) for es in sets} == {("email:e1:0", "email:e2:0")}
    assert all(es.link == "temporal:KIP-100" for es in sets)
