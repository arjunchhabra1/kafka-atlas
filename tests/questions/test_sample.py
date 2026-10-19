import subprocess
import sys
import textwrap
from datetime import timedelta

from psycopg.conninfo import make_conninfo

from kgrag.questions.sample import sample_evidence_sets
from kgrag.store.repo import get_documents

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
            d.doc_id if d.source_kind == "jira" else corpus.execute(
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
        assert (a in d1.text and b not in d1.text) or (b in d1.text and a not in d1.text)
        assert (a in d2.text) != (b in d2.text)
        assert (a in d1.text) != (a in d2.text)


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
