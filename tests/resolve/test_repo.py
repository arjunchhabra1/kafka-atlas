import pytest

from kgrag.resolve.interface import Resolution
from kgrag.store.repo import load_resolution, resolution_hash, save_resolution


def _res(tag: str) -> Resolution:
    return Resolution(
        assignments={"e1:m1": f"Person:{tag}", "e2:m1": f"Person:{tag}"},
        canonical_names={f"Person:{tag}": tag},
        entity_types={f"Person:{tag}": "Person"},
    )


@pytest.mark.integration
def test_save_load_roundtrip_and_replace(pg_conn):
    save_resolution(pg_conn, "t", _res("a"))
    assert load_resolution(pg_conn, "t") == _res("a")
    save_resolution(pg_conn, "other", _res("z"))
    save_resolution(pg_conn, "t", _res("b"))
    assert load_resolution(pg_conn, "t") == _res("b")
    assert load_resolution(pg_conn, "other") == _res("z")
    assert load_resolution(pg_conn, "missing").assignments == {}


@pytest.mark.integration
def test_resolution_hash_tracks_content(pg_conn):
    assert resolution_hash(pg_conn, "hash-missing") is None
    save_resolution(pg_conn, "h", _res("a"))
    first = resolution_hash(pg_conn, "h")
    assert first and len(first) == 16 and resolution_hash(pg_conn, "h") == first
    save_resolution(pg_conn, "h", _res("b"))
    assert resolution_hash(pg_conn, "h") != first
    save_resolution(pg_conn, "h", _res("a"))
    assert resolution_hash(pg_conn, "h") == first
    pg_conn.commit()
