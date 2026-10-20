import pytest

from kgrag.resolve.interface import Resolution
from kgrag.store.repo import load_resolution, save_resolution


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
