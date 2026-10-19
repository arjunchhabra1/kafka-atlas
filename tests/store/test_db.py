import pytest

from kgrag.store.db import apply_migrations


@pytest.mark.integration
def test_apply_migrations_idempotent(pg_conn):
    # Reset to an empty schema (the fixture already migrated once).
    pg_conn.execute("drop schema public cascade")
    pg_conn.execute("create schema public")
    pg_conn.commit()

    assert apply_migrations(pg_conn) == ["001_init.sql", "002_extraction.sql", "003_questions.sql"]
    assert apply_migrations(pg_conn) == []
    row = pg_conn.execute("select to_regclass('public.documents')").fetchone()
    assert row[0] == "documents"
