import pytest

from kgrag.store.db import apply_migrations


@pytest.mark.integration
def test_apply_migrations_idempotent(pg_conn):
    # Reset to an empty schema (the fixture already migrated once).
    pg_conn.execute("drop schema public cascade")
    pg_conn.execute("create schema public")
    pg_conn.commit()

    assert apply_migrations(pg_conn) == ["001_init.sql", "002_extraction.sql", "003_questions.sql",
                                       "004_resolutions.sql",
                                       "005_email_parent.sql"]
    assert apply_migrations(pg_conn) == []
    row = pg_conn.execute("select to_regclass('public.documents')").fetchone()
    assert row[0] == "documents"


@pytest.mark.integration
def test_migration_checksums_recorded_backfilled_and_drift_warned(pg_conn, tmp_path, monkeypatch,
                                                                   caplog):
    import hashlib
    import shutil

    from kgrag.store import db

    apply_migrations(pg_conn)
    rows = dict(pg_conn.execute("select name, checksum from schema_migrations").fetchall())
    first = sorted(db.MIGRATIONS_DIR.glob("*.sql"))[0]
    assert rows[first.name] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert all(rows.values())

    # rows from before checksums existed are backfilled, silently
    pg_conn.execute("update schema_migrations set checksum = null")
    pg_conn.commit()
    with caplog.at_level("WARNING", logger="kgrag.store.db"):
        assert apply_migrations(pg_conn) == []
    assert not caplog.records
    assert dict(pg_conn.execute("select name, checksum from schema_migrations").fetchall()) == rows

    # an applied migration edited afterwards: warn, do not fail or re-apply
    copy = tmp_path / "migrations"
    shutil.copytree(db.MIGRATIONS_DIR, copy)
    (copy / first.name).write_text(first.read_text() + "\n-- edited\n")
    monkeypatch.setattr(db, "MIGRATIONS_DIR", copy)
    with caplog.at_level("WARNING", logger="kgrag.store.db"):
        assert apply_migrations(pg_conn) == []
    assert any(first.name in r.getMessage() and "Recreate the database" in r.getMessage()
               for r in caplog.records)
