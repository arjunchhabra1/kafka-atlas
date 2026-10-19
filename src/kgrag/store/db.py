from pathlib import Path

import psycopg

from kgrag.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or get_settings().pg_dsn)


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply migrations/*.sql in name order; return the newly applied names."""
    conn.execute(
        "create table if not exists schema_migrations ("
        "name text primary key, applied_at timestamptz default now())"
    )
    conn.commit()
    done = {r[0] for r in conn.execute("select name from schema_migrations")}
    applied: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in done:
            continue
        conn.execute(path.read_text())
        conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        conn.commit()
        applied.append(path.name)
    return applied
