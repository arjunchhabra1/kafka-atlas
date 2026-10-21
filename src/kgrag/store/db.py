from pathlib import Path

import os
from threading import Lock
from typing import Any

import psycopg

from kgrag.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


_connector: Any = None
_connector_lock = Lock()


def _make_connector() -> Any:
    from google.cloud.sql.connector import Connector

    return Connector()


def _cloudsql_connect() -> psycopg.Connection:
    """Open a connection through the Cloud SQL Python Connector (psycopg driver)."""
    global _connector
    env = {}
    for name in ("INSTANCE_CONNECTION_NAME", "DB_USER", "DB_PASS", "DB_NAME"):
        if not os.environ.get(name):
            raise RuntimeError(f"KGRAG_DB_MODE=cloudsql requires the {name} environment variable")
        env[name] = os.environ[name]
    with _connector_lock:
        if _connector is None:
            _connector = _make_connector()
    return _connector.connect(
        env["INSTANCE_CONNECTION_NAME"], "psycopg",
        user=env["DB_USER"], password=env["DB_PASS"], db=env["DB_NAME"],
    )


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Default mode: `psycopg.connect(pg_dsn)`. `KGRAG_DB_MODE=cloudsql`: the Cloud SQL connector.

    An explicit `dsn` always wins (tests, tooling).
    """
    if dsn is None and get_settings().db_mode == "cloudsql":
        return _cloudsql_connect()
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
