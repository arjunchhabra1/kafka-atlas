import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import psycopg

from kgrag.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
log = logging.getLogger(__name__)


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


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply migrations/*.sql in name order; return the newly applied names.

    Each applied migration's sha256 is recorded (rows from before checksums were recorded are
    backfilled from the current file). If an applied migration's file has changed since, this
    warns (it never re-applies or fails): the database schema no longer matches the code.
    """
    conn.execute(
        "create table if not exists schema_migrations ("
        "name text primary key, applied_at timestamptz default now())"
    )
    conn.execute("alter table schema_migrations add column if not exists checksum text")
    conn.commit()
    done = dict(conn.execute("select name, checksum from schema_migrations").fetchall())
    applied: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        digest = _checksum(path)
        if path.name in done:
            if done[path.name] is None:
                conn.execute("update schema_migrations set checksum = %s where name = %s",
                             (digest, path.name))
            elif done[path.name] != digest:
                log.warning(
                    "migration %s changed after it was applied to this database; its schema may "
                    "not match the code. Recreate the database (e.g. `docker compose down -v`, "
                    "then re-run the pipeline) rather than editing applied migrations.", path.name)
            continue
        conn.execute(path.read_text())
        conn.execute("insert into schema_migrations (name, checksum) values (%s, %s)",
                     (path.name, digest))
        conn.commit()
        applied.append(path.name)
    conn.commit()
    return applied
