import pytest

from kgrag.config import get_settings
from kgrag.retrieve import index
from kgrag.store import db


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setattr(db, "_connector", None)
    yield
    get_settings.cache_clear()


class FakeConnector:
    def __init__(self):
        self.calls = []

    def connect(self, instance, driver, **kw):
        self.calls.append((instance, driver, kw))
        return "cloud-conn"


def _cloud_env(monkeypatch):
    monkeypatch.setenv("KGRAG_DB_MODE", "cloudsql")
    monkeypatch.setenv("INSTANCE_CONNECTION_NAME", "proj:region:inst")
    monkeypatch.setenv("DB_USER", "app")
    monkeypatch.setenv("DB_PASS", "s3cret")
    monkeypatch.setenv("DB_NAME", "kgrag")


def test_cloudsql_mode_uses_connector(monkeypatch):
    _cloud_env(monkeypatch)
    fake = FakeConnector()
    monkeypatch.setattr(db, "_make_connector", lambda: fake)
    monkeypatch.setattr(db.psycopg, "connect", lambda *a, **k: pytest.fail("must not use the DSN"))

    assert db.connect() == "cloud-conn"
    assert fake.calls == [("proj:region:inst", "psycopg",
                           {"user": "app", "password": "s3cret", "db": "kgrag"})]
    db.connect()
    assert len(fake.calls) == 2  # connector is created once, reused


def test_cloudsql_mode_missing_env_names_the_variable(monkeypatch):
    _cloud_env(monkeypatch)
    monkeypatch.delenv("DB_PASS")
    monkeypatch.setattr(db, "_make_connector", lambda: FakeConnector())
    with pytest.raises(RuntimeError, match="DB_PASS"):
        db.connect()


def test_default_mode_uses_dsn(monkeypatch):
    monkeypatch.setenv("KGRAG_PG_DSN", "postgresql://u:p@h:1/d")
    seen = []
    monkeypatch.setattr(db.psycopg, "connect", lambda dsn: seen.append(dsn) or "dsn-conn")
    monkeypatch.setattr(db, "_make_connector", lambda: pytest.fail("no connector in default mode"))

    assert db.connect() == "dsn-conn"
    assert seen == ["postgresql://u:p@h:1/d"]


def test_explicit_dsn_wins_even_in_cloudsql_mode(monkeypatch):
    _cloud_env(monkeypatch)
    monkeypatch.setattr(db.psycopg, "connect", lambda dsn: f"explicit:{dsn}")
    assert db.connect("postgresql://x") == "explicit:postgresql://x"


def test_vector_store_connection_default_is_dsn_string(monkeypatch):
    monkeypatch.setenv("KGRAG_PG_DSN", "postgresql://u:p@h:1/d")
    assert index.store_connection(None) == "postgresql+psycopg://u:p@h:1/d"


def test_vector_store_connection_cloudsql_is_engine_using_connect(monkeypatch):
    _cloud_env(monkeypatch)
    monkeypatch.setattr(db, "connect", lambda: "cloud-conn")
    engine = index.store_connection(None)
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.pool._creator() == "cloud-conn"  # SQLAlchemy opens connections via db.connect
    finally:
        engine.dispose()
