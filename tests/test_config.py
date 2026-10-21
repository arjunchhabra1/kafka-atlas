import os

import pytest

from kgrag.config import config_hash, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("KGRAG_"):
            monkeypatch.delenv(k)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_load_defaults():
    s = get_settings()
    assert s.budget_tokens == 4000
    assert s.chunk_tokens == 500
    assert s.models.judge == "claude-opus-5-5"


def test_config_hash_stable():
    s = get_settings()
    h = config_hash(s)
    assert len(h) == 12
    assert config_hash(s) == h
    changed = s.model_copy(update={"budget_tokens": 1234})
    assert config_hash(changed) != h


def test_env_overrides_yaml(monkeypatch):
    from kgrag.config import Settings

    monkeypatch.setenv("KGRAG_BUDGET_TOKENS", "999")
    monkeypatch.setenv("KGRAG_MODELS__JUDGE", "other-judge")
    s = Settings()
    assert s.budget_tokens == 999
    assert s.models.judge == "other-judge"


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    from kgrag.config import load_env

    env = tmp_path / ".env"
    env.write_text("KGRAG_TEST_DUMMY_KEY=from-file\nKGRAG_TEST_EXPORTED=from-file\n")
    monkeypatch.delenv("KGRAG_TEST_DUMMY_KEY", raising=False)
    monkeypatch.setenv("KGRAG_TEST_EXPORTED", "exported")
    load_env(env)
    import os

    assert os.environ["KGRAG_TEST_DUMMY_KEY"] == "from-file"
    assert os.environ["KGRAG_TEST_EXPORTED"] == "exported"
    monkeypatch.delenv("KGRAG_TEST_DUMMY_KEY")
