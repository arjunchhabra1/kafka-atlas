import os

import pytest

from kgrag.config import build_hash, get_settings, run_hash


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


def test_build_and_run_hashes_stable_and_scoped():
    from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GRAPH_ONLY_CONFIG

    s = get_settings()
    b, r = build_hash(s), run_hash(s)
    assert len(b) == len(r) == 12 and build_hash(s) == b and run_hash(s) == r
    # budget and answer/judge/embed models are run settings, not build settings
    for update in ({"budget_tokens": 1234},
                   {"models": s.models.model_copy(update={"answer": "other"})},
                   {"models": s.models.model_copy(update={"embed": "voyage-other"})}):
        c = s.model_copy(update=update)
        assert run_hash(c) != r and build_hash(c) == b, update
    # extraction model and chunking are build settings
    for update in ({"chunk_tokens": 400},
                   {"models": s.models.model_copy(update={"extract": "other"})},
                   {"prompt_versions": {"extract": "extract-v2"}}):
        c = s.model_copy(update=update)
        assert build_hash(c) != b and run_hash(c) == r, update
    assert run_hash(s.model_copy(update={"prompt_versions": {"judge": "judge-v9"}})) != r
    assert run_hash(s, DEFAULT_CONFIG) not in (r, run_hash(s, GRAPH_ONLY_CONFIG))


def test_hashes_ignore_secrets_dsns_pricing_and_guards():
    from kgrag.config import ModelPrice

    s = get_settings()
    b, r = build_hash(s), run_hash(s)
    for update in ({"neo4j_password": "s3cret"}, {"pg_dsn": "postgresql://u:pw@db:5432/x"},
                   {"neo4j_uri": "neo4j+s://x.databases.neo4j.io"}, {"cost_guard_usd": 99.0},
                   {"pricing": {"m": ModelPrice(input=1, output=2)}}, {"batch_discount": 1.0},
                   {"db_mode": "cloudsql"}):
        c = s.model_copy(update=update)
        assert build_hash(c) == b and run_hash(c) == r, update


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
