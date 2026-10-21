import pytest

from kgrag.observability import langchain_callbacks, traced


def _traced_fn():
    @traced("demo")
    def f(x):
        return x + 1

    return f


def test_traced_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert _traced_fn()(1) == 2
    assert langchain_callbacks() == []


@pytest.fixture
def dummy_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-dummy")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")


def test_traced_with_keys_returns_value(dummy_keys):
    f = _traced_fn()
    assert f(1) == 2
    assert f(2) == 3  # cached observed wrapper reused


def test_callbacks_with_keys_returns_one_handler(dummy_keys):
    cbs = langchain_callbacks()
    assert isinstance(cbs, list) and len(cbs) == 1


def test_tag_current_trace_noop_without_keys(monkeypatch):
    from kgrag.observability import tag_current_trace

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tag_current_trace(system="vector", question_id="q1", run_id="r1")


def test_tag_current_trace_never_raises_with_keys(dummy_keys):
    from kgrag.observability import tag_current_trace, traced

    @traced("tagged")
    def f():
        tag_current_trace(system="vector", question_id="q1", run_id="r1")
        return 1

    assert f() == 1
