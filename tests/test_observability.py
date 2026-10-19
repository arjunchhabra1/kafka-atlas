from kgrag.observability import langchain_callbacks, traced


def test_traced_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    @traced("demo")
    def f(x):
        return x + 1

    assert f(1) == 2
    assert langchain_callbacks() == []
