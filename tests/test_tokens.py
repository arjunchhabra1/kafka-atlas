from kgrag.tokens import count_tokens


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
