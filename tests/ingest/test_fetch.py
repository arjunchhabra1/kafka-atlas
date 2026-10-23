import httpx
import pytest
import respx

from kgrag.ingest.fetch import fetch_month, month_range

URL = "https://lists.apache.org/api/mbox.lua"
PARAMS = {"list": "dev@kafka.apache.org", "date": "2023-11"}


def test_month_range():
    assert month_range("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_range_single_and_empty():
    assert month_range("2024-01", "2024-01") == ["2024-01"]
    assert month_range("2024-02", "2024-01") == []


def test_invalid_month_rejected(tmp_path):
    with pytest.raises(ValueError):
        fetch_month("2023-13", tmp_path)
    with pytest.raises(ValueError):
        fetch_month("2023/11", tmp_path)
    with pytest.raises(ValueError):
        month_range("nope", "2024-01")


@respx.mock
def test_fetch_writes_file(tmp_path):
    route = respx.get(URL, params=PARAMS).mock(return_value=httpx.Response(200, content=b"From a\n"))
    dest = tmp_path / "nested" / "mbox"
    path = fetch_month("2023-11", dest, backoff=0)
    assert path == dest / "dev-kafka-2023-11.mbox"
    assert path.read_bytes() == b"From a\n"
    assert not list(dest.glob("*.part"))
    assert route.call_count == 1


@respx.mock
def test_fetch_skips_existing(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"new"))
    existing = tmp_path / "dev-kafka-2023-11.mbox"
    existing.write_bytes(b"old")
    assert fetch_month("2023-11", tmp_path, backoff=0) == existing
    assert existing.read_bytes() == b"old"
    assert route.call_count == 0


@respx.mock
def test_fetch_redownloads_empty_file(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"data"))
    (tmp_path / "dev-kafka-2023-11.mbox").write_bytes(b"")
    assert fetch_month("2023-11", tmp_path, backoff=0).read_bytes() == b"data"


@respx.mock
def test_fetch_retries_then_succeeds(tmp_path):
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, content=b"ok")]
    )
    path = fetch_month("2023-11", tmp_path, backoff=0)
    assert path.read_bytes() == b"ok"
    assert route.call_count == 3


@respx.mock
def test_fetch_retries_on_timeout(tmp_path):
    route = respx.get(URL).mock(side_effect=[httpx.ConnectTimeout("t"), httpx.Response(200, content=b"ok")])
    assert fetch_month("2023-11", tmp_path, backoff=0).read_bytes() == b"ok"
    assert route.call_count == 2


@respx.mock
def test_fetch_gives_up_after_three_attempts(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_month("2023-11", tmp_path, backoff=0)
    assert route.call_count == 3
    assert list(tmp_path.iterdir()) == []


@respx.mock
def test_fetch_4xx_raises_immediately(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_month("2023-11", tmp_path, backoff=0)
    assert route.call_count == 1
    assert list(tmp_path.iterdir()) == []


@respx.mock
def test_fetch_uses_supplied_client(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"x"))
    with httpx.Client() as client:
        fetch_month("2023-11", tmp_path, client=client, backoff=0)
        assert not client.is_closed
