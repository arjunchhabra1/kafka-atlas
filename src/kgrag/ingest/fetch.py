"""Download monthly mbox archives of the Kafka dev@ list from lists.apache.org."""

import re
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

MBOX_URL = "https://lists.apache.org/api/mbox.lua"
LIST_ADDRESS = "dev@kafka.apache.org"
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def _parse_month(month: str) -> tuple[int, int]:
    m = _MONTH_RE.match(month)
    if not m:
        raise ValueError(f"month must be YYYY-MM, got {month!r}")
    return int(m.group(1)), int(m.group(2))


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    y, m = _parse_month(start)
    end_y, end_m = _parse_month(end)
    months = []
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _download(client: httpx.Client, month: str, target: Path, backoff: float) -> None:
    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=backoff),
        reraise=True,
    )
    def attempt() -> None:
        part = target.with_name(target.name + ".part")
        try:
            with client.stream(
                "GET", MBOX_URL, params={"list": LIST_ADDRESS, "date": month}
            ) as resp:
                resp.raise_for_status()
                with part.open("wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)

    attempt()


def fetch_month(
    month: str,
    dest_dir: Path,
    client: httpx.Client | None = None,
    *,
    backoff: float = 1.0,
) -> Path:
    """Fetch one month's mbox into dest_dir, skipping if already downloaded.

    Retries up to 3 attempts total on 5xx, timeouts and transport errors, with
    exponential backoff (`backoff` is the wait multiplier in seconds; use 0 in
    tests). 4xx errors raise immediately. The file is written atomically.
    """
    _parse_month(month)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"dev-kafka-{month}.mbox"
    if target.exists() and target.stat().st_size > 0:
        return target

    if client is not None:
        _download(client, month, target, backoff)
    else:
        with httpx.Client(timeout=120, follow_redirects=True) as own:
            _download(own, month, target, backoff)
    return target
