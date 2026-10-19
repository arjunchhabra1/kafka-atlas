"""Strip quoting, signatures and bulky technical blocks from email bodies."""

import re

_ON_WROTE_START = re.compile(r"^On\s.+")
_EXC = re.compile(r"(?:[\w$]+\.)*[\w$]*(?:Exception|Error)\b")
_FRAME = re.compile(r"^\s+at\s+(?:[\w$.]+/)?([\w$.<>]+)\(")
_TRACE_CONT = re.compile(r"^\s*(?:\.\.\.\s+\d+\s+(?:more|common frames omitted)|Caused by:|Suppressed:)")
_DIFF_START = re.compile(r"^diff --git ")
_DIFF_LINE = re.compile(
    r"^(?:[ +\-\\]|@@|diff |index |new file mode|deleted file mode|similarity index|rename |Binary files)"
)
_LOG_LINE = re.compile(
    r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\s(?:INFO|WARN|ERROR|DEBUG)\s"
)
_LOG_MIN = 10


def _cut_on_wrote(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if _ON_WROTE_START.match(line) and any(
            l.rstrip().endswith("wrote:") for l in lines[i : i + 3]
        ):
            return lines[:i]
    return lines


def _cut_signature(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line == "-- ":
            return lines[:i]
    return lines


def _exception_name(line: str) -> str | None:
    matches = [m.group(0) for m in _EXC.finditer(line)]
    if not matches:
        return None
    best = next((m for m in matches if "." in m), matches[0])
    return best.rsplit(".", 1)[-1]


def _trace_end(lines: list[str], i: int) -> tuple[int, str] | None:
    """If a stack trace starts at line i return (end_index, summary)."""
    exc = _exception_name(lines[i])
    if exc is None or i + 1 >= len(lines):
        return None
    first = _FRAME.match(lines[i + 1])
    if not first:
        return None
    parts = first.group(1).split(".")
    cls = parts[-2] if len(parts) >= 2 else parts[0]
    j = i + 1
    while j < len(lines) and (_FRAME.match(lines[j]) or _TRACE_CONT.match(lines[j])):
        j += 1
    return j, f"[stack trace: {exc} in {cls}]"


def _diff_end(lines: list[str], i: int) -> int | None:
    start = lines[i]
    if _DIFF_START.match(start) or (
        start.startswith("--- a/") and i + 1 < len(lines) and lines[i + 1].startswith("+++ b/")
    ):
        j = i + 1
        while j < len(lines) and lines[j] and _DIFF_LINE.match(lines[j]):
            j += 1
        return j
    return None


def _log_end(lines: list[str], i: int) -> int | None:
    j = i
    while j < len(lines) and _LOG_LINE.search(lines[j]):
        j += 1
    return j if j - i >= _LOG_MIN else None


def _collapse_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if (trace := _trace_end(lines, i)) is not None:
            out.append(trace[1])
            i = trace[0]
        elif (end := _diff_end(lines, i)) is not None:
            out.append(f"[patch: {end - i} lines]")
            i = end
        elif (end := _log_end(lines, i)) is not None:
            out.append(f"[log: {end - i} lines]")
            i = end
        else:
            out.append(lines[i])
            i += 1
    return out


def clean_body(text: str) -> str:
    """Remove quoted replies, signatures; collapse traces, patches and logs."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = _cut_on_wrote(lines)
    lines = _cut_signature(lines)
    lines = [l for l in lines if not l.startswith(">")]
    lines = _collapse_blocks(lines)
    result = "\n".join(l.rstrip() for l in lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip()
