"""Interactive review of drafted questions: keep / edit / drop, saved after every keystroke."""
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import psycopg
import yaml
from rich.console import Console

from kgrag.questions.draft import NULL_ANSWER
from kgrag.store.repo import Document, QuestionRow, get_documents

DOC_CHARS = 1500
TYPE_ORDER = ("single_hop", "inference", "comparison", "temporal", "null")
KEYS = "k keep · e edit · d drop · s skip · q quit"

ReadKey = Callable[[], str]
Editor = Callable[[str], str | None]


def read_key() -> str:
    """One keystroke from the terminal (a line from stdin when it is not a TTY)."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q").strip()[:1].lower() or "q"
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "q" if ch in ("\x03", "\x04", "") else ch.lower()


def external_editor(text: str) -> str | None:
    """Edit text in $EDITOR (fallback vi); None when the editor fails."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "question.yaml"
        path.write_text(text)
        cmd = shlex.split(os.environ.get("EDITOR") or "vi")
        try:
            if subprocess.run([*cmd, str(path)]).returncode != 0:
                return None
        except OSError as exc:
            print(f"Could not run editor {cmd[0]!r}: {exc}", file=sys.stderr)
            return None
        return path.read_text()


def load_drafts(conn: psycopg.Connection, qtype: str | None = None) -> list[QuestionRow]:
    cols = ", ".join(QuestionRow.model_fields)
    rows = conn.execute(
        f"select {cols} from questions where status = 'draft' and (%(t)s::text is null or type = %(t)s) "
        "order by array_position(%(order)s::text[], type), id",
        {"t": qtype, "order": list(TYPE_ORDER)},
    ).fetchall()
    return [QuestionRow(**dict(zip(QuestionRow.model_fields, r, strict=True))) for r in rows]


def progress(conn: psycopg.Connection) -> dict[str, dict[str, int]]:
    out = {t: {"kept": 0, "edited": 0, "dropped": 0, "remaining": 0} for t in TYPE_ORDER}
    for qtype, status, n in conn.execute("select type, status, count(*) from questions group by 1, 2"):
        out[qtype]["remaining" if status == "draft" else status] = n
    return out


def validate_edit(conn: psycopg.Connection, q: QuestionRow, data: object) -> str | None:
    """Return a human-readable problem with the edited fields, or None when they are valid."""
    if not isinstance(data, dict):
        return "the file must be a YAML mapping with question, answer and gold_doc_ids"
    question, answer, gold = data.get("question"), data.get("answer"), data.get("gold_doc_ids")
    question = question.strip() if isinstance(question, str) else question
    answer = answer.strip() if isinstance(answer, str) else answer
    if not isinstance(question, str) or not question.strip():
        return "question must be non-empty"
    if not isinstance(answer, str) or not answer.strip():
        return "answer must be non-empty"
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        return "gold_doc_ids must be a list of doc ids"
    dupes = sorted({g for g in gold if gold.count(g) > 1})
    if dupes:
        return f"duplicate gold doc ids: {', '.join(dupes)}"
    if q.type == "single_hop" and len(gold) != 1:
        return "single_hop needs exactly 1 gold doc id"
    if q.type == "null" and gold:
        return "null questions need exactly 0 gold doc ids"
    if q.type not in ("single_hop", "null") and len(gold) < 2:
        return f"{q.type} needs at least 2 gold doc ids"
    if q.type == "null" and answer != NULL_ANSWER:
        return f"answer of a null question must stay: {NULL_ANSWER}"
    unknown = sorted(set(gold) - {d.doc_id for d in get_documents(conn, gold)})
    if unknown:
        return f"unknown doc ids: {', '.join(unknown)}"
    return None


def decide(conn: psycopg.Connection, qid: str, status: str, fields: dict | None = None) -> None:
    """Record a decision and commit immediately."""
    f = fields or {}
    conn.execute(
        "update questions set status = %s, reviewed_at = now(), "
        "question = coalesce(%s, question), answer = coalesce(%s, answer), "
        "gold_doc_ids = coalesce(%s, gold_doc_ids) where id = %s",
        (status, f.get("question"), f.get("answer"), f.get("gold_doc_ids"), qid),
    )
    conn.commit()


def _thread_subject(conn: psycopg.Connection, link: str | None) -> str | None:
    if not link or not link.startswith("thread:"):
        return None
    row = conn.execute("select subject from threads where thread_id = %s",
                       (link.removeprefix("thread:"),)).fetchone()
    return row[0] if row else None


def _doc_block(d: Document) -> str:
    text = d.text if len(d.text) <= DOC_CHARS else d.text[:DOC_CHARS] + f"\n... [truncated, {len(d.text)} chars]"
    return f"[{d.doc_id}] {d.date:%Y-%m-%d} · {d.sender} · {d.subject}\n{text}"


def render(conn: psycopg.Connection, q: QuestionRow, console: Console) -> None:
    def p(s: str = "") -> None:
        console.print(s, markup=False, highlight=False, emoji=False)

    p("\n" + "=" * 80)
    for t, c in progress(conn).items():
        p(f"{t:<11} kept {c['kept']:>3}  edited {c['edited']:>3}  dropped {c['dropped']:>3}  "
          f"remaining {c['remaining']:>3}")
    p("-" * 80)
    p(f"{q.type} · {q.origin or '?'} · {q.id}")
    p(f"Q: {q.question}")
    p(f"A: {q.answer}")
    if q.key_terms:
        p(f"key terms: {', '.join(q.key_terms)}")
    if q.type == "null":
        p(f"\nThread: {_thread_subject(conn, q.link) or '(unknown)'}")
    for d in get_documents(conn, q.gold_doc_ids):
        p("\n" + _doc_block(d))
    p("\n" + KEYS)


def _edit(
    conn: psycopg.Connection, q: QuestionRow, read: ReadKey, editor: Editor, console: Console
) -> bool:
    """Run the edit dialogue; True when a valid edit was saved."""
    text = yaml.safe_dump(
        {"question": q.question, "answer": q.answer, "gold_doc_ids": q.gold_doc_ids},
        sort_keys=False, allow_unicode=True,
    )
    while True:
        edited = editor(text)
        if edited is None:
            console.print("Editor did not save; edit cancelled.", markup=False, emoji=False)
            return False
        text = edited
        try:
            data = yaml.safe_load(text)
            error = validate_edit(conn, q, data)
        except yaml.YAMLError as exc:
            error = f"invalid YAML: {exc}"
        if error is None:
            fields = {"question": data["question"].strip(), "answer": data["answer"].strip(),
                      "gold_doc_ids": data["gold_doc_ids"]}
            if fields == {"question": q.question, "answer": q.answer, "gold_doc_ids": q.gold_doc_ids}:
                console.print("No changes; edit cancelled.", markup=False, emoji=False)
                return False
            decide(conn, q.id, "edited", fields)
            return True
        console.print(f"Invalid edit: {error}", markup=False, highlight=False, emoji=False)
        console.print("r re-open editor · c cancel", markup=False, emoji=False)
        while (k := read()) not in ("r", "c"):
            pass
        if k == "c":
            return False


def review_session(
    conn: psycopg.Connection,
    *,
    read_key: ReadKey = read_key,
    editor: Editor = external_editor,
    console: Console | None = None,
    qtype: str | None = None,
) -> None:
    console = console or Console()
    drafts = load_drafts(conn, qtype)
    if not drafts:
        console.print("No drafts left to review.", markup=False, emoji=False)
    for q in drafts:
        render(conn, q, console)
        while True:
            key = read_key()
            if key == "k":
                decide(conn, q.id, "kept")
            elif key == "d":
                decide(conn, q.id, "dropped")
            elif key == "e":
                if not _edit(conn, q, read_key, editor, console):
                    render(conn, q, console)
                    continue
            elif key == "q":
                console.print("Progress saved.", markup=False, emoji=False)
                return
            elif key != "s":
                continue
            break
    console.print("Session finished.", markup=False, emoji=False)
