"""Backfill emails.parent_message_id from the raw mbox files (all given/available months at once).

Parents can cross month boundaries, so pass every month in the database: by default all
`data/raw/dev-kafka-*.mbox`. Only parent_message_id is written; nothing else is touched.

    uv run python scripts/backfill_parents.py [mbox ...]
"""

import sys
from pathlib import Path

from kgrag.config import get_settings
from kgrag.ingest.parse import assign_parents, dedupe, parse_mbox
from kgrag.ingest.pipeline import email_id
from kgrag.store.db import apply_migrations, connect


def main(paths: list[Path]) -> None:
    conn = connect()
    apply_migrations(conn)
    raw = [e for p in paths for e in parse_mbox(p)]
    parents = assign_parents(dedupe(raw))
    with conn.cursor() as cur:
        cur.executemany("update emails set parent_message_id = %s where id = %s",
                        [(parent, email_id(mid)) for mid, parent in parents.items()])
    conn.commit()
    n = conn.execute("select count(parent_message_id) from emails").fetchone()[0]
    print(f"{len(parents)} parents computed; {n} emails now have parent_message_id")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or sorted((get_settings().data_dir / "raw").glob("dev-kafka-*.mbox"))
    main(args)
