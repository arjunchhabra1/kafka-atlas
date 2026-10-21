# Dev fixture

A small, hand-labelled slice of the dev@kafka.apache.org archive. Entity-resolution and
retrieval tests grade against it, and `tests/test_fixture_integrity.py` checks that it holds together.

| File | What it is |
|---|---|
| `dev_slice.mbox` | 58 human emails from every thread whose root subject names KIP-1032 (Jakarta / JavaEE 10 / Jetty 12 upgrade) or KIP-1043 (administration of groups), plus 7 Jira notifications for the linked tickets KAFKA-16437, KAFKA-16096, KAFKA-16891 and KAFKA-18393. The messages are copied byte-for-byte from `data/raw/dev-kafka-YYYY-MM.mbox`. |
| `recorded_extractions.json` | `{email_id: response}`: the raw cached Sonnet response (`llm_cache.response`, prompt `extract-v1`) for each human email. Tests run the real `parse_response` on it. |
| `gold_entities.json` | One cluster per real person or component (`entity`, `type`, `mention_ids`, `tags`, and an optional `note`). It covers every Person and Component mention that comes from header, Jira or LLM sources. Read the clusters that have a `note` first: those were judgment calls. |
| `dev_questions.json` | 10 questions (4 single_hop, 2 inference, 2 comparison, 1 temporal, 1 null), each with a short answer and `gold_doc_ids`. |
| `tuning_log.json` | The 5 hand-written real-graph questions that shaped graph-retriever rules (Task 15), with the rule each one motivated. It is provenance only: no test grades against it, and the rules must be re-validated on the dev split. |

## Ids

Every id in the gold files is the id that a fresh database produces when it is built from this
fixture alone, not the id in the production database:

- `{email_id}:h0`, `{email_id}:m{n}` and `email:{email_id}:{chunk}`. Email ids are content
  hashes, so they are the same in every database.
- `jira:{event_id}:a0` and `jira:{event_id}`. Event ids are serial, numbered 1–7 in archive
  order once `dev_slice.mbox` is ingested into an empty database.

To build that database: ingest `dev_slice.mbox` alone into an empty, migrated database. Then
insert `recorded_extractions.json` into `llm_cache` under the configured extract model and
`PROMPT_VERSION`, and run `persist_extractions`. Both `scripts/make_fixture.py` and the
integrity test use the same `load_slice` function to do this.

## Regenerating

```
docker compose up -d postgres   # the populated production database; it is only read
uv run python scripts/make_fixture.py --kips KIP-1032,KIP-1043
```

The script rewrites `dev_slice.mbox` and `recorded_extractions.json`. It then rebuilds the slice
in a throwaway `kgrag_fixture` database on the same server, checks the result and drops the
database. It never writes the gold files. If the slice changes, relabel `gold_entities.json`
and recheck `dev_questions.json` by hand, then run `uv run pytest tests/test_fixture_integrity.py`.

## Known gaps against the brief

- **Variant spellings.** The brief asks for one component written 3 ways. The best this slice
  has is `MirrorMaker 2` / `MirrorMaker2` (tagged `variant_spellings`). The other components
  that span several emails are each extracted with a single spelling, such as `Connect` five
  times. No candidate slice of about 50 emails met all three hard cases at once.
- **Shared first names.** These occur naturally in two forms. David Jacot and David Arthur
  share a first name. Christopher Shannon and Chris Egerton both sign as "Chris", and Egerton
  signs one mail "Other Chris".
