# Kafka Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Tasks marked 👤 USER are implemented by the human partner, not by agents.** When execution reaches a 👤 task, stop, report that the task's interface, fixtures and tests are ready, and continue with the next non-USER task whose dependencies do not include the unfinished 👤 task. Never write logic in `resolve/resolver.py` or `retrieve/graph_retriever.py` beyond the stubs defined here.

**Goal:** Build Kafka Atlas — a knowledge-graph RAG system over the Apache Kafka `dev@` list (2023–2025) — and measure it against a LangChain + pgvector baseline on single-hop and multi-hop questions, served via FastAPI on GCP.

**Architecture:** A CLI batch pipeline (`ingest → extract → resolve → build → index → questions → eval → report`) writes emails/documents/vectors to Postgres+pgvector and the knowledge graph to Neo4j. Two retrievers (vector baseline; user-written graph retriever) feed one shared LangChain answer chain. The same retrievers and chain are served by a FastAPI app on Cloud Run; Langfuse traces everything.

**Tech Stack:** Python 3.12, uv, Postgres 16 + pgvector, Neo4j 5, LangChain (`langchain-postgres`, `langchain-voyageai`, `langchain-anthropic`), Anthropic SDK (Message Batches), FastAPI, Typer, pydantic v2, psycopg 3, neo4j driver, rapidfuzz, tiktoken, httpx, pytest + testcontainers, Docker, GCP Cloud Run / Cloud SQL / Secret Manager, Neo4j AuraDB Free, Langfuse Cloud.

**Spec:** `docs/superpowers/specs/2026-09-28-kafka-kg-rag-design.md`

## Global Constraints

- Corpus: `dev@kafka.apache.org`, months `2023-01` through `2025-12` inclusive; keep all human emails (no sampling cap).
- Models (in `config.yaml`): extraction, drafting, answering `claude-sonnet-5-5`; judge `claude-opus-5-5`; embeddings: Voyage general-purpose text model, pinned in Task 13.
- One `VoyageAIEmbeddings` instance is shared by baseline indexing/queries and `KnowledgeGraph` vector search.
- Retrieval `budget_tokens` default `4000`; email chunk size `500` tokens. All token counts use `kgrag.tokens.count_tokens` (tiktoken `cl100k_base`) — the one counter for every budget and chunk decision.
- Both retrievers return `RetrievalResult` over the same `documents` table; both systems use the same answer chain.
- Question set: ~300 reviewed (single-hop ~100, inference ~80, comparison ~50, temporal ~30, null ~40); split 20% dev / 80% test, stratified by type, fixed seed.
- Evidence sampling for questions uses only header/thread and regex-ID links — never Neo4j.
- ER quality target: B³ F1 ≥ `0.85` on `tests/fixtures/gold_entities.json`.
- Every results file records model IDs, prompt versions and config hash. Headline numbers come only from the test-split run.
- `extract` and `questions` print a cost estimate and require `--yes` or interactive confirmation above `cost_guard_usd` (default `5.00`).
- Langfuse is optional at runtime: with no `LANGFUSE_*` keys, tracing is a no-op.
- Test markers: `integration` (needs Docker), `quality` (reports metrics; non-blocking in default run), `live` (hits paid APIs; never run by default).

## Review Focus

1. **Odd email encodings** (non-UTF-8 charsets, HTML-only or multipart bodies) — parsing must decode to text, never crash the month. Test in Task 3.
2. **Jira notification subjects the parser doesn't know** (e.g. `[Comment Edited]`, `[Work logged]`) — return `None`, counted as unparsed, no exception. Test in Task 4.
3. **A single document larger than the remaining budget** — packing skips it and keeps filling with smaller ones; never exceeds the budget. Test in Task 13.
4. **Questions that name no known entity** — graph retriever must still return a valid (possibly empty) `RetrievalResult`; answer chain then reports insufficient information. Test in Task 14 (contract) and Task 16.
5. **Neo4j paused/unreachable during an eval run** — fail fast with a clear error before writing any results, never record a run full of zeros. Test in Task 20.

---

## File map

```
pyproject.toml  config.yaml  docker-compose.yml  Dockerfile  .env.example  README.md
migrations/001_init.sql  migrations/002_extraction.sql  migrations/003_questions.sql
scripts/deploy.sh
src/kgrag/
  config.py          # Settings (pydantic-settings) loaded from config.yaml + env
  tokens.py          # count_tokens
  observability.py   # Langfuse setup, traced decorator, langchain callback
  cli.py             # Typer app; one subcommand per stage
  store/db.py        # connect(), apply_migrations()
  store/repo.py      # all SQL reads/writes for emails, docs, mentions, questions
  ingest/fetch.py    ingest/parse.py   ingest/clean.py   ingest/jira.py   ingest/pipeline.py
  extract/regex_ids.py  extract/schema.py  extract/prompts.py  extract/llm.py  extract/pipeline.py
  resolve/interface.py  resolve/exact_match.py  resolve/resolver.py (👤)
  graph/types.py  graph/build.py  graph/kg.py
  retrieve/types.py  retrieve/packing.py  retrieve/embeddings.py  retrieve/index.py
  retrieve/vector.py  retrieve/graph_retriever.py (👤)
  answer/prompt.py  answer/chain.py
  questions/sample.py  questions/draft.py  questions/review.py  questions/split.py
  eval/bcubed.py  eval/metrics.py  eval/judge.py  eval/run.py  eval/report.py
  api/main.py
tests/  (mirrors src; conftest.py provides pg/neo4j testcontainer fixtures)
tests/fixtures/  dev_slice.mbox  jira_samples/  recorded_extractions.json
                 gold_entities.json  dev_questions.json  recorded_embeddings.json
```

---

## Phase 0 — Foundation

### Task 1: Project scaffold, config, DB plumbing, tracing

**Files:**
- Create: `pyproject.toml`, `config.yaml`, `.env.example`, `docker-compose.yml`, `migrations/001_init.sql`, `src/kgrag/{config,tokens,observability,cli}.py`, `src/kgrag/store/db.py`
- Test: `tests/test_config.py`, `tests/test_tokens.py`, `tests/store/test_db.py`, `tests/conftest.py`

**Interfaces:**
- Produces:
  - `Settings` (pydantic-settings) with fields `pg_dsn: str`, `neo4j_uri: str`, `neo4j_user: str`, `neo4j_password: str`, `models: ModelConfig(extract, draft, answer, judge, embed: str)`, `budget_tokens: int = 4000`, `chunk_tokens: int = 500`, `cost_guard_usd: float = 5.0`, `prompt_versions: dict[str,str]`, `data_dir: Path`; `get_settings() -> Settings` (cached); `config_hash(s: Settings) -> str` (sha256 of sorted JSON, first 12 hex chars).
  - `count_tokens(text: str) -> int`.
  - `connect(dsn: str | None = None) -> psycopg.Connection`; `apply_migrations(conn) -> list[str]` (applies `migrations/*.sql` in name order, records them in `schema_migrations`, returns newly applied names).
  - `traced(name: str)` decorator; `langchain_callbacks() -> list` (empty list when Langfuse keys absent).
  - Typer `app` in `cli.py`, entry point `kgrag`.
  - pytest fixtures `pg_conn` (pgvector/pgvector:pg16 testcontainer, migrations applied) and `neo4j_driver` (neo4j:5 testcontainer), both session-scoped, marked `integration`.
- `001_init.sql`: tables `emails(id text pk, message_id text unique, thread_id text, date timestamptz, sender_name text, sender_address text, subject text, body_clean text, kind text check (kind in ('human','jira')))`, `threads(thread_id text pk, root_message_id text, subject text)`, `jira_events(id serial pk, ticket text, action text, actor text, date timestamptz, status text, fix_version text, source_email_id text references emails)`, `documents(doc_id text pk, source_kind text, source_id text, chunk_index int, text text, token_count int, date timestamptz, sender text, subject text)`, `llm_cache(email_id text, prompt_version text, model text, response jsonb, primary key(email_id, prompt_version, model))`, `failures(id serial pk, stage text, item_id text, error text, at timestamptz default now())`; `create extension if not exists vector`.

- [ ] **Step 1: Write failing tests** — `test_settings_load_defaults` (`get_settings().budget_tokens == 4000`, `.chunk_tokens == 500`, `.models.judge == "claude-opus-5-5"`); `test_config_hash_stable` (same settings → same 12-char hash; changing `budget_tokens` changes it); `test_count_tokens` (`count_tokens("") == 0`, `count_tokens("hello world") == 2`); `test_traced_noop_without_keys` (decorated fn returns its value with `LANGFUSE_*` unset); `test_apply_migrations_idempotent` (integration: first call returns `["001_init.sql"]`, second returns `[]`, table `documents` exists).
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py tests/test_tokens.py tests/store -v` — Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** the files above. `docker-compose.yml` services: `postgres` (`pgvector/pgvector:pg16`, port 5432) and `neo4j` (`neo4j:5`, ports 7474/7687, `NEO4J_AUTH=neo4j/devpassword`). Langfuse via `langfuse` SDK's `observe` and `CallbackHandler`, guarded by key presence. Register markers in `pyproject.toml`; default `addopts = "-m 'not live'"`.
- [ ] **Step 4: Run** the same command — Expected: PASS. Also `docker compose up -d && uv run kgrag --help` lists the app.
- [ ] **Step 5: Commit** `feat: project scaffold, config, db plumbing, tracing`

---

## Phase 1 — Ingestion

### Task 2: Fetch monthly mbox archives

**Files:** Create `src/kgrag/ingest/fetch.py`; Test `tests/ingest/test_fetch.py`

**Interfaces:**
- Produces: `month_range(start: str, end: str) -> list[str]` (`"2023-01"`…, inclusive); `fetch_month(month: str, dest_dir: Path, client: httpx.Client | None = None) -> Path`. URL: `https://lists.apache.org/api/mbox.lua?list=dev@kafka.apache.org&date={month}`. Writes `dest_dir/dev-kafka-{month}.mbox`; skips download if the file exists and is non-empty; 3 retries with exponential backoff on 5xx/timeouts.

- [ ] **Step 1: Write failing tests** (`respx` mocks): `test_month_range` (`month_range("2023-11","2024-02") == ["2023-11","2023-12","2024-01","2024-02"]`); `test_fetch_writes_file`; `test_fetch_skips_existing` (no HTTP call made); `test_fetch_retries_then_succeeds` (two 503s then 200 → file written).
- [ ] **Step 2: Run** `uv run pytest tests/ingest/test_fetch.py -v` — FAIL.
- [ ] **Step 3: Implement** with `httpx` + `tenacity`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: mbox fetcher`

### Task 3: Parse, clean, classify, thread

**Files:** Create `src/kgrag/ingest/parse.py`, `src/kgrag/ingest/clean.py`; Test `tests/ingest/test_parse.py`, `tests/ingest/test_clean.py`

**Interfaces:**
- Produces:
  - `RawEmail` (pydantic): `message_id, in_reply_to: str | None, references: list[str], date: datetime, sender_name, sender_address, subject, body: str`.
  - `parse_mbox(path: Path) -> Iterator[RawEmail]` — decodes declared charsets with `errors="replace"`; prefers `text/plain` part, falls back to HTML stripped to text; skips messages without `Message-ID` (logged to `failures` by the caller).
  - `dedupe(emails: Iterable[RawEmail]) -> list[RawEmail]` — first occurrence per `message_id`.
  - `assign_threads(emails: list[RawEmail]) -> dict[str, str]` — message_id → thread_id (thread_id = root message_id; follows `In-Reply-To`, then last known `References` entry; orphans are their own root).
  - `classify_kind(sender_name: str, sender_address: str) -> Literal["human","jira"]` — `jira` if name ends with `(Jira)`, address starts with `jira@`, or sender matches `jenkins|github|buildbot|notifications@` (case-insensitive).
  - `clean_body(text: str) -> str` — removes quoted lines (`^>`), "On … wrote:" blocks and everything after, signature after a line equal to `-- `; collapses Java stack traces to `[stack trace: {Exception} in {first frame class}]`, unified diffs to `[patch: {n} lines]`, and runs of ≥10 log-format lines to `[log: {n} lines]`.

- [ ] **Step 1: Write failing tests:** `test_quoted_reply_removed`; `test_on_wrote_block_removed` (`"Thanks!\n\nOn Tue, Mar 5, 2024 at 10:00 AM Jun Rao <jun@x.com> wrote:\n> old"` → `"Thanks!"`); `test_signature_removed`; `test_stack_trace_collapsed` (input with `java.lang.NullPointerException` and `at org.apache.kafka.clients.consumer.KafkaConsumer.poll(...)` → contains `"[stack trace: NullPointerException in KafkaConsumer]"`); `test_patch_collapsed` (`"[patch: 12 lines]"`); `test_classify_kind` (`("Chia-Ping Tsai (Jira)", "jira@apache.org") → "jira"`, `("Jun Rao", "jun@confluent.io") → "human"`); `test_dedupe`; `test_assign_threads` (3-message chain + orphan); **Review Focus #1:** `test_parse_latin1_and_html_only` (a latin-1 encoded body and an HTML-only message both parse to non-empty text; no exception).
- [ ] **Step 2: Run** `uv run pytest tests/ingest/test_parse.py tests/ingest/test_clean.py -v` — FAIL.
- [ ] **Step 3: Implement** using stdlib `mailbox` + `email.policy.default`; HTML via `html.parser`-based stripping (no new dependency).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: mbox parsing and body cleaning`

### Task 4: Jira notification parser

**Files:** Create `src/kgrag/ingest/jira.py`, `tests/fixtures/jira_samples/*.eml`; Test `tests/ingest/test_jira.py`

**Interfaces:**
- Produces: `JiraEvent` (pydantic): `ticket: str` (e.g. `KAFKA-15432`), `action: Literal["created","assigned","resolved","commented","updated","reopened","closed"]`, `actor: str`, `date: datetime`, `status: str | None`, `fix_version: str | None`, `source_email_id: str`, `title: str`. `parse_jira_notification(email: RawEmail, email_id: str) -> JiraEvent | None`. `render_event(e: JiraEvent) -> str` — e.g. `"Alice Smith resolved KAFKA-1234 (Fix consumer rebalance) as Fixed in 3.7.0 on 2024-03-05."`.

- [ ] **Step 1: Capture samples.** Download one month (`fetch_month("2024-03", ...)`) and save 8 real Jira notifications covering Created, Assigned, Resolved (with `Fix Version/s`), Commented, Updated, Reopened, plus one unknown subject (e.g. `[Comment Edited]` or `[Work logged]`) to `tests/fixtures/jira_samples/`. Base the regexes on these real subjects/bodies (subject form observed: `[jira] [Created] (KAFKA-NNNNN) Title`; actor = sender display name minus ` (Jira)`).
- [ ] **Step 2: Write failing tests:** one per sample asserting `ticket`, `action`, `actor`, and for Resolved `status` + `fix_version`; `test_render_event` exact string above for a constructed event; **Review Focus #2:** `test_unknown_subject_returns_none`.
- [ ] **Step 3: Run** `uv run pytest tests/ingest/test_jira.py -v` — FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** — PASS.
- [ ] **Step 6: Commit** `feat: jira notification parser`

### Task 5: Ingest pipeline, documents table, `kgrag ingest`

**Files:** Create `src/kgrag/store/repo.py`, `src/kgrag/ingest/pipeline.py`; Modify `src/kgrag/cli.py`; Test `tests/ingest/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `repo.upsert_emails(conn, rows: list[EmailRow]) -> int`, `repo.insert_jira_events(conn, events: list[JiraEvent]) -> int`, `repo.rebuild_documents(conn, chunk_tokens: int) -> int`, `repo.get_documents(conn, doc_ids: list[str]) -> list[Document]`, `repo.log_failure(conn, stage, item_id, error)`.
  - `EmailRow` (pydantic): `id, message_id, thread_id, date, sender_name, sender_address, subject, body_clean, kind` (mirrors the `emails` table).
  - `Document` (pydantic): `doc_id, source_kind: Literal["email","jira"], source_id, chunk_index, text, token_count, date, sender, subject`.
  - `doc_id` format: `email:{email_id}:{chunk_index}` and `jira:{event_id}`.
  - `email_id` = first 16 hex chars of sha256(message_id).
  - `ingest(conn, mbox_paths: list[Path]) -> IngestStats(emails_human, emails_jira, jira_events, unparsed_jira, failures, documents)`.
  - CLI: `kgrag ingest --from 2023-01 --to 2025-12` (fetch then ingest; prints `IngestStats`).
- Chunking: human email bodies ≤ `chunk_tokens` → one chunk; longer → split on paragraph boundaries into ≤ `chunk_tokens` pieces. Each Jira event → one document whose text is `render_event`. Jira notification emails themselves are not documents.

- [ ] **Step 1: Write failing tests** (integration, on a 6-message test mbox built in the test incl. 2 Jira notifications): `test_ingest_counts` (human=4, jira=2, jira_events=2); `test_ingest_idempotent` (second run leaves row counts unchanged); `test_long_email_chunked` (a 1,200-token body → 3 docs, each `token_count <= 500`); `test_doc_ids_format`.
- [ ] **Step 2: Run** `uv run pytest tests/ingest/test_pipeline.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Real-data check.** `uv run kgrag ingest --from 2023-01 --to 2025-12`. Expected: human emails ≈ 11,900 (±10%) and jira emails ≈ 13,000 (±15%), matching the archive stats in spec §2. Record the actual numbers in the commit message.
- [ ] **Step 6: Commit** `feat: ingest pipeline (human=N, jira=M)`

---

## Phase 2 — Extraction

### Task 6: Regex ID extractor

**Files:** Create `src/kgrag/extract/regex_ids.py`; Test `tests/extract/test_regex_ids.py`

**Interfaces:**
- Produces: `IdMention(kind: Literal["KIP","JiraTicket","Release"], id: str, span: tuple[int,int])`; `extract_ids(text: str) -> list[IdMention]`. Normalization: `kip-848`/`KIP 848` → `KIP-848`; `kafka-1234` → `KAFKA-1234`; releases match `\b[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?\b` only when within 3 words of `release|version|branch|RC|rc` or in the form `x.y.z`; normalized to `x.y` or `x.y.z` as written (`3.7` and `3.7.0` stay distinct — ER does not merge IDs).

- [ ] **Step 1: Write failing tests:** `test_kip_variants`; `test_ticket`; `test_release_needs_context` (`"bump to 3.7.0"` → Release `3.7.0`; `"ratio 3.5 seconds"` → none); `test_no_false_kafka_word` (`"Kafka 3"` alone → none).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: regex ID extraction`

### Task 7: LLM extraction schema, prompt, batch runner, cache

**Files:** Create `src/kgrag/extract/{schema,prompts,llm}.py`; Test `tests/extract/test_llm.py`

**Interfaces:**
- Produces:
  - `Mention(mention_id: str, surface: str, type: Literal["Person","Component","KIP","JiraTicket","Release"], email_id: str, context: str)`; `Relation(type: Literal["PROPOSED","VOTED","REVIEWED","BLOCKS","AFFECTS","DEPENDS_ON","SUPERSEDES"], source_mention_id: str, target_mention_id: str, vote_value: Literal[1,0,-1] | None = None, binding: bool | None = None, confidence: float)`; `ExtractionResult(email_id: str, mentions: list[Mention], relations: list[Relation])`. `mention_id` = `{email_id}:m{n}`.
  - `PROMPT_VERSION = "extract-v1"`; `build_request(email: EmailRow, thread_subject: str) -> dict` (Anthropic Messages params with structured output: `output_config={"format": {"type": "json_schema", "schema": <mentions+relations schema>}}`; forced `tool_choice` returns a 400 on `claude-sonnet-5-5`, so do not use it; the fixed system prompt carries `cache_control` so the shared prefix is cached across requests).
  - `parse_response(email_id: str, message: dict) -> ExtractionResult` — raises `ExtractionError` on schema-invalid output or relations pointing to unknown mention ids.
  - `estimate_cost_usd(n_emails: int, avg_tokens: int, model: str) -> float` — per-token prices read from `config.yaml` `pricing:` (fill current prices from Anthropic's pricing page when writing the config).
  - `run_extraction(conn, emails: list[EmailRow], client, *, batch: bool = True) -> ExtractionStats(done, cached, failed)` — skips emails with a `llm_cache` row for `(email_id, PROMPT_VERSION, model)`; submits the rest via Message Batches API, polls to completion, stores raw responses in `llm_cache`; a failed/invalid item is retried once synchronously, then logged to `failures` (stage `extract`).

- [ ] **Step 1: Write failing tests** (fake client object recording calls): `test_build_request_uses_structured_output` (has `output_config.format`, no `tool_choice`, system block has `cache_control`); `test_parse_valid`; `test_parse_rejects_dangling_relation`; `test_cache_hit_skips_call` (integration: pre-seeded cache → client called with 0 requests); `test_invalid_then_failure_logged` (invalid twice → 1 row in `failures`, stats.failed == 1); `test_prompt_version_change_reprocesses`.
- [ ] **Step 2: Run** `uv run pytest tests/extract/test_llm.py -v` — FAIL.
- [ ] **Step 3: Implement.** Prompt text in `prompts.py`: instructs extraction of mentions *as written* (no normalization or merging), the closed relation set with one-line definitions, `VOTED` only for explicit `+1/0/-1` votes (`binding` when the voter writes "binding"), and context = the sentence containing the mention. Wrap client calls with `traced("extract")`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: LLM extraction with batch + cache`

### Task 8: Extraction persistence, anchor hints, `kgrag extract`

**Files:** Create `migrations/002_extraction.sql`, `src/kgrag/extract/pipeline.py`; Modify `src/kgrag/store/repo.py`, `src/kgrag/cli.py`; Test `tests/extract/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 5–7.
- Produces:
  - Tables `mentions(mention_id pk, surface, type, email_id, context, source text check (source in ('llm','regex','header','jira')))`, `relations(id serial pk, type, source_mention_id, target_mention_id, vote_value int, binding bool, confidence real, email_id)`.
  - Header mentions: one `Person` mention per human email sender (`surface` = sender name, `context` = `"<sender_name> <sender_address>"`, source `header`). Jira actor mentions: one `Person` mention per event actor (source `jira`). Regex mentions for KIP/JiraTicket/Release (source `regex`).
  - `AnchorHints(sender_pairs: list[tuple[str, str]], jira_actors: list[str])` (name ↔ address pairs as seen in headers; distinct Jira display names); `repo.load_mentions(conn) -> list[Mention]`, `repo.load_anchor_hints(conn) -> AnchorHints`.
  - CLI: `kgrag extract [--limit N] [--yes]` — prints cost estimate, enforces `cost_guard_usd`, runs extraction, persists mentions/relations from cache.

- [ ] **Step 1: Write failing tests** (integration, recorded responses for 3 emails): `test_persist_from_cache` (mentions/relations row counts match recorded results); `test_header_and_jira_mentions_created`; `test_anchor_hints` (pairs include `("Jun Rao","jun@...")`); `test_cost_guard_blocks_without_yes` (estimate above threshold, no `--yes`, non-interactive → exit code 2, no API call).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Real run.** `uv run kgrag extract --limit 200`, inspect 10 results by hand (`kgrag extract --show 10`), adjust the prompt if mentions are merged/normalized or relations hallucinated (bump `PROMPT_VERSION`), then run the full corpus. Record emails done/failed and actual cost.
- [ ] **Step 6: Commit** `feat: extraction pipeline (done=N, failed=M, cost=$X)`

### Task 9: Dev fixture

**Files:** Create `tests/fixtures/dev_slice.mbox`, `tests/fixtures/recorded_extractions.json`, `tests/fixtures/gold_entities.json`, `tests/fixtures/dev_questions.json`, `scripts/make_fixture.py`; Test `tests/test_fixture_integrity.py`

**Interfaces:**
- Produces (consumed by Tasks 10–16, 23):
  - `dev_slice.mbox`: ~50 real human emails + their thread's Jira notifications around 2–3 KIPs (pick KIPs with ≥15 human emails and ≥2 linked Jira tickets in 2023–2025), raw from the downloaded archives.
  - `recorded_extractions.json`: `{email_id: ExtractionResult}` from Task 8's cache for those emails.
  - `gold_entities.json`: `{"clusters": [{"entity": "Jun Rao", "type": "Person", "mention_ids": [...]}, ...]}` covering every Person and Component mention in the slice (header, jira, llm). Must include: ≥2 people sharing a first name; ≥1 person appearing via personal address, `@apache.org` address and Jira display name; ≥1 component written ≥3 ways.
  - `dev_questions.json`: 10 entries `{id, question, answer, type, gold_doc_ids}` — 4 single-hop, 2 inference, 2 comparison, 1 temporal, 1 null (`gold_doc_ids: []`), answerable only from the slice.

- [ ] **Step 1: Write failing test** `test_fixture_integrity`: every mention id in `gold_entities.json` exists in recorded extractions or derived header/jira mentions; no mention in two clusters; every `gold_doc_ids` entry exists after ingesting `dev_slice.mbox`; the three hard-case requirements above hold (encoded as checks on cluster metadata `"tags": ["shared_first_name" | "multi_identity" | "variant_spellings"]`).
- [ ] **Step 2: Run** — FAIL (files missing).
- [ ] **Step 3: Build** with `scripts/make_fixture.py --kips KIP-a,KIP-b,...`; draft gold clusters and questions by reading the slice.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: 👤 USER CHECKPOINT.** Ask the human partner to review `gold_entities.json` and `dev_questions.json` (their ER and retrieval tests grade against these). Apply their corrections; rerun Step 4.
- [ ] **Step 6: Commit** `test: dev fixture (reviewed)`

---

## Phase 3 — Entity resolution

### Task 10: Resolver interface, exact-match resolver, ER test suite, `kgrag resolve`

**Files:** Create `src/kgrag/resolve/{interface,exact_match,resolver}.py`, `src/kgrag/eval/bcubed.py`; Modify `src/kgrag/cli.py`, `src/kgrag/store/repo.py`; Test `tests/resolve/test_contract.py`, `tests/resolve/test_quality.py`, `tests/eval/test_bcubed.py`

**Interfaces:**
- Consumes: `Mention`, `AnchorHints` (Task 8), fixtures (Task 9).
- Produces:
  - `Resolution(assignments: dict[str, str], canonical_names: dict[str, str], entity_types: dict[str, str])` — mention_id → entity_id; entity_id → canonical name; entity_id → type.
  - `class Resolver(Protocol): name: str; def resolve(self, mentions: list[Mention], hints: AnchorHints) -> Resolution`.
  - `ExactMatchResolver` (`name = "exact"`): groups by `(type, surface.strip().casefold())`. KIP/JiraTicket/Release mentions are grouped by normalized ID for every resolver (applied by `resolve_all`, not by resolvers).
  - `resolve_all(resolver: Resolver, mentions, hints) -> Resolution` — passes only `Person` and `Component` mentions to the resolver, merges in ID-based entities; entity_id format `{type}:{slug}` for IDs (e.g. `KIP:KIP-848`) and resolver-chosen opaque strings otherwise, prefixed with type.
  - `resolver.py`: `class UserResolver: name = "user"; def resolve(self, mentions, hints) -> Resolution: raise NotImplementedError("👤 implement entity resolution")`.
  - `bcubed_f1(pred: dict[str, str], gold: dict[str, str]) -> float` (B³ precision/recall averaged over mentions present in gold).
  - `repo.save_resolution(conn, resolver_name, resolution)` (table `resolutions(resolver, mention_id, entity_id, canonical_name, entity_type)`), `repo.load_resolution(conn, resolver_name) -> Resolution`.
  - CLI: `kgrag resolve --resolver {exact,user}`.

- [ ] **Step 1: Write failing tests:**
  - `test_bcubed_perfect == 1.0`; `test_bcubed_all_singletons` (gold one cluster of 4 → pred all singletons → precision 1.0, recall 0.25, F1 0.4); `test_bcubed_all_merged`.
  - Contract (parametrized over `[ExactMatchResolver(), UserResolver()]`, on fixture mentions): `test_every_mention_assigned`; `test_deterministic` (two runs → identical assignments); `test_no_cross_type_merge`; `test_canonical_name_for_every_entity`.
  - Quality (`@pytest.mark.quality`, `UserResolver` only): `test_bcubed_at_least_0_85`; `test_shared_first_name_not_merged`; `test_multi_identity_merged`; `test_variant_spellings_merged` — each driven by the `tags` in `gold_entities.json`. Each prints its score.
- [ ] **Step 2: Run** `uv run pytest tests/resolve tests/eval/test_bcubed.py -v` — Expected: B³ tests FAIL (missing), contract tests for `exact` FAIL (missing); after Step 3, `UserResolver` cases fail with `NotImplementedError` (expected until Task 11).
- [ ] **Step 3: Implement** everything above except `UserResolver` logic.
- [ ] **Step 4: Run** — Expected: all `exact` contract tests and B³ tests PASS; `UserResolver` tests fail with `NotImplementedError` only. Record exact-match B³ on the fixture in the commit message (it is the ablation's floor).
- [ ] **Step 5: Commit** `feat: resolver interface, exact-match baseline, ER tests (exact B3=X)`

### Task 11: 👤 USER — Implement entity resolution

**Files:** Modify `src/kgrag/resolve/resolver.py` (and any new modules under `src/kgrag/resolve/`).

**Interfaces:** Implements `UserResolver.resolve(mentions: list[Mention], hints: AnchorHints) -> Resolution` from Task 10. Receives only `Person` and `Component` mentions. Must not touch the database.

**Done when:**
- [ ] `uv run pytest tests/resolve -m "not quality"` — all contract tests PASS for `user`.
- [ ] `uv run pytest tests/resolve -m quality -s` — B³ F1 ≥ 0.85 and the three edge-case tests PASS (target adjustable; record what you reach).
- [ ] `uv run kgrag resolve --resolver user` completes on the full corpus; spot-check the 20 largest Person clusters.
- [ ] Commit `feat: entity resolution (B3=X on dev fixture)`.

---

## Phase 4 — Knowledge graph

### Task 12: Graph build to Neo4j, `kgrag build`

**Files:** Create `src/kgrag/graph/types.py`, `src/kgrag/graph/build.py`; Modify `src/kgrag/cli.py`; Test `tests/graph/test_build.py`

**Interfaces:**
- Consumes: emails/threads/jira_events/mentions/relations (Tasks 5, 8), `Resolution` (Task 10).
- Produces:
  - Node labels `Person, KIP, JiraTicket, Release, Component, Email`, each with `id` (entity_id or `email:{email_id}`), `name`, and for `Email`: `date, subject, sender`. Uniqueness constraint on `id` per label.
  - Relationship types and sources exactly as spec §4 table, each with properties `source_doc_ids: list[str]`, `date: datetime`, `extractor: str`, `confidence: float` (1.0 for header/regex/jira). Parallel facts between the same pair with the same type are merged into one relationship accumulating `source_doc_ids`.
  - `(:GraphMeta {resolver, built_at, config_hash, node_count, rel_count})` singleton; `GraphMeta` dataclass with the same fields in `graph/types.py`.
  - `build_graph(conn, driver, resolver_name: str) -> BuildStats(nodes: dict[str,int], rels: dict[str,int])` — clears the database, creates constraints, loads via `UNWIND $rows AS r MERGE ...` in batches of 1,000.
  - `read_graph_meta(driver) -> GraphMeta | None`.
  - CLI: `kgrag build --resolver {exact,user}`.

- [ ] **Step 1: Write failing tests** (integration; fixture ingested + recorded extractions + `exact` resolution): `test_node_counts_by_label` (Email count == human emails in slice); `test_sent_and_replied_to`; `test_jira_edges` (a Resolved event yields `RESOLVED` and `FIX_VERSION`); `test_edge_provenance_present` (every relationship has non-empty `source_doc_ids` that exist in `documents`); `test_rebuild_idempotent` (two builds → identical counts); `test_graph_meta_written`.
- [ ] **Step 2: Run** `uv run pytest tests/graph/test_build.py -v` — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: neo4j graph build`

### Task 13: Embeddings, pgvector index, budget packing, vector baseline, `kgrag index`

**Files:** Create `src/kgrag/retrieve/{types,packing,embeddings,index,vector}.py`, `tests/fixtures/recorded_embeddings.json`, `scripts/record_embeddings.py`; Modify `config.yaml`, `src/kgrag/cli.py`; Test `tests/retrieve/test_packing.py`, `tests/retrieve/test_vector.py`

**Interfaces:**
- Consumes: `Document`, `repo.get_documents` (Task 5).
- Produces:
  - `Evidence(doc_id: str, text: str, score: float)`; `RetrievalResult(evidence: list[Evidence], trace: dict = {})`.
  - `pack_to_budget(candidates: list[Evidence], budget_tokens: int) -> list[Evidence]` — in order, include each candidate whose `count_tokens(text)` fits the remaining budget, skip ones that don't, stop when nothing remaining can fit; dedupe by `doc_id`.
  - `get_embeddings() -> Embeddings` — process-wide singleton `VoyageAIEmbeddings(model=settings.models.embed)`; in tests replaced by `RecordedEmbeddings(path)` (dict text→vector; raises on unknown text) via `set_embeddings_for_tests(e)`.
  - Collections in `PGVector` (`langchain-postgres`): `chunks` (one entry per `documents` row, metadata `doc_id`) and `node_names` (one entry per graph node with `name`, metadata `node_id`, `label`).
  - `build_index(conn, driver) -> IndexStats(chunks: int, node_names: int)`; CLI `kgrag index` (re-embeds only missing `doc_id`s / node ids).
  - `VectorRetriever(store).retrieve(question: str, budget_tokens: int) -> RetrievalResult` — similarity search with `k=50`, cosine scores, then `pack_to_budget`; `trace = {"k": 50, "considered": n}`.

- [ ] **Step 1: Pin the embedding model.** Check Voyage AI's current docs for their general-purpose text embedding model; set `models.embed` in `config.yaml` and note the dimension. Run `scripts/record_embeddings.py` (live, once) to embed every fixture document, node name and `dev_questions.json` question into `recorded_embeddings.json`.
- [ ] **Step 2: Write failing tests:** packing — `test_respects_budget`; `test_dedupes`; **Review Focus #3:** `test_skips_oversized_then_continues` (budget 100, candidates of 150, 40, 40 tokens → returns the two 40-token items). Vector (integration, recorded embeddings): `test_index_counts`; `test_retrieve_returns_gold_for_single_hop` (for the fixture's single-hop questions, at least one gold doc in top results — print recall); `test_retrieve_within_budget`; `test_index_incremental` (second `build_index` embeds 0 new items).
- [ ] **Step 3: Run** `uv run pytest tests/retrieve/test_packing.py tests/retrieve/test_vector.py -v` — FAIL. **Step 4: Implement.** **Step 5: Run** — PASS.
- [ ] **Step 6: Commit** `feat: pgvector index and vector baseline (embed model: X)`

### Task 14: KnowledgeGraph wrapper, graph-retriever stub, retrieval test suite

**Files:** Create `src/kgrag/graph/kg.py`, `src/kgrag/retrieve/graph_retriever.py`; Test `tests/graph/test_kg.py`, `tests/retrieve/test_graph_contract.py`, `tests/retrieve/test_graph_quality.py`, `tests/retrieve/conftest.py`

**Interfaces:**
- Consumes: graph (Task 12), `PGVector` collections + `get_embeddings` (Task 13), `repo.get_documents`.
- Produces:
  - `Node(id: str, label: str, name: str, props: dict)`; `Edge(type: str, source_id: str, target_id: str, props: dict)` in `graph/types.py`.
  - `class KnowledgeGraph` (constructed with `driver`, `conn`, `store`):
    - `find_nodes(name: str, label: str | None = None, fuzzy: bool = False, limit: int = 10) -> list[Node]` (fuzzy via `rapidfuzz.process.extract` over cached names, score ≥ 80)
    - `neighbors(node_id: str, edge_types: list[str] | None = None, direction: Literal["out","in","both"] = "both", since: date | None = None, until: date | None = None) -> list[tuple[Edge, Node]]`
    - `source_docs(edge: Edge) -> list[str]`
    - `search_nodes(query: str, k: int = 10) -> list[tuple[Node, float]]` (`node_names` collection)
    - `search_chunks(query: str, k: int = 20) -> list[Evidence]` (`chunks` collection)
    - `documents(doc_ids: list[str]) -> list[Document]`
    - `cypher(query: str, params: dict | None = None) -> list[dict]`
  - `graph_retriever.py`: `def retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int) -> RetrievalResult: raise NotImplementedError("👤 implement graph retrieval")`. Imports `pack_to_budget` for the user's convenience.
  - `tests/retrieve/conftest.py`: session fixture `fixture_kg` (always uses the `exact` resolver, so recorded node-name embeddings stay valid) — ingests `dev_slice.mbox`, loads recorded extractions, resolves with `exact`, builds graph, indexes with recorded embeddings, returns a `KnowledgeGraph`.

- [ ] **Step 1: Write failing tests:**
  - `test_kg.py` (integration): one test per method on `fixture_kg` with concrete fixture entities (e.g. `find_nodes("KIP-848")[0].label == "KIP"`; fuzzy finds a person from a misspelled name; `neighbors` date filter excludes an older edge; `source_docs` ids exist; `cypher("MATCH (n:KIP) RETURN count(n) AS c")[0]["c"] >= 2`).
  - `test_graph_contract.py` (integration, against `graph_retriever.retrieve`): `test_respects_budget` (budgets 500 and 4000); `test_doc_ids_exist`; `test_deterministic`; `test_returns_retrieval_result_type`; **Review Focus #4:** `test_no_entity_question_returns_valid_result` (`"What's a good pizza recipe?"` → `RetrievalResult`, no exception, within budget).
  - `test_graph_quality.py` (`quality`): per fixture question, gold-source recall within 4000 tokens; prints a single-hop vs multi-hop table and the vector baseline's numbers for the same questions; asserts nothing beyond running (non-blocking by design).
- [ ] **Step 2: Run** `uv run pytest tests/graph/test_kg.py tests/retrieve/test_graph_contract.py -v` — `test_kg` FAIL (missing); contract tests fail.
- [ ] **Step 3: Implement** `KnowledgeGraph`, the stub, and the conftest.
- [ ] **Step 4: Run** — `test_kg.py` PASS; graph contract tests fail with `NotImplementedError` only.
- [ ] **Step 5: Commit** `feat: KnowledgeGraph wrapper and graph retrieval test suite`

### Task 15: 👤 USER — Implement graph retrieval

**Files:** Modify `src/kgrag/retrieve/graph_retriever.py` (and any new modules under `src/kgrag/retrieve/`).

**Interfaces:** Implements `retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int) -> RetrievalResult` (Task 14). Put entities found, paths walked and scores into `trace` — they appear in the API response, Langfuse, and report examples.

**Done when:**
- [ ] `uv run pytest tests/retrieve/test_graph_contract.py` — all PASS.
- [ ] `uv run pytest tests/retrieve/test_graph_quality.py -s` — runs; record single-hop/multi-hop recall vs the vector baseline on the fixture.
- [ ] Tuning uses only the fixture and the dev split (Task 19); the test split stays untouched until the final run.
- [ ] Commit `feat: graph retrieval (fixture multi-hop recall X vs vector Y)`.

---

## Phase 5 — Answering

### Task 16: Shared answer chain

**Files:** Create `src/kgrag/answer/prompt.py`, `src/kgrag/answer/chain.py`; Test `tests/answer/test_chain.py`

**Interfaces:**
- Consumes: `RetrievalResult`, `repo.get_documents`.
- Produces: `ANSWER_PROMPT_VERSION = "answer-v1"`; `format_sources(evidence: list[Evidence], docs: dict[str, Document]) -> str` (one block per source: `[doc_id] date · sender · subject\n text`); `Answer(text: str, cited_doc_ids: list[str], insufficient: bool)`; `answer(question: str, result: RetrievalResult, *, llm: BaseChatModel | None = None) -> Answer` (default `ChatAnthropic(model=settings.models.answer, temperature=0)`, Langfuse callbacks attached). Structured output: `{answer, cited_doc_ids, insufficient}`. System prompt (exact): `"Answer the question using only the sources below. Cite the doc_id of every source you use. If the sources do not contain enough information to answer, set insufficient to true and say so."`
- [ ] **Step 1: Write failing tests** (LangChain fake chat model returning fixed structured output): `test_format_sources_layout`; `test_answer_parses_citations`; `test_citations_filtered_to_given_sources` (cited ids not in the evidence are dropped); **Review Focus #4:** `test_empty_evidence_short_circuits` (no evidence → `Answer(insufficient=True)` without calling the model).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: shared answer chain`

---

## Phase 6 — Question set

### Task 17: Evidence sampling and question drafting, `kgrag questions draft`

**Files:** Create `migrations/003_questions.sql`, `src/kgrag/questions/sample.py`, `src/kgrag/questions/draft.py`; Modify `src/kgrag/cli.py`, `src/kgrag/store/repo.py`; Test `tests/questions/test_sample.py`, `tests/questions/test_draft.py`

**Interfaces:**
- Consumes: Postgres tables only (`documents`, `emails`, `threads`, `jira_events`, regex mentions). **Must not import `kgrag.graph` or take a Neo4j driver** (enforced by test).
- Produces:
  - Table `questions(id text pk, type text, question text, answer text, gold_doc_ids text[], status text check (status in ('draft','kept','edited','dropped')), split text, draft_model text, prompt_version text, reviewed_at timestamptz)`.
  - `QuestionRow` (pydantic): mirrors the `questions` table.
  - `EvidenceSet(type: QuestionType, doc_ids: list[str], link: str)` where `link` describes the connection (e.g. `"shared:KIP-848"`, `"thread:<id>"`); `QuestionType = Literal["single_hop","inference","comparison","temporal","null"]`.
  - `sample_evidence_sets(conn, qtype: QuestionType, n: int, seed: int) -> list[EvidenceSet]` — single-hop: 1 doc; inference/temporal: 2–3 docs from different threads sharing a regex ID or a thread chain; comparison: 2 docs about two different IDs of the same kind; null: 0 docs (drafted from a random thread's topic).
  - `DRAFT_PROMPT_VERSION = "draft-v1"`; `draft_question(es: EvidenceSet, docs: list[Document], llm) -> DraftQuestion(question, answer, type, gold_doc_ids)`.
  - `answerable_from_single_source(draft: DraftQuestion, docs, llm) -> bool` — asks the model per individual gold doc; multi-hop drafts answerable from any one doc are relabeled `single_hop` (gold = that doc) or dropped if single-hop quota is full.
  - `null_question_absent(draft, conn) -> bool` — keyword search (Postgres full-text over `documents.text`) for the draft's key terms finds no doc stating the answer; drafts failing this are dropped.
  - CLI: `kgrag questions draft --pilot` (30 drafts, 6 per type) and `kgrag questions draft --counts single_hop=100,inference=80,comparison=50,temporal=30,null=40 [--yes]` (drafts ~1.3× each count to absorb review drops; cost guard applies).
- [ ] **Step 1: Write failing tests:** `test_sampler_does_not_use_graph` (assert `kgrag.graph` not in `sys.modules` after importing and running `sample_evidence_sets` in a subprocess); `test_sampler_deterministic_by_seed`; `test_inference_sets_share_id_across_threads`; `test_comparison_sets_two_distinct_ids`; `test_single_source_filter_relabels` (fake llm answers from doc A alone → draft becomes `single_hop`); `test_null_keyword_check`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Pilot.** `uv run kgrag questions draft --pilot`, then 👤 USER reviews the 30 via Task 18's CLI (if Task 18 isn't done yet, do it first and return here). Revise the draft prompt from what was dropped/edited (bump `DRAFT_PROMPT_VERSION`).
- [ ] **Step 6: Commit** `feat: evidence sampling and question drafting (pilot keep rate X%)`

### Task 18: Review CLI and dev/test split

**Files:** Create `src/kgrag/questions/review.py`, `src/kgrag/questions/split.py`; Modify `src/kgrag/cli.py`; Test `tests/questions/test_review.py`, `tests/questions/test_split.py`

**Interfaces:**
- Produces:
  - `kgrag questions review [--type T]` — shows each `draft` question with its gold docs' text; keys `k` keep, `e` edit (opens `$EDITOR` on question/answer/gold ids as YAML), `d` drop, `q` quit; writes status immediately (resumable); prints progress `kept/edited/dropped/remaining` per type.
  - `assign_splits(questions: list[QuestionRow], dev_frac: float = 0.2, seed: int = 13) -> dict[str, Literal["dev","test"]]` — stratified by type, only `kept`/`edited`; `kgrag questions split` persists it and refuses to reassign an existing split unless `--force`.
- [ ] **Step 1: Write failing tests:** `test_review_keep_drop_persist` (drive with scripted input; restart resumes at the first `draft`); `test_edit_updates_fields`; `test_split_stratified` (100 per type → 20 dev each); `test_split_deterministic`; `test_split_excludes_dropped`; `test_split_refuses_reassign`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with Typer + `rich` prompts. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: question review CLI and splits`
- [ ] **Step 6: 👤 USER CHECKPOINT (after Task 17 full draft).** Human partner reviews all drafts in sessions of 30–50 until ≈300 kept/edited questions, then runs `kgrag questions split`.

---

## Phase 7 — Evaluation

### Task 19: Metrics

**Files:** Create `src/kgrag/eval/metrics.py`; Test `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `source_recall(retrieved: list[str], gold: list[str]) -> float | None` (None when gold is empty); `full_evidence(retrieved, gold) -> bool | None`; `bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]` (mean, lo, hi); `paired_diff_ci(a: list[float], b: list[float], n=10_000, seed=0) -> tuple[float, float, float]` (resamples question indices jointly).
- [ ] **Step 1: Write failing tests:** `source_recall(["a","b"], ["a","c"]) == 0.5`; `source_recall(["a"], []) is None`; `full_evidence(["a","c","x"], ["a","c"]) is True`; `bootstrap_ci([1.0]*50)` → `(1.0, 1.0, 1.0)`; `paired_diff_ci([1,1,1,0]*25, [0,1,1,0]*25)[0] == 0.25` and the interval excludes 0; seeds make results deterministic.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with numpy. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: retrieval metrics and bootstrap CIs`

### Task 20: Judge, eval runner, report, judge validation

**Files:** Create `src/kgrag/eval/judge.py`, `src/kgrag/eval/run.py`, `src/kgrag/eval/report.py`; Modify `src/kgrag/cli.py`; Test `tests/eval/test_judge.py`, `tests/eval/test_run.py`, `tests/eval/test_report.py`

**Interfaces:**
- Consumes: retrievers (Tasks 13, 14/15), `answer` (Task 16), questions + splits (Task 18), metrics (Task 19), `read_graph_meta` (Task 12).
- Produces:
  - `JUDGE_PROMPT_VERSION = "judge-v1"`; `judge(question: str, gold_answer: str, answer: Answer, llm=None) -> Literal["correct","partial","incorrect"]` (model `settings.models.judge`); for null questions, scoring is rule-based: `correct` iff `answer.insufficient`.
  - `run_eval(conn, driver, *, system: Literal["vector","graph"], split: Literal["dev","test"], resolver: str | None) -> Path` — for `graph`, first checks `read_graph_meta(driver).resolver == resolver` and Neo4j reachability, raising `EvalPreconditionError` before any question runs; per question records `{question_id, type, retrieved_doc_ids, recall, full_evidence, answer, verdict, latency_ms, tokens_in, tokens_out}`; writes `results/{run_id}.json` with header `{run_id, system, resolver, split, models, prompt_versions, config_hash, started_at, finished_at}`. `run_id = f"{system}-{resolver or 'na'}-{split}-{timestamp}"`. Each question runs in a Langfuse trace tagged `system`, `question_id`, `run_id`. Refuses `split="test"` for a `(system, resolver)` that already has a test run unless `--force` (one-shot rule).
  - `render_report(run_paths: list[Path], trace_examples: int = 3) -> str` — markdown: per system × {single-hop, multi-hop, each multi-hop type (marked *indicative*), null} tables for recall, full-evidence rate, answer accuracy (correct = 1, partial = 0.5), each `mean [lo, hi]`; paired differences vs the vector baseline on the same questions; cost and latency per question; side-by-side traces for `trace_examples` multi-hop questions where systems disagree.
  - `kgrag judge-check --run <id> --n 50` — samples 50 judged answers, asks the human for a verdict on each (resumable), prints agreement % and Cohen's κ; stored in `results/judge_check.json` and included by the report.
  - CLI: `kgrag eval --system {vector,graph} --split {dev,test} [--resolver R]`, `kgrag report results/*.json > results/REPORT.md`.
- [ ] **Step 1: Write failing tests:** `test_null_scored_by_abstention`; `test_judge_parses_verdict` (fake llm); **Review Focus #5:** `test_graph_eval_fails_fast_when_neo4j_down` (driver pointing at a closed port → `EvalPreconditionError`, no results file created) and `test_graph_eval_rejects_resolver_mismatch`; `test_results_file_header_fields`; `test_test_split_one_shot`; `test_report_tables` (two synthetic result files → markdown contains `Multi-hop` row with `[lo, hi]` and an *indicative* label).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: judge, eval runner, report`

---

## Phase 8 — Serving and deployment

### Task 21: FastAPI service

**Files:** Create `src/kgrag/api/main.py`; Test `tests/api/test_main.py`

**Interfaces:**
- Consumes: `VectorRetriever`, `graph_retriever.retrieve`, `KnowledgeGraph`, `answer`, `repo.get_documents`.
- Produces:
  - `POST /ask` body `{question: str (1–1000 chars), system: "vector" | "graph"}` → `200 {answer, insufficient, sources: [{doc_id, subject, date, sender, snippet}], trace, latency_ms}`; snippet = first 300 chars of the doc text.
  - `GET /health` → `200 {"postgres": "ok", "neo4j": "ok"}`; if Neo4j unreachable → `503 {"neo4j": "unavailable", "hint": "AuraDB Free instances pause after inactivity; resume it in the Aura console."}`; Postgres down → `503`.
  - Errors: unknown `system` or empty question → `422`; database unreachable during `/ask` → `503` with the same hint for Neo4j; `NotImplementedError` from the graph retriever → `501 {"detail": "graph retrieval not implemented yet"}`.
  - Every request runs inside a Langfuse trace tagged `system`; app built by `create_app(deps: Deps | None = None) -> FastAPI` where `Deps` holds `vector_retriever`, `kg`, `graph_retrieve`, `answer_fn`, `conn_factory`, so tests inject fakes.
- [ ] **Step 1: Write failing tests** (TestClient, injected fakes): `test_ask_vector_ok` (response fields present, sources match evidence); `test_ask_unknown_system_422`; `test_ask_empty_question_422`; `test_ask_graph_not_implemented_501`; `test_health_ok`; `test_health_neo4j_down_503_with_hint`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. Then `docker compose up -d && uv run uvicorn kgrag.api.main:app` and `curl -s localhost:8000/health` → `{"postgres":"ok","neo4j":"ok"}`.
- [ ] **Step 5: Commit** `feat: FastAPI /ask and /health`

### Task 22: Container and GCP deployment

**Files:** Create `Dockerfile`, `.dockerignore`, `scripts/deploy.sh`, `scripts/load_cloud.sh`; Modify `src/kgrag/store/db.py`, `src/kgrag/config.py`; Test `tests/store/test_db_modes.py`

**Interfaces:**
- Produces:
  - `connect()` supports `KGRAG_DB_MODE=cloudsql` using `google-cloud-sql-connector` (`INSTANCE_CONNECTION_NAME`, `DB_USER`, `DB_PASS`, `DB_NAME`); default mode unchanged.
  - `Dockerfile`: `python:3.12-slim`, uv install from lockfile, non-root user, `CMD uvicorn kgrag.api.main:app --host 0.0.0.0 --port $PORT`.
  - `scripts/deploy.sh PROJECT REGION` — builds with Cloud Build, deploys to Cloud Run service `kafka-atlas` with `--add-cloudsql-instances`, secrets from Secret Manager (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `NEO4J_PASSWORD`, `DB_PASS`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) via `--set-secrets`, `--min-instances 0`.
  - `scripts/load_cloud.sh` — runs `kgrag ingest/extract(from cache)/resolve/build/index` from the laptop against Cloud SQL (via Cloud SQL Auth Proxy) and AuraDB.
- [ ] **Step 1: Write failing test:** `test_cloudsql_mode_uses_connector` (connector factory mocked; asserts called with instance name) and `test_default_mode_uses_dsn`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Local container check.** `docker build -t kafka-atlas . && docker run --env-file .env -p 8080:8080 -e PORT=8080 kafka-atlas`, then `curl -s localhost:8080/health` → both `ok`.
- [ ] **Step 6: Verify AuraDB Free limits.** Read the current limits in the Aura console/docs; compare with `GraphMeta.node_count/rel_count` from the full build. Record both in the README. If over, stop and raise it with the human partner (fallback: AuraDB Professional).
- [ ] **Step 7: 👤 USER CHECKPOINT — cloud resources.** The human partner creates (or approves creating) the GCP project, Cloud SQL Postgres instance with `vector` extension, AuraDB Free instance, Langfuse Cloud project, and Secret Manager secrets; these involve their accounts and billing. Then run `scripts/load_cloud.sh` and `scripts/deploy.sh`; verify `curl $SERVICE_URL/health` and one `/ask` per system, and that the request appears in Langfuse.
- [ ] **Step 8: Commit** `feat: container and Cloud Run deployment`

---

## Phase 9 — Smoke test and write-up

### Task 23: End-to-end smoke test and README

**Files:** Create `tests/test_smoke.py`, `README.md`

**Interfaces:**
- Consumes: everything; recorded extractions and embeddings; fake chat models for answer/judge.
- Produces: `test_smoke` (integration, no network): ingest `dev_slice.mbox` → load recorded extractions → resolve `exact` → build → index → `run_eval(system="vector", split="dev")` on `dev_questions.json` → `render_report` produces a non-empty markdown with a multi-hop row. A second case runs `system="graph"` and is `skip`ped with reason "👤 graph retriever not implemented" while the stub raises `NotImplementedError`. Must finish in < 60 s.
- README sections: what it is (engineering knowledge assistant framing); architecture diagram (Mermaid); quickstart (`docker compose up`, `.env`, pipeline commands); results (pasted from `results/REPORT.md`, test split only, with CIs); judge agreement; how the question set was built (MultiHop-RAG recipe, header/regex-only sampling, single-source filter, human review); deployment; limitations copied from spec §12 plus anything learned.
- [ ] **Step 1: Write the failing smoke test.** **Step 2: Run** `uv run pytest tests/test_smoke.py -v` — FAIL until wiring gaps are fixed. **Step 3: Fix wiring only** (no new features). **Step 4: Run** — PASS (graph case skipped until Task 15).
- [ ] **Step 5: Write README** with a results placeholder that `kgrag report` fills; no numbers typed by hand.
- [ ] **Step 6: Final run (after Tasks 11, 15, 18 complete).** `kgrag eval` on the test split for `vector`, `graph --resolver exact` (after `kgrag build --resolver exact`), `graph --resolver user` (after `kgrag build --resolver user`); `kgrag judge-check` (👤 labels 50); `kgrag report`; paste into README.
- [ ] **Step 7: Commit** `docs: README and final results`

---

## Dependency summary

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → **11 👤** · 10 → 12 → 13 → 14 → **15 👤** · 13 → 16 · 5 → 17 ↔ 18 → **Task 18 Step 6 👤** · 19 → 20 (needs Task 15 and Task 18 Step 6 for the graph/test runs) · 16 → 21 → 22 · all → 23.
Agents can complete Tasks 1–10, 12–14, 16–17 (through the pilot), 18–22 (to the user checkpoints) and 23 Steps 1–5 without waiting on 👤 tasks; Task 12 builds and tests with the `exact` resolver.
