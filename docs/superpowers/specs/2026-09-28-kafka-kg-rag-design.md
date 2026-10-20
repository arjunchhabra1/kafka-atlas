# Kafka Atlas: Graph vs Vector Retrieval on Engineering Mailing Lists — Design

Date: 2026-09-28 (revised 2026-10-14: Neo4j / pgvector / LangChain / FastAPI /
GCP / Langfuse stack)
Status: Draft for review

Stack: Python, Neo4j (AuraDB), Postgres + pgvector (Cloud SQL), LangChain,
FastAPI, Docker, GCP Cloud Run, Langfuse.

## 1. Goal and framing

Build and evaluate an **engineering knowledge assistant** over the Apache Kafka
developer mailing list (`dev@kafka.apache.org`), standing in for the private
Slack/email/Jira history a real company would index. Compare **knowledge-graph
retrieval** against a **plain vector RAG baseline** on **single-hop** and
**multi-hop** questions.

- Audience: portfolio project. Target rigor is a clear, honest comparison —
  held-out test set, paired comparison, confidence intervals, stated
  limitations — not publication-grade.
- **The user implements entity resolution and graph retrieval.** Everything else
  (ingestion, extraction, graph build, baseline, question set, evaluation) is
  built around those two components, which ship with interfaces, fixtures and
  tests (see §4 and §6).

### Success criteria

1. The full pipeline runs end to end from raw mbox files to a results report.
2. A reviewed question set of about 300 questions exists, with a held-out test
   split that is evaluated once for the final numbers.
3. The README reports, for vector baseline, graph + exact-match ER, and graph +
   user ER: retrieval metrics and answer accuracy split by single-hop vs
   multi-hop, with 95% bootstrap CIs, plus a limitations section. Any headline
   figure (e.g. "multi-hop recall +N points over the baseline") is taken from
   the held-out test run, never set in advance.
4. A FastAPI `/ask` service is deployed on Cloud Run against Cloud SQL and
   AuraDB, with Langfuse traces for every request.

### Non-goals

Web UI, real-time ingestion, fine-tuning, publication-grade statistics, CI/CD
pipelines, frameworks that implement ER or graph retrieval (LlamaIndex
PropertyGraph, GraphRAG). LangChain is used only for the vector baseline and
the shared answer chain.

## 2. Data

### Corpus

- Source: monthly mbox archives of `dev@kafka.apache.org` from the Apache list
  archive API (`lists.apache.org`), **January 2023 – December 2025**.
- Measured volume (archive stats, 2026-09-28): 2023: 8,140 emails / 4,004 human;
  2024: 10,134 / 4,578; 2025: 7,027 / 3,279. About half of all traffic is
  automated Jira notifications.
- **Human corpus: all human-authored emails in the window (~11.9k)**, no
  sampling cap. ~310–330 distinct human senders per year; the top 30 senders
  write 55–63% of human mail. 148–182 distinct KIPs and 1.7k–2.8k Jira tickets
  appear in subject lines per year.
- Raw mbox files are kept on disk so every later stage can rerun from scratch.

### Parsing and cleaning

- Deduplicate by `Message-ID`; build threads from `In-Reply-To`/`References`.
- Strip quoted replies (`>` lines, "On … wrote:" blocks) and signatures.
- Collapse patches, stack traces and log dumps into short placeholders, e.g.
  `[stack trace: NullPointerException in KafkaConsumer]`.
- Tag each message `kind = human | jira` by sender pattern (`(Jira)` display
  name, `jira@` address, CI/bot senders).

### Jira notifications

- Parsed deterministically (no LLM) into `jira_events(ticket, action, actor,
  date, status, fix_version, source_email_id)`.
- **Fairness rule:** each event is also rendered as a short text document
  (e.g. "Alice resolved KAFKA-1234 as Fixed in 3.7.0") and indexed in the
  vector corpus. Both systems see the same information; the graph only
  organizes it differently.

### Storage

Postgres with the pgvector extension — `docker compose` locally, Cloud SQL for
Postgres in production. Tables: `emails(id, message_id, thread_id, date,
sender_name, sender_address, subject, body_clean, kind)`, `threads`,
`jira_events`, `documents` (the unified retrievable units: human emails/chunks
and Jira-event texts, each with a stable `doc_id`), `llm_cache`, `failures`,
plus the embedding collection managed by LangChain `PGVector`. Schema changes
are applied with plain SQL migration files.

### Dev fixture

About 50 hand-picked emails and Jira events centered on 2–3 KIPs, committed as
JSON under `tests/fixtures/`, with:
- `gold_entities.json`: hand-labeled clusters of mentions that refer to the same
  entity, including hard cases (shared first names; one person across personal
  address, `@apache.org` address and Jira handle).
- `dev_questions.json`: 5–10 questions with gold source `doc_id`s, covering
  single-hop and multi-hop.

## 3. Models

Hosted APIs for all model steps. Model IDs live in `config.yaml` and are recorded
in every results file.

Embeddings are called through LangChain's `VoyageAIEmbeddings`
(`langchain-voyageai`). A single instance is shared by the `PGVector` store
(indexing and baseline queries) and by the `KnowledgeGraph` vector-search
helpers, so both systems use identical embeddings.

| Step | Model |
|---|---|
| Extraction | `claude-sonnet-5-5` (via Message Batches API) |
| Question drafting | `claude-sonnet-5-5` |
| Answer generation (both systems) | `claude-sonnet-5-5` |
| Answer judge | `claude-opus-5-5` (different from the answer model to reduce self-preference) |
| Embeddings (both systems) | Voyage AI's current general-purpose text embedding model, selected and pinned in `config.yaml` during project setup |

## 4. Knowledge graph

### Schema (closed)

Nodes: `Person`, `KIP`, `JiraTicket`, `Release`, `Component`, `Email`
(provenance).

| Edge | From → To | Source |
|---|---|---|
| `SENT` | Person → Email | headers |
| `REPLIED_TO` | Email → Email | headers |
| `MENTIONS` | Email → KIP / JiraTicket / Release | regex |
| `FILED`, `ASSIGNED`, `RESOLVED` | Person → JiraTicket | Jira events |
| `FIX_VERSION` | JiraTicket → Release | Jira events |
| `PROPOSED` | Person → KIP | LLM |
| `VOTED` (value +1/0/−1, binding flag) | Person → KIP / Release | LLM |
| `REVIEWED` | Person → KIP / JiraTicket | LLM |
| `BLOCKS` | KIP / JiraTicket → Release | LLM |
| `AFFECTS` | KIP / JiraTicket → Component / Release | LLM |
| `DEPENDS_ON`, `SUPERSEDES` | KIP → KIP | LLM |

Every edge carries `source_doc_ids`, `date`, `extractor` (`header | regex |
jira | llm`) and `confidence`.

### Extraction

- **Regex extractor:** KIP IDs, Jira ticket IDs and release versions,
  normalized (`kip-848` → `KIP-848`). These IDs are canonical and bypass ER.
- **LLM extractor:** one structured-output call per human email (headers,
  thread subject, cleaned body). Output is **mentions, not entities**:
  `Mention(mention_id, surface, type, email_id, context)` plus relations between
  mentions, validated with pydantic. Cached by `(email_id, prompt_version,
  model)`; runs through the Batch API.

### Entity resolution — USER-IMPLEMENTED

- Location: `src/kgrag/resolve/resolver.py`, implementing the `Resolver`
  protocol in `resolve/interface.py`.
- Input: all `Mention`s plus `AnchorHints` (sender name ↔ address pairs from
  headers; Jira actor usernames from events).
- Output: `Resolution` = `mention_id → entity_id` mapping plus
  `entity_id → canonical_name`.
- Scope: primarily `Person` and `Component`.
- Provided tests:
  - contract: every mention is assigned; entity IDs are stable across runs;
    types are never merged across (`Person` never merges with `Component`).
  - quality: B³ F1 against `gold_entities.json`, target ≥ 0.85 (adjustable);
    named edge-case tests for the hard cases above.
- **Comparison resolver** (`resolve/exact_match.py`, written by Claude): merges
  only exact, case-normalized surface strings. Used as an ablation to measure
  what ER contributes.

Entity resolution is pure Python with no database access, so its tests run
without Docker.

### Graph build

Resolved mentions, regex/header/Jira edges and LLM relations are written to
**Neo4j** — `docker compose` locally, **AuraDB Free** in production — using
batched `UNWIND … MERGE` with uniqueness constraints on node IDs. A rebuild
clears and reloads the graph. Edge properties carry the provenance fields
above. The graph is exposed through a `KnowledgeGraph` wrapper (§5).

Expected size: ~20k nodes, ~100k relationships. AuraDB Free caps node and
relationship counts; the published caps are verified during setup. If they
are exceeded, fall back to AuraDB Professional (from $0.09/GB/hour, 1 GB
minimum, as listed on 2026-10-14).

## 5. Retrieval and answering

### Fairness rules

- Both retrievers receive the same `budget_tokens` (default 4,000).
- Both return `RetrievalResult` containing ranked `Evidence(doc_id, text,
  score)` over the same `documents` table, formatted identically for the answer
  model.

### Vector baseline (Claude)

LangChain + pgvector. One chunk per human email (split at ~500 tokens if
longer); one document per Jira-event text. Embedded into a LangChain `PGVector`
collection. A thin wrapper takes candidates in cosine-similarity order until
`budget_tokens` is filled and returns a `RetrievalResult`.

### Graph retrieval — USER-IMPLEMENTED

- Location: `src/kgrag/retrieve/graph_retriever.py`.
- Signature: `retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int)
  -> RetrievalResult`. `RetrievalResult` also carries a free-form `trace`
  (linked entities, paths walked) used for debugging and write-up examples.
- `KnowledgeGraph` (wraps the Neo4j Python driver) helpers, optional to use:
  node lookup by name (exact and fuzzy), neighbors/edges filtered by type and
  date, edge → source docs, vector search over node names and over document
  chunks (shared pgvector index), and raw `cypher(query, params)`.
- Entity linking in the question, traversal strategy (Python, Cypher, or
  both) and evidence ranking are entirely the user's design.
- Provided tests (run against a Neo4j testcontainer loaded with the dev
  fixture):
  - contract: respects budget; every `doc_id` exists; deterministic.
  - quality (reported, non-blocking marker): gold-source recall within budget on
    `dev_questions.json`, split single-hop / multi-hop.

### Answer generation (Claude)

One LangChain chain (fixed prompt and model) shared by all systems: answer only
from the given sources, cite `doc_id`s, and state "insufficient information"
when the sources do not support an answer.

## 6. Question set and evaluation

### Question types

| Type | Gold sources | Target count |
|---|---|---|
| Single-hop | 1 | ~100 |
| Inference (bridge) | 2–3 | ~55 |
| Comparison | 2 | ~50 |
| Temporal | 2–3 | ~55 |
| Null (unanswerable) | 0 | ~40 |

Total ~300 reviewed questions.

### Construction (MultiHop-RAG recipe)

1. **Evidence sampling uses only header and regex links** (same thread, shared
   KIP or ticket ID) — never the knowledge graph — so the question set is not
   biased toward the graph system.
2. An LLM drafts question, short answer, gold `doc_id`s and type from each
   evidence set. Null questions are drafted about plausible but absent facts
   and checked against the corpus by keyword search.
3. Automatic filter: a multi-hop draft that the LLM can answer from any single
   gold source alone is relabeled single-hop or dropped.
4. **Pilot:** draft and review ~30 questions first, revise the drafting prompt,
   then generate the rest.
5. Human review via a resumable CLI (keep / edit / drop), done in sessions of
   30–50. Estimated total ≤ 5 hours.

### Splits

20% dev (for tuning the user's retriever), 80% test (held out; evaluated once
for the final report).

### Systems

1. Vector baseline
2. Graph retrieval + exact-match resolver
3. Graph retrieval + user resolver

### Metrics

- Retrieval: gold-source recall within budget; **full-evidence rate** (all gold
  sources retrieved) for multi-hop.
- Answers: LLM judge labels correct / partial / incorrect against the reviewed
  answer; null questions scored on correct abstention.
- Judge validation: the user hand-labels ~50 answers; report agreement.
- Cost and latency per question.
- All metrics per question type with 95% bootstrap CIs; system differences
  computed paired over the same questions. The headline comparison is
  single-hop vs multi-hop; per-multi-hop-type results are reported as
  indicative (~40 test questions each).

### Outputs

`results/<run_id>.json` (with model IDs, prompt versions, config hash),
auto-generated markdown tables, and side-by-side retrieval traces for selected
questions.

## 7. Serving, deployment and observability

### API (FastAPI)

- `POST /ask {question, system: "vector" | "graph"}` → `{answer, sources:
  [{doc_id, subject, date, sender, snippet}], trace}`.
- `GET /health` → status of Postgres and Neo4j; reports explicitly when the
  AuraDB instance is paused.
- Both systems are served from the same process using the same retrievers and
  answer chain as the evaluation.

### Deployment (GCP)

- One Docker image, deployed to **Cloud Run** (scale to zero) with a
  `scripts/deploy.sh` wrapping `gcloud run deploy`.
- **Cloud SQL for Postgres** (pgvector enabled) via the Cloud SQL Python
  connector.
- **AuraDB Free** for Neo4j, reached over its public `neo4j+s://` URI.
- API keys and database credentials in **Secret Manager**, mounted as env vars.
- The batch pipeline (ingest → eval) runs from the developer machine via the
  CLI against the same cloud databases; only the serving path is deployed.
- README documents resuming a paused AuraDB instance before a demo.

### Observability (Langfuse)

- Langfuse Cloud (free tier) via its LangChain callback handler and decorator
  SDK.
- Traces every LLM call (extraction, drafting, answering, judging), every
  retrieval call, and every `/ask` request, tagged with `system`,
  `question_id` (during eval) and `run_id`.

## 8. Project structure

```
kafka-atlas/
  pyproject.toml            # uv
  config.yaml
  docker-compose.yml        # Postgres+pgvector, Neo4j (local dev)
  Dockerfile
  migrations/               # SQL
  scripts/deploy.sh
  src/kgrag/
    ingest/  store/  extract/
    resolve/   interface.py  exact_match.py  resolver.py (USER)
    graph/     build.py  kg.py (KnowledgeGraph)
    retrieve/  types.py  vector.py  graph_retriever.py (USER)
    answer/  questions/  eval/
    api/       main.py (FastAPI)
    observability.py        # Langfuse setup
    cli.py
  tests/  tests/fixtures/
  data/      # gitignored: raw mbox, LLM cache exports
  results/
```

Stages are separate CLI commands: `ingest → extract → resolve → build → index →
questions → eval → report`. Each reads prior outputs from the databases and is
idempotent, so changing the resolver requires only `resolve → build → eval`.

## 9. Error handling

- API calls: retry with exponential backoff; failed batch items re-queued.
- LLM outputs: schema-validated; one retry on failure, then logged to
  `failures` and skipped.
- Malformed emails: logged and skipped; counts shown in the report.
- Cost guard: `extract` and `questions` print a token-cost estimate and require
  confirmation above a configurable threshold.
- `/ask`: returns 503 with a clear message when a database is unreachable or
  AuraDB is paused; 400 on an unknown `system`.

## 10. Testing

- Unit tests for Claude-written components: parsing, cleaning, regex, Jira
  parser, metrics, bootstrap.
- Contract and quality tests for the user's resolver (pure Python) and
  retriever (Neo4j testcontainer) (§4, §5).
- API tests with FastAPI's test client against the local `docker compose`
  stack.
- End-to-end smoke test on the dev fixture using recorded LLM and embedding
  responses against the local `docker compose` stack: no API spend, runs in
  under a minute.

## 11. Stretch goals

- Run the same pipeline on the MultiHop-RAG benchmark as an external check.
- Hybrid (BM25 + vector) baseline as a stronger comparison point.
- CI (GitHub Actions) running unit and contract tests.

## 12. Known limitations (to carry into the README)

- Single corpus, single time window; results may not generalize.
- Questions are LLM-drafted from header/regex-linked evidence, so they
  over-represent relations that are visible through explicit IDs and threads.
- The LLM judge is validated on only ~50 labels.
- The user's retriever is tuned on the dev split by the same person who
  reviewed the questions.
