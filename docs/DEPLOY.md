# Deploying Kafka Atlas to GCP

Cloud Run runs the API (`kgrag serve`), Cloud SQL Postgres (with `vector`) holds documents and
embeddings, and AuraDB holds the graph. Secrets live in Secret Manager; Langfuse Cloud receives traces.

## Graph size vs AuraDB Free

Current local graph (`GraphMeta`, resolver `user`, built 2026-10-21): **24,376 nodes, 54,127 relationships**.
AuraDB Free has historically allowed 200,000 nodes and 400,000 relationships; confirm the current
limits in the Aura console before loading. If the limits are lower than the numbers above, stop and
use AuraDB Professional instead.

## Prerequisites (you create these; they use your accounts and billing)

1. A GCP project with billing, and APIs enabled: `run`, `cloudbuild`, `artifactregistry`, `sqladmin`, `secretmanager`.
2. An Artifact Registry Docker repo named `kafka-atlas` in your region.
3. A Cloud SQL Postgres instance (use 16 or newer). Create the database `kgrag` and a user `kgrag`.
   Migration 001 runs `create extension vector`; the user must be allowed to (the console-created user is).
4. An AuraDB instance; note its `neo4j+s://...` URI and password.
5. A Langfuse Cloud project (public and secret key).
6. Secret Manager secrets (names exactly): `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `NEO4J_PASSWORD`,
   `DB_PASS`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `KGRAG_API_KEY` (a long random string clients send as `X-API-Key`).
   Grant the Cloud Run runtime service account `Secret Manager Secret Accessor` and `Cloud SQL Client`.
7. Locally: `gcloud auth login`, `cloud-sql-proxy`, and `uv`.
8. `models.embed` in `config.yaml` must be a real Voyage model (not `voyage-TBD-...`) before `index` and `/ask` vector search work.

## Commands, in order

```bash
export INSTANCE_CONNECTION_NAME=PROJECT:REGION:INSTANCE
export NEO4J_URI=neo4j+s://XXXX.databases.neo4j.io

# 1. Load data from your laptop (prints the plan and asks first; DRY_RUN=1 to only print)
DB_PASS=... NEO4J_PASSWORD=... FROM=YYYY-MM TO=YYYY-MM scripts/load_cloud.sh PROJECT REGION

# 2. Build and deploy (prints the plan and asks first; DRY_RUN=1 to only print)
scripts/deploy.sh PROJECT REGION

# 3. Verify (the service is IAM-protected by default; set ALLOW_UNAUTHENTICATED=1 to make it public)
URL=$(gcloud run services describe kafka-atlas --project PROJECT --region REGION --format 'value(status.url)')
AUTH="Authorization: Bearer $(gcloud auth print-identity-token)"
curl -s -H "$AUTH" $URL/health
curl -s -H "$AUTH" -H "X-API-Key: $KGRAG_API_KEY" -H 'content-type: application/json' \
  -d '{"question":"...","system":"vector"}' $URL/ask      # and again with "graph"
```
Then check the trace appears in Langfuse. When `KGRAG_API_KEY` is set, `/ask` returns 401 without a matching
`X-API-Key`; `/health` stays open.

Overridable env vars: `deploy.sh` takes `MAX_INSTANCES` (default 2), `CONCURRENCY` (10), `DB_USER`/`DB_NAME`
(`kgrag`), `NEO4J_USER`, `LANGFUSE_HOST`, `REPO`, `SERVICE`. `load_cloud.sh` takes `PROXY_PORT` (5433), `RESOLVER` (`exact`).

## Resuming a paused AuraDB

AuraDB Free pauses after inactivity (and deletes instances left paused for long enough). If `/health` returns
503 with `"neo4j": "unavailable"`, open the Aura console, select the instance and click Resume; wait until it
is Running, then retry. If it was deleted, recreate it, update `NEO4J_URI` / the `NEO4J_PASSWORD` secret,
rerun `scripts/load_cloud.sh` (only `build` is needed for the graph) and `scripts/deploy.sh`.

## Cost notes

- Cloud SQL bills for as long as the instance runs, even with zero traffic. Stop it (`gcloud sql instances patch INSTANCE --activation-policy NEVER`)
  or delete it when not demoing.
- Cloud Run scales to zero (`--min-instances 0`), so idle cost is near zero but the first request is a cold start.
- `/ask` calls paid APIs (Anthropic, Voyage). Spend is bounded by `--max-instances` x `--concurrency`
  (default 2 x 10) and by the optional `X-API-Key`; keep the service non-public unless you set `KGRAG_API_KEY`.
- Cloud Build and Artifact Registry storage are small but not free; delete old images when done.
