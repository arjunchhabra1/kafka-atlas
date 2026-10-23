#!/usr/bin/env bash
# Copy the LOCAL evaluated state into Cloud SQL (via the Cloud SQL Auth Proxy), then build the
# graph in AuraDB from it. Nothing is re-extracted, re-resolved or (by default) re-embedded:
# the cloud serves exactly the corpus, extractions, resolutions and vectors that were evaluated.
#
# usage: (export DB_PASS and NEO4J_PASSWORD first, see docs/DEPLOY.md)
#        INSTANCE_CONNECTION_NAME=proj:region:inst NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
#        scripts/load_cloud.sh PROJECT REGION
#
# Steps: pg_dump (from the local docker-compose Postgres) the data of the pipeline tables plus
# the pgvector tables -> start cloud-sql-proxy -> apply migrations to the (empty) Cloud SQL
# database -> restore the dump in one transaction -> `kgrag build --resolver $RESOLVER` against
# AuraDB. REINDEX=1 skips the vector tables and runs `kgrag index` against Cloud SQL instead
# (re-embeds everything: Voyage cost). The target database must be empty (fresh instance/db).
#
# Needs: docker compose (local stack running), psql >= 16 on PATH (e.g. `brew install libpq`),
# gcloud (logged in), cloud-sql-proxy, uv. Prints the plan and asks first (YES=1 skips the
# prompt, DRY_RUN=1 only prints). Passwords are read from the environment and never printed.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROJECT REGION   (both are required; see docs/DEPLOY.md)" >&2
  exit 2
fi
PROJECT=$1
REGION=$2
cd "$(dirname "$0")/.."   # docker compose and uv run from the repo root

: "${INSTANCE_CONNECTION_NAME:?set INSTANCE_CONNECTION_NAME (project:region:instance)}"
: "${NEO4J_URI:?set NEO4J_URI (the AuraDB neo4j+s:// URI)}"
: "${DB_PASS:?set DB_PASS (Cloud SQL user password)}"
: "${NEO4J_PASSWORD:?set NEO4J_PASSWORD (AuraDB password)}"

DB_USER=${DB_USER:-kgrag}
DB_NAME=${DB_NAME:-kgrag}
NEO4J_USER=${NEO4J_USER:-neo4j}
PROXY_PORT=${PROXY_PORT:-5433}
RESOLVER=${RESOLVER:-user}
REINDEX=${REINDEX:-0}
LOCAL_SERVICE=${LOCAL_SERVICE:-postgres}   # docker compose service holding the evaluated state
LOCAL_DB=${LOCAL_DB:-kgrag}
LOCAL_USER=${LOCAL_USER:-kgrag}

case "$RESOLVER" in exact|user) ;; *) echo "RESOLVER must be exact or user" >&2; exit 2;; esac

# Pipeline tables (schema comes from migrations; only their rows are copied).
DATA_TABLES=(emails threads jira_events documents llm_cache mentions relations resolutions
             questions evidence_attempts)
# pgvector tables are created by langchain-postgres, not by migrations: schema + rows.
VECTOR_TABLES=(langchain_pg_collection langchain_pg_embedding)

STEPS=("pg_dump --data-only: ${DATA_TABLES[*]}  (local compose service '$LOCAL_SERVICE')")
if [[ "$REINDEX" == "1" ]]; then
  STEPS+=("(REINDEX=1: vector tables NOT copied; re-embedded in Cloud SQL below)")
else
  STEPS+=("pg_dump schema+data: ${VECTOR_TABLES[*]}")
fi
STEPS+=(
  "cloud-sql-proxy --port $PROXY_PORT $INSTANCE_CONNECTION_NAME   (background, stopped on exit)"
  "apply migrations to Cloud SQL; refuse unless its pipeline/vector tables are empty"
  "psql --single-transaction: restore the dump into Cloud SQL"
  "uv run kgrag build --resolver $RESOLVER --yes   (WIPES and rebuilds the AuraDB graph)"
)
if [[ "$REINDEX" == "1" ]]; then
  STEPS+=("uv run kgrag index   (Voyage cost: embeds every chunk and node name)")
fi

echo "Project: $PROJECT   Region: $REGION"
echo "Source:    local docker compose service '$LOCAL_SERVICE' (db $LOCAL_DB) -- the evaluated state"
echo "Cloud SQL: $INSTANCE_CONNECTION_NAME via cloud-sql-proxy on 127.0.0.1:$PROXY_PORT (user $DB_USER, db $DB_NAME)"
echo "AuraDB:    $NEO4J_URI (user $NEO4J_USER)"
echo "No extraction, resolution or (unless REINDEX=1) embedding is re-run: no Anthropic/Voyage cost."
echo
echo "Will run, in order:"
for s in "${STEPS[@]}"; do echo "  $s"; done
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: nothing executed."
  exit 0
fi

for tool in docker psql cloud-sql-proxy uv; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
PSQL_MAJOR=$(psql --version | sed -E 's/[^0-9]*([0-9]+).*/\1/')
if (( PSQL_MAJOR < 16 )); then
  echo "psql $PSQL_MAJOR is too old for a pg16 dump; install a client >= 16" >&2
  exit 1
fi

local_sql() {
  docker compose exec -T "$LOCAL_SERVICE" psql -U "$LOCAL_USER" -d "$LOCAL_DB" -tAc "$1"
}
echo "Local state:"
for t in emails documents llm_cache mentions relations questions; do
  echo "  $t: $(local_sql "select count(*) from $t")"
done
N_RES=$(local_sql "select count(*) from resolutions where resolver = '$RESOLVER'")
echo "  resolutions ($RESOLVER): $N_RES"
if [[ "$N_RES" == "0" ]]; then
  echo "no stored '$RESOLVER' resolution locally; run \`kgrag resolve --resolver $RESOLVER\` first" >&2
  exit 1
fi
if [[ "$REINDEX" != "1" ]]; then
  if [[ "$(local_sql "select to_regclass('langchain_pg_embedding') is not null")" != "t" ]]; then
    echo "no local vector index (langchain_pg_embedding); run \`kgrag index\` locally first," >&2
    echo "or pass REINDEX=1 to embed in Cloud SQL instead (Voyage cost)" >&2
    exit 1
  fi
  echo "  vectors: $(local_sql "select count(*) from langchain_pg_embedding")"
fi
echo

if [[ "${YES:-0}" != "1" ]]; then
  read -r -p "Proceed? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 1; }
fi

DUMP=$(mktemp -t kgrag-load.XXXXXX)
chmod 600 "$DUMP"
PROXY_PID=""
cleanup() {
  rm -f "$DUMP"
  [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

TABLE_ARGS=()
for t in "${DATA_TABLES[@]}"; do TABLE_ARGS+=(-t "public.$t"); done
echo ">>> pg_dump (data only) of the pipeline tables"
docker compose exec -T "$LOCAL_SERVICE" pg_dump -U "$LOCAL_USER" -d "$LOCAL_DB" \
  --data-only --no-owner --no-privileges "${TABLE_ARGS[@]}" > "$DUMP"
if [[ "$REINDEX" != "1" ]]; then
  VEC_ARGS=()
  for t in "${VECTOR_TABLES[@]}"; do VEC_ARGS+=(-t "public.$t"); done
  echo ">>> pg_dump (schema + data) of the vector tables"
  docker compose exec -T "$LOCAL_SERVICE" pg_dump -U "$LOCAL_USER" -d "$LOCAL_DB" \
    --no-owner --no-privileges "${VEC_ARGS[@]}" >> "$DUMP"
fi
echo "    dump: $(wc -c < "$DUMP" | tr -d ' ') bytes"

ENCODED_PASS=$(python3 -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["DB_PASS"], safe=""))')
export KGRAG_DB_MODE=dsn
export KGRAG_PG_DSN="postgresql://$DB_USER:$ENCODED_PASS@127.0.0.1:$PROXY_PORT/$DB_NAME"
export KGRAG_NEO4J_URI="$NEO4J_URI" KGRAG_NEO4J_USER="$NEO4J_USER" KGRAG_NEO4J_PASSWORD="$NEO4J_PASSWORD"
export PGPASSWORD="$DB_PASS"   # psql reads it from the environment, never the command line
CLOUD_PSQL=(psql -h 127.0.0.1 -p "$PROXY_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1)

cloud-sql-proxy --project "$PROJECT" --port "$PROXY_PORT" "$INSTANCE_CONNECTION_NAME" &
PROXY_PID=$!
UP=0
for _ in $(seq 1 30); do
  kill -0 "$PROXY_PID" 2>/dev/null || { echo "cloud-sql-proxy exited early; check gcloud login and instance name" >&2; exit 1; }
  if (exec 3<>"/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null; then UP=1; break; fi
  sleep 1
done
if [[ "$UP" != "1" ]]; then
  echo "cloud-sql-proxy did not open 127.0.0.1:$PROXY_PORT within 30s" >&2
  exit 1
fi

echo ">>> apply migrations to Cloud SQL"
uv run python -c 'from kgrag.store.db import apply_migrations, connect
c = connect(); print("applied:", apply_migrations(c) or "nothing new"); c.close()'

echo ">>> check the target is empty"
NONEMPTY=""
for t in "${DATA_TABLES[@]}"; do
  n=$("${CLOUD_PSQL[@]}" -tAc "select count(*) from $t")
  if [[ "$n" != "0" ]]; then NONEMPTY+=" $t($n)"; fi
done
if [[ "$REINDEX" != "1" ]] && \
   [[ "$("${CLOUD_PSQL[@]}" -tAc "select to_regclass('langchain_pg_embedding') is not null")" == "t" ]]; then
  NONEMPTY+=" langchain_pg_embedding(exists)"
fi
if [[ -n "$NONEMPTY" ]]; then
  echo "Cloud SQL database $DB_NAME is not empty:$NONEMPTY" >&2
  echo "recreate the database (see docs/DEPLOY.md) and run this script again" >&2
  exit 1
fi

echo ">>> restore into Cloud SQL (one transaction)"
"${CLOUD_PSQL[@]}" --single-transaction -q -f "$DUMP" >/dev/null
for t in emails documents resolutions; do
  echo "    $t: $("${CLOUD_PSQL[@]}" -tAc "select count(*) from $t")"
done

echo ">>> kgrag build --resolver $RESOLVER --yes"
uv run kgrag build --resolver "$RESOLVER" --yes
if [[ "$REINDEX" == "1" ]]; then
  echo ">>> kgrag index"
  uv run kgrag index
fi
echo "Load complete: Cloud SQL holds the local evaluated state; AuraDB holds the '$RESOLVER' graph."
