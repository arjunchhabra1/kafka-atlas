#!/usr/bin/env bash
# Load the corpus into Cloud SQL (via the Cloud SQL Auth Proxy) and AuraDB from this machine.
#
# usage: INSTANCE_CONNECTION_NAME=proj:region:inst NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
#        DB_PASS=... NEO4J_PASSWORD=... FROM=2023-01 TO=2024-06 scripts/load_cloud.sh PROJECT REGION
#
# Needs: gcloud (already logged in), cloud-sql-proxy, uv. Prints the plan and asks first
# (YES=1 skips the prompt, DRY_RUN=1 only prints). Passwords are read from the environment and
# never printed. `kgrag extract` reads the local extraction cache; it only calls the API (behind
# its own cost guard) for emails that are not cached.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROJECT REGION   (both are required; see docs/DEPLOY.md)" >&2
  exit 2
fi
PROJECT=$1
REGION=$2

: "${INSTANCE_CONNECTION_NAME:?set INSTANCE_CONNECTION_NAME (project:region:instance)}"
: "${NEO4J_URI:?set NEO4J_URI (the AuraDB neo4j+s:// URI)}"
: "${DB_PASS:?set DB_PASS (Cloud SQL user password)}"
: "${NEO4J_PASSWORD:?set NEO4J_PASSWORD (AuraDB password)}"
: "${FROM:?set FROM (first month, YYYY-MM)}"
: "${TO:?set TO (last month, YYYY-MM)}"

DB_USER=${DB_USER:-kgrag}
DB_NAME=${DB_NAME:-kgrag}
NEO4J_USER=${NEO4J_USER:-neo4j}
PROXY_PORT=${PROXY_PORT:-5433}
RESOLVER=${RESOLVER:-exact}

STEPS=(
  "kgrag ingest --from $FROM --to $TO"
  "kgrag extract"
  "kgrag resolve --resolver $RESOLVER"
  "kgrag build --resolver $RESOLVER --yes"
  "kgrag index"
)

echo "Project: $PROJECT   Region: $REGION"
echo "Cloud SQL: $INSTANCE_CONNECTION_NAME via cloud-sql-proxy on 127.0.0.1:$PROXY_PORT (user $DB_USER, db $DB_NAME)"
echo "AuraDB:    $NEO4J_URI (user $NEO4J_USER)  -- 'build' WIPES and rebuilds this graph"
echo
echo "Will run, in order:"
echo "  cloud-sql-proxy --port $PROXY_PORT $INSTANCE_CONNECTION_NAME   (background, stopped on exit)"
for s in "${STEPS[@]}"; do echo "  uv run $s"; done
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: nothing executed."
  exit 0
fi
if [[ "${YES:-0}" != "1" ]]; then
  read -r -p "Proceed? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 1; }
fi

ENCODED_PASS=$(python3 -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["DB_PASS"], safe=""))')
export KGRAG_DB_MODE=dsn
export KGRAG_PG_DSN="postgresql://$DB_USER:$ENCODED_PASS@127.0.0.1:$PROXY_PORT/$DB_NAME"
export KGRAG_NEO4J_URI="$NEO4J_URI" KGRAG_NEO4J_USER="$NEO4J_USER" KGRAG_NEO4J_PASSWORD="$NEO4J_PASSWORD"

cloud-sql-proxy --project "$PROJECT" --port "$PROXY_PORT" "$INSTANCE_CONNECTION_NAME" &
PROXY_PID=$!
trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  (exec 3<>"/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null && break
  sleep 1
done

for s in "${STEPS[@]}"; do
  echo ">>> $s"
  # shellcheck disable=SC2086
  uv run $s
done
echo "Load complete."
