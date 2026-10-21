#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run as `kafka-atlas`.
#
# usage: INSTANCE_CONNECTION_NAME=proj:region:inst NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
#        scripts/deploy.sh PROJECT REGION
#
# Prints the full plan, then asks for confirmation (YES=1 skips the prompt, DRY_RUN=1 only prints).
# Secret values are never read or printed: Cloud Run pulls them from Secret Manager by name.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROJECT REGION   (both are required; see docs/DEPLOY.md)" >&2
  exit 2
fi
PROJECT=$1
REGION=$2

: "${INSTANCE_CONNECTION_NAME:?set INSTANCE_CONNECTION_NAME (project:region:instance of the Cloud SQL instance)}"
: "${NEO4J_URI:?set NEO4J_URI (the AuraDB neo4j+s:// URI)}"

SERVICE=${SERVICE:-kafka-atlas}
REPO=${REPO:-kafka-atlas}                       # Artifact Registry repository (must already exist)
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE"
DB_USER=${DB_USER:-kgrag}
DB_NAME=${DB_NAME:-kgrag}
NEO4J_USER=${NEO4J_USER:-neo4j}
LANGFUSE_HOST=${LANGFUSE_HOST:-https://cloud.langfuse.com}
# Spend protection: /ask triggers paid API calls, so cap scale-out and per-instance concurrency.
MAX_INSTANCES=${MAX_INSTANCES:-2}
CONCURRENCY=${CONCURRENCY:-10}

# Cloud Run secret env var = Secret Manager secret name (latest version).
SECRETS="ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
SECRETS+=",VOYAGE_API_KEY=VOYAGE_API_KEY:latest"
SECRETS+=",KGRAG_NEO4J_PASSWORD=NEO4J_PASSWORD:latest"
SECRETS+=",DB_PASS=DB_PASS:latest"
SECRETS+=",LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC_KEY:latest"
SECRETS+=",LANGFUSE_SECRET_KEY=LANGFUSE_SECRET_KEY:latest"
SECRETS+=",KGRAG_API_KEY=KGRAG_API_KEY:latest"

ENV_VARS="KGRAG_DB_MODE=cloudsql,INSTANCE_CONNECTION_NAME=$INSTANCE_CONNECTION_NAME"
ENV_VARS+=",DB_USER=$DB_USER,DB_NAME=$DB_NAME"
ENV_VARS+=",KGRAG_NEO4J_URI=$NEO4J_URI,KGRAG_NEO4J_USER=$NEO4J_USER,LANGFUSE_HOST=$LANGFUSE_HOST"

BUILD=(gcloud builds submit --project "$PROJECT" --tag "$IMAGE" .)
DEPLOY=(gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE"
  --add-cloudsql-instances "$INSTANCE_CONNECTION_NAME"
  --min-instances 0 --max-instances "$MAX_INSTANCES" --concurrency "$CONCURRENCY"
  --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS")
if [[ "${ALLOW_UNAUTHENTICATED:-0}" == "1" ]]; then
  DEPLOY+=(--allow-unauthenticated)
else
  DEPLOY+=(--no-allow-unauthenticated)
fi

echo "Project: $PROJECT   Region: $REGION   Service: $SERVICE"
echo "Max instances: $MAX_INSTANCES   Concurrency: $CONCURRENCY   Min instances: 0"
echo "Public access: $([[ "${ALLOW_UNAUTHENTICATED:-0}" == "1" ]] && echo "allowed (protected by X-API-Key)" || echo "no (IAM-authenticated callers only)")"
echo "Secrets (names only) from Secret Manager in $PROJECT: ${SECRETS//,/ }"
echo
echo "Will run:"
printf '  %q' "${BUILD[@]}"; echo
printf '  %q' "${DEPLOY[@]}"; echo
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: nothing executed."
  exit 0
fi
if [[ "${YES:-0}" != "1" ]]; then
  read -r -p "Proceed? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 1; }
fi

"${BUILD[@]}"
"${DEPLOY[@]}"
echo
echo "Service URL: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
