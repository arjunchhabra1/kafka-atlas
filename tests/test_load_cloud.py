"""scripts/load_cloud.sh restores the local evaluated state; it never re-pays extraction."""

import os
import subprocess

from tests.conftest import ROOT

SCRIPT = ROOT / "scripts" / "load_cloud.sh"
ENV = {"INSTANCE_CONNECTION_NAME": "p:r:i", "NEO4J_URI": "neo4j+s://x.databases.neo4j.io",
       "DB_PASS": "db-secret-value", "NEO4J_PASSWORD": "neo-secret-value", "DRY_RUN": "1"}


def _plan(**extra):
    env = {**os.environ, **ENV, **extra}
    out = subprocess.run(["bash", str(SCRIPT), "proj", "region"], env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout


def test_dry_run_plan_restores_local_state_and_builds_user_graph():
    plan = _plan()
    assert "kgrag extract" not in plan and "kgrag ingest" not in plan
    assert "kgrag resolve" not in plan and "kgrag index" not in plan
    assert "pg_dump --data-only" in plan and "resolutions" in plan and "llm_cache" in plan
    assert "langchain_pg_embedding" in plan
    assert "kgrag build --resolver user --yes" in plan
    assert "db-secret-value" not in plan and "neo-secret-value" not in plan
    assert "nothing executed" in plan


def test_reindex_and_resolver_are_explicit():
    plan = _plan(REINDEX="1", RESOLVER="exact")
    assert "kgrag index" in plan and "Voyage cost" in plan
    assert "kgrag build --resolver exact --yes" in plan
    bad = subprocess.run(["bash", str(SCRIPT), "proj", "region"], capture_output=True, text=True,
                         env={**os.environ, **ENV, "RESOLVER": "fuzzy"})
    assert bad.returncode == 2 and "RESOLVER must be" in bad.stderr
