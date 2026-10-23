"""The suite must not pick up the project's real .env."""

import os

from typer.testing import CliRunner

from kgrag.cli import app


def test_project_dotenv_does_not_populate_environ_in_suite():
    from kgrag.config import load_env

    load_env()  # no path: the project-root .env, neutralised by the autouse fixture
    CliRunner().invoke(app, ["--help"])  # the CLI callback also calls load_env()
    for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "KGRAG_API_KEY",
              "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        assert k not in os.environ
