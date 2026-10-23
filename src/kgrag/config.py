import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ModelConfig(BaseModel):
    extract: str
    draft: str
    answer: str
    judge: str
    embed: str


class ModelPrice(BaseModel):
    input: float
    output: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KGRAG_", env_nested_delimiter="__", yaml_file=CONFIG_PATH
    )

    db_mode: Literal["dsn", "cloudsql"] = "dsn"  # cloudsql: connect via the Cloud SQL connector
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    models: ModelConfig
    budget_tokens: int = 4000
    chunk_tokens: int = 500
    cost_guard_usd: float = 5.0
    prompt_versions: dict[str, str] = {}
    pricing: dict[str, ModelPrice] = {}
    batch_discount: float = 0.5
    data_dir: Path = Path("data")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init > env > .env > YAML (env vars override YAML).
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if s.db_mode == "cloudsql":
        _refuse_dev_defaults(s)
    return s


def _refuse_dev_defaults(s: Settings) -> None:
    """A cloud deployment must not silently fall back to the local-dev values in config.yaml."""
    if s.neo4j_password == "devpassword":
        raise ValueError("cloudsql mode: neo4j_password is the dev default; set KGRAG_NEO4J_PASSWORD")
    if any(h in s.neo4j_uri for h in ("localhost", "127.0.0.1", "[::1]")):
        raise ValueError("cloudsql mode: neo4j_uri points at localhost; set KGRAG_NEO4J_URI")


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def build_hash(s: Settings) -> str:
    """What the stored graph and documents depend on: extraction model and prompt, chunking.

    Never credentials, DSNs, pricing or guards, so it is stable across machines and secrets.
    """
    from kgrag.extract.prompts import PROMPT_VERSION

    return _digest({
        "extract_model": s.models.extract,
        "extract_prompt": s.prompt_versions.get("extract", PROMPT_VERSION),
        "chunk_tokens": s.chunk_tokens,
    })


def run_hash(s: Settings, graph_config=None) -> str:
    """What an eval run's numbers depend on beyond the data: answer/judge/embedding models,
    token budget, answer/judge prompts and (graph runs) the retriever config's fingerprint.

    `graph_config` is a GraphRetrieverConfig or None (vector runs, or the system-independent
    part shared by all runs).
    """
    from kgrag.answer.prompt import ANSWER_PROMPT_VERSION
    from kgrag.eval.judge import JUDGE_PROMPT_VERSION

    return _digest({
        "answer_model": s.models.answer, "judge_model": s.models.judge,
        "embed_model": s.models.embed, "budget_tokens": s.budget_tokens,
        "answer_prompt": s.prompt_versions.get("answer", ANSWER_PROMPT_VERSION),
        "judge_prompt": s.prompt_versions.get("judge", JUDGE_PROMPT_VERSION),
        "graph_config": None if graph_config is None else graph_config.fingerprint(),
    })


def load_env(path: Path | None = None) -> None:
    """Load the project-root `.env` into os.environ without overriding exported variables."""
    from dotenv import load_dotenv

    load_dotenv(path if path is not None else CONFIG_PATH.parent / ".env", override=False)
