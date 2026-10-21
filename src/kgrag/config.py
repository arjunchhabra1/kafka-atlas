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
    return Settings()


def config_hash(s: Settings) -> str:
    payload = json.dumps(s.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
