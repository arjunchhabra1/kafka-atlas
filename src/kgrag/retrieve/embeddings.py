"""Embedding model access: the shared Voyage instance, plus offline stand-ins for tests."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from langchain_core.embeddings import Embeddings

from kgrag.config import get_settings

PLACEHOLDER_MODEL_PREFIX = "voyage-TBD"
_SETUP_HINT = "set models.embed in config.yaml and VOYAGE_API_KEY in .env"

_embeddings: Embeddings | None = None


class EmbeddingsNotConfiguredError(RuntimeError):
    pass


def embeddings_configured() -> bool:
    """True when a real embedding model and API key are set (never calls the provider)."""
    model = get_settings().models.embed
    return not model.startswith(PLACEHOLDER_MODEL_PREFIX) and bool(os.environ.get("VOYAGE_API_KEY"))


def get_embeddings() -> Embeddings:
    """Process-wide embeddings; the Voyage client is built lazily on first use."""
    global _embeddings
    if _embeddings is None:
        if not embeddings_configured():
            raise EmbeddingsNotConfiguredError(_SETUP_HINT)
        from langchain_voyageai import VoyageAIEmbeddings

        _embeddings = VoyageAIEmbeddings(model=get_settings().models.embed)
    return _embeddings


def set_embeddings_for_tests(e: Embeddings | None) -> None:
    global _embeddings
    _embeddings = e


class RecordedEmbeddings(Embeddings):
    """Replays vectors from scripts/record_embeddings.py: {"model", "documents", "queries"}.

    Documents and queries are separate because Voyage embeds them with different input types.
    Unknown text is an error. `model` is the model the vectors were recorded with.
    """

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text())
        self.model: str = data.get("model", "")
        self.documents: dict[str, list[float]] = data["documents"]
        self.queries: dict[str, list[float]] = data["queries"]

    @staticmethod
    def _get(table: dict[str, list[float]], kind: str, text: str) -> list[float]:
        try:
            return table[text]
        except KeyError:
            raise KeyError(f"text not in recorded {kind} embeddings: {text[:80]!r}") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get(self.documents, "document", t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get(self.queries, "query", text)


class HashEmbeddings(Embeddings):
    """Deterministic L2-normalised bag-of-words hash vectors: texts sharing words are similar."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            v[0], norm = 1.0, 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def offline_embeddings(recorded: Path, model: str | None = None) -> Embeddings:
    """The tests' stand-in: RecordedEmbeddings when `recorded` exists *and* was recorded with
    `model` (default models.embed), otherwise HashEmbeddings. A recording made with another
    model is stale and is not used."""
    model = model or get_settings().models.embed
    if Path(recorded).exists():
        rec = RecordedEmbeddings(recorded)
        if rec.model == model:
            return rec
    return HashEmbeddings(dim=64)
