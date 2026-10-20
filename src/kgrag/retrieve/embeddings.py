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


def get_embeddings() -> Embeddings:
    """Process-wide embeddings; the Voyage client is built lazily on first use."""
    global _embeddings
    if _embeddings is None:
        model = get_settings().models.embed
        if model.startswith(PLACEHOLDER_MODEL_PREFIX) or not os.environ.get("VOYAGE_API_KEY"):
            raise EmbeddingsNotConfiguredError(_SETUP_HINT)
        from langchain_voyageai import VoyageAIEmbeddings

        _embeddings = VoyageAIEmbeddings(model=model)
    return _embeddings


def set_embeddings_for_tests(e: Embeddings | None) -> None:
    global _embeddings
    _embeddings = e


class RecordedEmbeddings(Embeddings):
    """Replays vectors recorded by scripts/record_embeddings.py; unknown text is an error."""

    def __init__(self, path: Path):
        self.vectors: dict[str, list[float]] = json.loads(Path(path).read_text())

    def _get(self, text: str) -> list[float]:
        try:
            return self.vectors[text]
        except KeyError:
            raise KeyError(f"text not in recorded embeddings: {text[:80]!r}") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get(text)


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
