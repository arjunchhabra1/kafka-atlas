from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def split_tokens(text: str, max_tokens: int) -> list[str]:
    """Split text at token boundaries into pieces that each re-count to <= max_tokens."""
    enc = _encoding()
    ids = enc.encode(text)
    pieces: list[str] = []
    i = 0
    while i < len(ids):
        n = min(max_tokens, len(ids) - i)
        piece = enc.decode(ids[i : i + n])
        while n > 1 and len(enc.encode(piece)) > max_tokens:
            n -= 1
            piece = enc.decode(ids[i : i + n])
        pieces.append(piece)
        i += n
    return pieces
