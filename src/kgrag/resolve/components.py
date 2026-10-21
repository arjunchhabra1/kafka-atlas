"""Component resolution: normalize spellings into a key, group by key, name each group."""

import re
from collections import Counter, defaultdict

from kgrag.extract.schema import Mention
from kgrag.resolve.names import fold

_PREFIX_RE = re.compile(r"^(the|apache|kafka)[\s_-]+", re.IGNORECASE)
_SOURCE_EXT_RE = re.compile(r"\.(java|scala|kt|py)$", re.IGNORECASE)
_QUALIFIED_RE = re.compile(r"^[a-z_][\w]*(\.[a-z_][\w]*)+\.([A-Z]\w*)$")  # org.example.Foo
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
# A version worth stripping: dotted ("3.4") or multi-digit ("12"), optionally "v"-prefixed.
_VERSION_RE = re.compile(r"^(?P<base>.*\S)\s+(?P<version>v?(\d+\.\d+(\.\d+)*|\d{2,}))$", re.IGNORECASE)


def _strip_decorations(surface: str) -> str:
    """Drop a source-file extension, a Java package prefix and leading "the/apache/kafka" words."""
    text = _SOURCE_EXT_RE.sub("", surface.strip())
    qualified = _QUALIFIED_RE.match(text)
    if qualified:
        text = qualified.group(2)
    while True:
        shorter = _PREFIX_RE.sub("", text, count=1)
        if shorter == text or not shorter.strip() or shorter[0].isdigit():
            return text
        text = shorter


def component_key(surface: str) -> str:
    """Spelling-insensitive key: "Widget Maker 2" == "widget-maker2", "Kafka Widget" == "Widget".

    Class names stay distinct ("FooExtension" != "FooExtensionContext"), and a plural "s" is
    only dropped from ordinary words, never from a CamelCase identifier ("Serdes").
    """
    text = _strip_decorations(surface)
    key = re.sub(r"[\s_-]+", "", fold(text))
    if not _CAMEL_RE.search(text) and len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


def split_version(surface: str) -> tuple[str, str | None]:
    """Split a trailing version: "Widget 12" -> ("Widget", "12"). Single digits are kept
    ("Widget 2" is often a different product, not a version)."""
    match = _VERSION_RE.match(surface.strip())
    if not match:
        return surface.strip(), None
    return match.group("base"), match.group("version")


def _canonical(mentions: list[Mention]) -> str:
    """Most frequent surface; ties go to the longest, then alphabetical."""
    counts = Counter(m.surface.strip() for m in mentions)
    return min(counts, key=lambda s: (-counts[s], -len(s), s))


def _version_base(key: str, by_key: dict[str, list[Mention]]) -> str:
    """The key this versioned group should join, or `key` itself.

    "Widget 2.0" joins "Widget2" if that product exists (its major version is part of the
    name), else the bare "Widget" if that exists.
    """
    for surface in sorted({m.surface for m in by_key[key]}):
        base, version = split_version(surface)
        if not version:
            continue
        major = version.lstrip("vV").split(".")[0]
        for target in (component_key(f"{base}{major}"), component_key(base)):
            if target in by_key and target != key:
                return target
    return key


def group_components(mentions: list[Mention]) -> list[tuple[str, list[Mention]]]:
    """Group Component mentions into entities: [(canonical name, members)], sorted by name.

    Mentions sharing a `component_key` are one entity; "Base 12" also joins "Base" when a
    bare "Base" exists (otherwise the versioned spelling stands alone).
    """
    by_key: dict[str, list[Mention]] = defaultdict(list)
    for m in sorted(mentions, key=lambda m: m.mention_id):
        by_key[component_key(m.surface)].append(m)
    merged: dict[str, list[Mention]] = defaultdict(list)
    for key in sorted(by_key):
        merged[_version_base(key, by_key)].extend(by_key[key])
    groups = [(_canonical(ms), sorted(ms, key=lambda m: m.mention_id)) for ms in merged.values()]
    return sorted(groups, key=lambda g: (g[0], g[1][0].mention_id))
