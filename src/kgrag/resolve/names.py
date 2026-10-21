"""Person-name helpers for entity resolution: cleaning, parsing and comparing names.

Everything here is a small pure function so it can be unit-tested on its own.
"""

import re
import unicodedata
from dataclasses import dataclass

# Generic English diminutives -> formal first name. Deliberately small: each entry is a
# common nickname, not something tuned to a particular dataset.
FORMAL_FIRST_NAME: dict[str, str] = {
    "chris": "christopher",
    "dave": "david",
    "mike": "michael",
    "matt": "matthew",
    "nick": "nicholas",
    "tom": "thomas",
    "bill": "william",
    "will": "william",
    "rob": "robert",
    "bob": "robert",
    "jim": "james",
    "joe": "joseph",
    "dan": "daniel",
    "steve": "steven",
    "alex": "alexander",
    "andy": "andrew",
    "ben": "benjamin",
    "sam": "samuel",
    "tony": "anthony",
}

# Words that can precede a name in an extracted surface ("Hi Chris", "Other Chris").
_LEADING_FILLERS = {"hi", "hello", "hey", "dear", "thanks", "thank", "other", "mr", "mrs", "ms", "dr"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd"}
_GREETING_RE = re.compile(r"^\W*(hi|hello|hey|dear|greetings|good (morning|afternoon|evening))\b")
_RELAY_RE = re.compile(r"\s+via\s+\S+\s*$", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def fold(text: str) -> str:
    """Lower-case and strip accents, so "José" and "jose" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_relay_name(raw: str) -> bool:
    """True for "Jane Doe via dev": the list rewrote the sender, so its address is shared."""
    return bool(_RELAY_RE.search(raw))


def clean_display_name(raw: str) -> str:
    """Turn a header/Jira display name into "First Last" form.

    Drops a "via <list>" suffix, quotes, parenthesised notes and Jr/Sr-style suffixes,
    and reorders "Last, First" into "First Last".
    """
    name = _RELAY_RE.sub("", raw)
    name = re.sub(r"\([^)]*\)", " ", name).replace('"', " ")
    parts = [p.strip() for p in name.split(",") if p.strip()]
    parts = [p for p in parts if fold(p).strip(".") not in _NAME_SUFFIXES]
    if len(parts) == 2:
        parts = [parts[1], parts[0]]
    return " ".join(" ".join(parts).split())


@dataclass(frozen=True)
class PersonName:
    """A folded, tokenized name. Tokens keep their order; hyphens are removed inside tokens."""

    tokens: tuple[str, ...]

    @property
    def first(self) -> str | None:
        return self.tokens[0] if self.tokens else None

    @property
    def last(self) -> str | None:
        return self.tokens[-1] if len(self.tokens) > 1 else None

    @property
    def last_is_initial(self) -> bool:
        """ "Jane D." -- a first name plus a surname initial."""
        return self.last is not None and len(self.last) == 1

    @property
    def is_full(self) -> bool:
        """A first name plus a real (not abbreviated) surname."""
        return self.last is not None and not self.last_is_initial


def parse_name(raw: str) -> PersonName:
    """Parse any name surface ("Hi Jane", "Jane Q. Doe", "Doe, Jane") into a PersonName.

    Single-letter middle initials are dropped; leading greetings are stripped as long as
    something is left.
    """
    text = fold(clean_display_name(raw))
    if "@" not in text:  # "Jane Q.Doe" -> "Jane Q Doe"; leave addresses in one piece
        text = text.replace(".", " ")
    tokens = [re.sub(r"[^\w]", "", t) for t in text.split()]
    tokens = [t for t in tokens if t]
    while len(tokens) > 1 and tokens[0] in _LEADING_FILLERS:
        tokens = tokens[1:]
    if len(tokens) > 2:
        tokens = [tokens[0], *(t for t in tokens[1:-1] if len(t) > 1), tokens[-1]]
    return PersonName(tuple(tokens))


def formal(first: str) -> str:
    """The formal form of a first name ("chris" -> "christopher"); unknown names unchanged."""
    return FORMAL_FIRST_NAME.get(first, first)


def full_key(name: PersonName) -> str | None:
    """Identity key for a full name: formal first name + surname ("bob smith" -> "robert smith")."""
    if not name.is_full:
        return None
    return f"{formal(name.first)} {name.last}"


def first_names_compatible(mentioned: str, known: str) -> bool:
    """Can the first name `mentioned` in text refer to someone known as `known`?

    Asymmetric on purpose: "Chris" may mean a "Christopher", but someone who writes
    "Christopher" is not talking about a person who goes by "Chris".
    """
    if mentioned == known:
        return True
    if mentioned not in FORMAL_FIRST_NAME:
        return False
    return formal(mentioned) == formal(known)


def is_greeting(context: str) -> bool:
    """True if the sentence opens with a salutation ("Hi Jane,"), i.e. it addresses someone."""
    return bool(_GREETING_RE.match(fold(context)))


def address_local_part(address: str) -> str | None:
    """Folded alphanumeric local part of an email address ("Jane.Doe@x.org" -> "janedoe")."""
    if not EMAIL_RE.fullmatch(address.strip()):
        return None
    return re.sub(r"[^\w]|_", "", fold(address.strip().split("@", 1)[0])) or None


def is_surname_typo(a: str, b: str) -> bool:
    """True if two surnames (5+ letters) are equal or one edit apart ("cadonna"/"cadona").

    Short names are excluded: "lee"/"lea" are different people far too often.
    """
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))
