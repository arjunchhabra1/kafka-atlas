"""Person-name helpers for entity resolution: cleaning, parsing and comparing names.

Everything here is a small pure function so it can be unit-tested on its own.
"""

import re
import unicodedata
from collections.abc import Collection
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
# A name right after one of these words is addressed or talked about, never a sign-off.
_GREETING_WORDS = {"hi", "hello", "hey", "dear", "welcome", "congrats", "congratulations"}
_ADDRESSING_WORDS = _GREETING_WORDS | {"ping", "cc", "ask", "asked", "by", "to", "from", "with",
                                       "for", "of", "about", "as", "and", "or"}
_CLOSING_WORDS = {"thanks", "thx", "cheers", "best", "regards", "sincerely"}
_CO_SIGNERS_RE = re.compile(r"^\s*(&|and)(\s+[\w']+){1,3}\W*$")  # "Jane & Bo", "Jane and the X team"
_SELF_INTRO_RE = re.compile(r"\b(i'm|i am|this is|my name is)\s*$")
_NOT_NAME_TOKENS = {"and", "or"}
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
        """A first name plus a real (not abbreviated) surname, and not a list of people."""
        return (self.last is not None and not self.last_is_initial and len(self.tokens) <= 4
                and not _NOT_NAME_TOKENS & set(self.tokens))


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


def _bare_name(surface: str) -> str:
    """The folded name in a surface, without leading fillers: "Hi Jane" -> "jane", "-Jane" -> "jane"."""
    words = fold(" ".join(surface.split())).strip("-~ ").split()
    while len(words) > 1 and re.sub(r"\W", "", words[0]) in _LEADING_FILLERS:
        words = words[1:]
    return " ".join(words).strip("-~ ")


def name_role(context: str, surface: str, surnames: Collection[str] = ()) -> str:
    """How a first name is used in its sentence: "self", "greeting", "other" or "absent".

    "self": the writer naming themselves. Either a self-introduction ("I'm Jane, ...") or
    the name closes the sentence -- nothing after it except one of `surnames`, co-signers
    ("Jane & Bo") or "on behalf of ..." -- as in "Thanks,\nJane", "Hope this helps, Jane",
    "-- Jane".
    "greeting": "Hi Jane", "Dear Jane" -- someone other than the writer, for certain.
    "other": the name is addressed or talked about: more text follows it ("Thanks, Jane.
    Nice work", "Jane's comment", "Jane: can you..."), it is an "@Jane" handle, or it follows
    a greeting or a preposition ("Hi Jane", "as suggested by Jane", "Bo and Jane").
    "absent": the extractor's sentence does not contain the name, so it tells us nothing.
    """
    name = _bare_name(surface)
    text = fold(context)
    found = list(re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text)) if name else []
    if not found:
        return "absent"
    at = found[-1].start()  # the last occurrence: sign-offs come at the end
    before, after = text[:at], text[at + len(name):]
    if before.rstrip().endswith("@"):
        return "other"
    if _SELF_INTRO_RE.search(before):
        return "self"
    rest = re.findall(r"[\w']+", after)
    previous_word = (re.findall(r"[\w']+", before) or [""])[-1]
    if previous_word in _GREETING_WORDS:
        return "greeting"
    starts_line = not before.rsplit("\n", 1)[-1].strip(" ,-~")
    co_signed = bool(_CO_SIGNERS_RE.match(after)) and (starts_line or previous_word in _CLOSING_WORDS)
    closes_sentence = (not rest or (len(rest) == 1 and rest[0] in surnames)
                       or rest[:3] == ["on", "behalf", "of"] or co_signed)
    return "self" if closes_sentence and previous_word not in _ADDRESSING_WORDS else "other"


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
