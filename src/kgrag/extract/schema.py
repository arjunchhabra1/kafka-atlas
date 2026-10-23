"""Data models and the JSON schema for LLM entity/relation extraction."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

MentionType = Literal["Person", "Component", "KIP", "JiraTicket", "Release"]
RelationType = Literal[
    "PROPOSED", "VOTED", "REVIEWED", "BLOCKS", "AFFECTS", "DEPENDS_ON", "SUPERSEDES"
]
MENTION_TYPES: tuple[str, ...] = MentionType.__args__  # type: ignore[attr-defined]
RELATION_TYPES: tuple[str, ...] = RelationType.__args__  # type: ignore[attr-defined]


class Mention(BaseModel):
    mention_id: str  # `{email_id}:m{n}`
    surface: str
    type: MentionType
    email_id: str
    context: str


class Relation(BaseModel):
    type: RelationType
    source_mention_id: str
    target_mention_id: str
    vote_value: Literal[1, 0, -1] | None = None
    binding: bool | None = None
    confidence: float


class AnchorHints(BaseModel):
    """Identity evidence for entity resolution: header name/address pairs and Jira display names."""

    sender_pairs: list[tuple[str, str]]  # distinct (sender_name, sender_address), sorted
    jira_actors: list[str]  # distinct Jira display names, sorted


class ExtractionResult(BaseModel):
    email_id: str
    mentions: list[Mention]
    relations: list[Relation]


# What the model returns: local ids ("m1", ...), rewritten by parse_response.
class RawMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_id: str
    surface: str
    type: MentionType
    context: str


class RawRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: RelationType
    source: str
    target: str
    vote_value: Literal[1, 0, -1] | None
    binding: bool | None
    confidence: float  # clamped to [0, 1] by parse_response


class RawExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mentions: list[RawMention]
    relations: list[RawRelation]


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mentions", "relations"],
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["local_id", "surface", "type", "context"],
                "properties": {
                    "local_id": {"type": "string", "description": "m1, m2, ... unique per email"},
                    "surface": {"type": "string", "description": "The text exactly as written"},
                    "type": {"type": "string", "enum": list(MENTION_TYPES)},
                    "context": {"type": "string", "description": "The sentence containing it"},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "source", "target", "vote_value", "binding", "confidence"],
                "properties": {
                    "type": {"type": "string", "enum": list(RELATION_TYPES)},
                    "source": {"type": "string", "description": "local_id of the source mention"},
                    "target": {"type": "string", "description": "local_id of the target mention"},
                    "vote_value": {"anyOf": [{"type": "integer", "enum": [1, 0, -1]}, {"type": "null"}]},
                    "binding": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                    "confidence": {"type": "number", "description": "0.0 to 1.0"},
                },
            },
        },
    },
}
