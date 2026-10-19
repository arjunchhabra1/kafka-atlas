from pydantic import BaseModel, Field


class Evidence(BaseModel):
    doc_id: str
    text: str
    score: float


class RetrievalResult(BaseModel):
    evidence: list[Evidence]
    trace: dict = Field(default_factory=dict)
