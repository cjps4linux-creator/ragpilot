from pydantic import BaseModel, Field
# Removed Optional in favor of X | None
import datetime


class Document(BaseModel):
    """A source doc (meeting note, CRM record, report)."""
    id: int | None = None
    source_type: str = "meeting_note"   # meeting_note | crm_record | report
    title: str
    raw_text: str
    author: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    meta: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit with embedding + metadata for filtered vector search."""
    id: int | None = None
    doc_id: int
    text: str
    embedding: list[float] = Field(default_factory=list)
    source_type: str
    entity: str | None = None       # e.g. investor/company name for graph
    author: str | None = None
    # populated at query time
    score: float = 0.0
    cited: bool = False


class Citation(BaseModel):
    chunk_id: int
    snippet: str
    source_type: str
    entity: str | None = None


class Answer(BaseModel):
    """Grounded answer. Every claim MUST map to a citation (hallucination control)."""
    text: str
    citations: list[Citation]
    grounded: bool           # True only if no unsupported span detected
    retrieval_score: float   # mean similarity of supporting chunks
    graph_paths: list[list[str]] = Field(default_factory=list)
