"""Ingestion: chunk raw docs, embed, extract entities, build graph edges.

Demonstrates 'extract structured insights from unstructured data' (meeting notes,
CRM records) + 'knowledge graph / relationship mapping' from the JD.
"""
import re
from ragpilot import store
from ragpilot.models import Document, Chunk
from ragpilot.adapters import embed


# entity extraction: capitalized multi-word phrases (investor/company style)
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")


def _chunk_text(text: str, size: int = 240) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [text]


def _extract_entities(text: str) -> list[str]:
    seen, out = set(), []
    for m in _ENTITY_RE.findall(text):
        if m.lower() not in ("the", "this", "that", "we", "i") and m not in seen:
            seen.add(m)
            out.append(m)
    return out[:8]


def ingest(doc: Document) -> dict:
    did = store.insert_document(doc)
    entities = _extract_entities(doc.raw_text + " " + doc.title)
    # graph: doc author/entity relations
    for ent in entities:
        store.upsert_entity(ent, "entity")
        if doc.author:
            store.add_edge(doc.author, "mentioned_in", ent)
    chunks = _chunk_text(doc.raw_text)
    stored = 0
    for ch in chunks:
        ents = _extract_entities(ch)
        primary = ents[0] if ents else None
        for e in ents:
            store.upsert_entity(e, "entity")
        chunk = Chunk(
            doc_id=did, text=ch, embedding=embed(ch),
            source_type=doc.source_type,
            entity=primary, author=doc.author)
        store.insert_chunk(chunk)
        stored += 1
    # document-level graph: author mentions all extracted entities
    for ent in entities:
        store.upsert_entity(ent, "entity")
        if doc.author:
            store.add_edge(doc.author, "mentioned_in", ent)
    return {"doc_id": did, "chunks": stored, "entities": len(entities)}
