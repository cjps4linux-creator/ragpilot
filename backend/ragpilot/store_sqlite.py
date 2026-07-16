"""SQLite store: documents, chunks (with embeddings), entities (graph), edges.

Demonstrates the data-engineering layer behind a knowledge system:
- vector storage as BLOB (swap for pgvector in prod)
- cosine similarity retrieval w/ metadata filtering
- entity + edge tables for relationship mapping (knowledge graph)
"""
import os
import sqlite3
import threading
import json
import struct

from ragpilot.config import DB_PATH
from ragpilot.models import Document, Chunk

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _emb_to_blob(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _blob_to_emb(b) -> list[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


def init():
    with _lock, _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, title TEXT,
            raw_text TEXT, author TEXT, created_at TEXT, meta TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER, text TEXT,
            embedding BLOB, source_type TEXT, entity TEXT, author TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, type TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT, a TEXT, rel TEXT, b TEXT)""")


# --- documents / chunks ---
def insert_document(doc: Document) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO documents (source_type, title, raw_text, author, created_at, meta) VALUES (?,?,?,?,?,?)",
            (doc.source_type, doc.title, doc.raw_text, doc.author, doc.created_at, json.dumps(doc.meta)))
        return cur.lastrowid


def insert_chunk(ch: Chunk) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO chunks (doc_id, text, embedding, source_type, entity, author) VALUES (?,?,?,?,?,?)",
            (ch.doc_id, ch.text, _emb_to_blob(ch.embedding), ch.source_type, ch.entity, ch.author))
        return cur.lastrowid


def all_chunks() -> list[Chunk]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM chunks").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["embedding"] = _blob_to_emb(d.pop("embedding"))
        out.append(Chunk(**d))
    return out


# --- retrieval: cosine + metadata filter ---
def retrieve(query_emb: list[float], top_k: int, source_types: list[str] | None = None,
             entity: str | None = None) -> list[Chunk]:
    chunks = all_chunks()
    scored = []
    for ch in chunks:
        if source_types and ch.source_type not in source_types:
            continue
        if entity and ch.entity and ch.entity.lower() != entity.lower():
            continue
        ch.score = _cosine(query_emb, ch.embedding)
        scored.append(ch)
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return round(dot / (na * nb), 3) if na and nb else 0.0


# --- knowledge graph ---
def upsert_entity(name: str, etype: str = "entity"):
    with _lock, _conn() as c:
        c.execute("INSERT OR IGNORE INTO entities (name, type) VALUES (?,?)", (name, etype))
        return c.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()["id"]


def add_edge(a: str, rel: str, b: str):
    with _lock, _conn() as c:
        c.execute("INSERT INTO edges (a, rel, b) VALUES (?,?,?)", (a, rel, b))


def graph_paths(entity: str) -> list[list[str]]:
    with _lock, _conn() as c:
        edges = c.execute("SELECT a, rel, b FROM edges").fetchall()
    paths = []
    for e in edges:
        if e["a"] == entity or e["b"] == entity:
            paths.append([e["a"], e["rel"], e["b"]])
    return paths


def reset():
    with _lock, _conn() as c:
        c.execute("DELETE FROM chunks")
        c.execute("DELETE FROM documents")
        c.execute("DELETE FROM entities")
        c.execute("DELETE FROM edges")
