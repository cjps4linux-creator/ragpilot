"""pgvector store backend (PostgreSQL + vector index).

Production vector store for ragpilot. Activated with:
    VECTOR_BACKEND=pgvector
    DATABASE_URL=postgresql://user:pass@host:5432/db

Same interface as store_sqlite so ingest/retrieval are backend-agnostic.
Requires: psycopg[binary], pgvector extension enabled on the DB.
"""
import os
import threading
import json

import psycopg
from ragpilot.config import DATABASE_URL
from ragpilot.models import Document, Chunk

_dsn = DATABASE_URL or os.getenv("DATABASE_URL", "")
_pool = None
_lock = threading.Lock()


def _conn():
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = psycopg.ConnectionPool(_dsn, open=True, max_size=4)
    return _pool.getconn()


def _put(conn):
    _pool.putconn(conn)


def init():
    with _conn() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        c.execute("""CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY, source_type TEXT, title TEXT,
            raw_text TEXT, author TEXT, created_at TEXT, meta JSONB)""")
        c.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY, doc_id INTEGER REFERENCES documents(id),
            text TEXT, embedding vector(384), source_type TEXT,
            entity TEXT, author TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY, name TEXT UNIQUE, type TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS edges (
            id SERIAL PRIMARY KEY, a TEXT, rel TEXT, b TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                  "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);")
        c.commit()


def insert_document(doc: Document) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO documents (source_type, title, raw_text, author, created_at, meta) "
            "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (doc.source_type, doc.title, doc.raw_text, doc.author, doc.created_at, json.dumps(doc.meta)))
        return cur.fetchone()[0]


def insert_chunk(ch: Chunk) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO chunks (doc_id, text, embedding, source_type, entity, author) "
            "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (ch.doc_id, ch.text, ch.embedding, ch.source_type, ch.entity, ch.author))
        return cur.fetchone()[0]


def all_chunks() -> list[Chunk]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM chunks").fetchall()
    out = []
    for r in rows:
        d = r._asdict()
        out.append(Chunk(**d))
    return out


def retrieve(query_emb: list[float], top_k: int, source_types: list[str] | None = None,
             entity: str | None = None) -> list[Chunk]:
    emb = "[" + ",".join(str(x) for x in query_emb) + "]"
    with _conn() as c:
        if source_types:
            rows = c.execute(
                "SELECT * FROM chunks WHERE source_type = ANY(%s) "
                "ORDER BY embedding <=> %s LIMIT %s",
                (source_types, emb, top_k)).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (emb, top_k)).fetchall()
    out = []
    for r in rows:
        d = r._asdict()
        out.append(Chunk(**d))
    # pg returns cosine distance; recompute similarity manually for an accurate score
    for ch in out:
        ch.score = _cosine(query_emb, ch.embedding)
    if entity:
        out = [ch for ch in out if ch.entity and ch.entity.lower() == entity.lower()]
    return out


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return round(dot / (na * nb), 3) if na and nb else 0.0


def upsert_entity(name: str, etype: str = "entity"):
    with _conn() as c:
        c.execute("INSERT INTO entities (name, type) VALUES (%s,%s) "
                  "ON CONFLICT (name) DO UPDATE SET type=EXCLUDED.type "
                  "RETURNING id", (name, etype))
        return c.fetchone()[0]


def add_edge(a: str, rel: str, b: str):
    with _conn() as c:
        c.execute("INSERT INTO edges (a, rel, b) VALUES (%s,%s,%s)", (a, rel, b))


def graph_paths(entity: str) -> list[list[str]]:
    with _conn() as c:
        rows = c.execute("SELECT a, rel, b FROM edges").fetchall()
    return [[r["a"], r["rel"], r["b"]] for r in rows
            if r["a"] == entity or r["b"] == entity]


def reset():
    with _conn() as c:
        c.execute("DELETE FROM chunks")
        c.execute("DELETE FROM documents")
        c.execute("DELETE FROM entities")
        c.execute("DELETE FROM edges")
