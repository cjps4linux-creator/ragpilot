"""Vector store backend dispatcher.

Default backend: SQLite (embeddings as BLOB) -- zero-dep, runs anywhere.
Production backend: pgvector (PostgreSQL + vector index) -- set
VECTOR_BACKEND=pgvector and DATABASE_URL=postgresql://...

Both backends expose the SAME interface used by ingest.py / retrieval.py, so
the RAG contract (chunk, embed, retrieve, graph) is backend-agnostic.
"""
import os

BACKEND = os.getenv("VECTOR_BACKEND", "sqlite").lower()


def _backend():
    if BACKEND == "pgvector":
        try:
            from ragpilot import store_pg
            return store_pg
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(f"pgvector backend requested but unavailable: {e}")
    from ragpilot import store_sqlite
    return store_sqlite


# delegate every symbol the rest of the app imports from `store`
def __getattr__(name):
    return getattr(_backend(), name)
