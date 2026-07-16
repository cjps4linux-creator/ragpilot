"""ragpilot — RAG & knowledge-systems reference implementation.

Submodules are imported explicitly by callers, e.g.
  from ragpilot import store, metrics, models, ingest, retrieval, evalset, config
`store` is a backend dispatcher (sqlite default, pgvector optional).
"""
