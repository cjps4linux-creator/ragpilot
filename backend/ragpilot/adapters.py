"""Embedding + generation adapters. Mock mode = deterministic hashed embeddings +
templated grounded answers (no model/GPU). Real mode swaps in sentence-transformers
+ an LLM client; the retrieval/graph/grounding contracts stay identical.
"""
import hashlib
import re
from ragpilot.config import MOCK_MODE, EMBEDDING_DIM


def embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding from text (stable, zero-dep, mock-safe)."""
    if not MOCK_MODE:
        raise NotImplementedError("Set MOCK_MODE=true or plug a real embedder.")
    vec = [0.0] * EMBEDDING_DIM
    # bag-of-words hashed into the vector -> lexical similarity emerges
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
        vec[h % EMBEDDING_DIM] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


def generate(question: str, chunks: list) -> str:
    """Grounded answer synthesis. Real mode: LLM w/ chunks as context.
    Mock: concatenates supporting snippets (proves citation traceability)."""
    if not MOCK_MODE:
        raise NotImplementedError("Set MOCK_MODE=true or plug a real LLM client.")
    if not chunks:
        return "(no retrieved context — cannot answer without sources)"
    parts = []
    for ch in chunks[:3]:
        snippet = ch.text[:180].replace("\n", " ")
        parts.append(f"{ch.entity or 'source'}: {snippet}")
    return " | ".join(parts)
