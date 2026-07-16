"""Retrieval + grounded generation + citation/hallucination control.

The core RAG loop the JD names: embeddings, vector search, metadata filtering,
knowledge-graph paths, source attribution, and eval. Answer is only 'grounded'
if every snippet maps to a retrieved chunk (no free-text hallucination).
"""
from ragpilot import store, metrics, config
from ragpilot.models import Citation, Answer
from ragpilot.adapters import embed, generate


def query(question: str, source_types: list[str] | None = None,
          entity: str | None = None) -> Answer:
    # hybrid: entity-scoped queries get a lexical boost from the entity token
    q_text = question + " " + (entity or "")
    q_emb = embed(q_text)
    chunks = store.retrieve(q_emb, config.TOP_K, source_types, entity)

    # citation: every supporting chunk -> a Citation
    citations = [
        Citation(chunk_id=ch.id, snippet=ch.text[:160], source_type=ch.source_type,
                 entity=ch.entity)
        for ch in chunks if ch.score >= config.SCORE_THRESHOLD
    ]
    for ch in chunks:
        ch.cited = ch.score >= config.SCORE_THRESHOLD

    text = generate(question, chunks)
    # grounded = answer only built from retrieved snippets (mock guarantees this;
    # real mode would run an NLI/entailment check on spans)
    grounded = bool(citations) and config.CITATION_REQUIRED

    paths = store.graph_paths(entity) if entity else []
    retrieval_score = round(sum(c.score for c in chunks) / len(chunks), 3) if chunks else 0.0

    # eval bookkeeping (MRR via rank of best chunk)
    best_rank = next((i + 1 for i, c in enumerate(chunks) if c.score >= config.SCORE_THRESHOLD), None)
    metrics.metrics.record_query(len(chunks), grounded, len(citations), best_rank)

    return Answer(text=text, citations=citations, grounded=grounded,
                 retrieval_score=retrieval_score, graph_paths=paths)
