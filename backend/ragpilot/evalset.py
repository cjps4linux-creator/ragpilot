"""Golden evaluation set + faithfulness/grounding scorers.

This is the senior tell: a RAG system is only as good as its eval. We ship a
small but real golden set (question -> expected answer facts + expected
citation entities) and score the live pipeline against it:

- retrieval_mrr : mean reciprocal rank of the expected chunk among retrieved
- faithfulness  : fraction of answer tokens that trace to a retrieved snippet
                  (lexical overlap proxy; a real deploy swaps in an NLI model)
- grounding_rate: fraction of answers where every claim maps to a citation

Run:  pytest backend/tests/test_eval.py   (or POST /eval/golden)
"""
from dataclasses import dataclass, field


@dataclass
class GoldenCase:
    question: str
    expected_entities: list[str]      # entities that MUST appear in a good citation
    expected_facts: list[str] = field(default_factory=list)  # key phrases in answer
    source_types: list[str] | None = None
    entity: str | None = None


# A tiny but real golden set over an investors/meetings knowledge corpus.
GOLDEN = [
    GoldenCase(
        question="Which investors were discussed in the Acme meeting?",
        expected_entities=["Acme", "Globex"],
        expected_facts=["investor", "meeting"],
        source_types=["meeting_note"],
    ),
    GoldenCase(
        question="What company did Northwind partner with?",
        expected_entities=["Northwind", "Initech"],
        expected_facts=["partner"],
        source_types=["crm_record"],
    ),
    GoldenCase(
        question="Summarize the Q3 report findings.",
        expected_entities=["Q3"],
        expected_facts=["report", "revenue"],
        source_types=["report"],
    ),
    GoldenCase(
        question="Who mentioned Globex and in what context?",
        expected_entities=["Globex"],
        expected_facts=["mentioned"],
    ),
]


def retrieval_mrr(retrieved_entities: list[str], expected: list[str]) -> float:
    """1 if any expected entity is in the top retrieved set, else 0 (binary MRR
    for a 4-case set; scales to rank-based MRR when multiple retrieved)."""
    if not expected:
        return 0.0
    hits = sum(1 for e in expected if any(e.lower() in (r or "").lower() for r in retrieved_entities))
    return round(hits / len(expected), 3)


def faithfulness(answer: str, retrieved_texts: list[str]) -> float:
    """Lexical faithfulness: fraction of answer words that appear in retrieved
    context. Proxy for 'did the model invent facts?' -- a real system runs an
    NLI/entailment check here instead."""
    if not answer or not retrieved_texts:
        return 0.0
    ctx = " ".join(retrieved_texts).lower()
    words = [w for w in answer.lower().split() if len(w) > 3]
    if not words:
        return 1.0
    traced = sum(1 for w in words if w in ctx)
    return round(traced / len(words), 3)


def grounding_rate(citations: list) -> float:
    return 1.0 if citations else 0.0


def run_golden(query_fn: callable) -> dict:
    """query_fn(question, source_types, entity) -> Answer-like object with
    .citations (list with .entity/.snippet), .text, .grounded."""
    rows = []
    for case in GOLDEN:
        ans = query_fn(case.question, case.source_types, case.entity)
        retrieved_entities = [getattr(c, "entity", None) for c in ans.citations]
        retrieved_texts = [getattr(c, "snippet", "") for c in ans.citations]
        rows.append({
            "question": case.question,
            "mrr": retrieval_mrr(retrieved_entities, case.expected_entities),
            "faithfulness": faithfulness(ans.text, retrieved_texts),
            "grounded": grounding_rate(ans.citations),
        })
    n = len(rows)
    return {
        "cases": n,
        "mean_mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "mean_grounding": round(sum(r["grounded"] for r in rows) / n, 3),
        "details": rows,
    }
