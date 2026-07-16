"""Embedding + generation adapters.

Two modes, selected by env:
- MOCK_MODE=true (default): deterministic hashed embeddings + templated grounded
  answers. Zero dependencies, runs anywhere, proves the retrieval/grounding contract.
- MOCK_MODE=false: real sentence-transformers embedder + a pluggable LLM.

LLM providers (when not mock):
- OPENAI_API_KEY set  -> OpenAI chat completions
- AWS configured      -> Amazon Bedrock Converse API (Anthropic / Nova)
If neither is configured, falls back to mock generation so the service still runs.

The retrieval / graph / grounding contracts in retrieval.py are identical regardless
of which adapter is active -- that is the whole point of the seam.
"""
import hashlib
import os
import re
from ragpilot.config import MOCK_MODE, EMBEDDING_DIM


# --------------------------------------------------------------------------- embed
def embed(text: str) -> list[float]:
    """Return a dense vector for `text`.

    Real path uses sentence-transformers when available; otherwise the
    deterministic hashed bag-of-words vector (so CI / zero-dep runs still work).
    """
    if MOCK_MODE:
        return _mock_embed(text)
    try:
        from sentence_transformers import SentenceTransformer  # lazy import
        model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        # degrade gracefully instead of crashing the ingest pipeline
        return _mock_embed(text)


def _mock_embed(text: str) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
        vec[h % EMBEDDING_DIM] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


# ------------------------------------------------------------------------ generate
def generate(question: str, chunks: list) -> str:
    """Grounded answer synthesis from retrieved chunks.

    Real mode calls an LLM with the chunks as context. Mock mode concatenates
    supporting snippets (proves citation traceability with no model/GPU).
    """
    if MOCK_MODE:
        return _mock_generate(question, chunks)

    context = "\n\n".join(
        f"[{i+1}] ({c.entity or c.source_type}): {c.text[:400]}"
        for i, c in enumerate(chunks[:5])
    )
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say you cannot answer. "
        "Cite sources by their [n] number.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = _llm_complete(prompt)
    return answer or _mock_generate(question, chunks)


def _mock_generate(question: str, chunks: list) -> str:
    if not chunks:
        return "(no retrieved context — cannot answer without sources)"
    parts = []
    for ch in chunks[:3]:
        snippet = ch.text[:180].replace("\n", " ")
        parts.append(f"{ch.entity or 'source'}: {snippet}")
    return " | ".join(parts)


def _llm_complete(prompt: str) -> str | None:
    """Try OpenAI, then Bedrock; return None if neither configured."""
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    # Amazon Bedrock (Converse API)
    if os.getenv("AWS_REGION") and os.getenv("BEDROCK_MODEL"):
        try:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
            resp = client.converse(
                modelId=os.getenv("BEDROCK_MODEL"),
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.1, "maxTokens": 512},
            )
            return resp["output"]["message"]["content"][0]["text"].strip()
        except Exception:
            pass
    return None
