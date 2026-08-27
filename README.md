# ragpilot

RAG and knowledge-systems reference implementation — retrieval-augmented generation over unstructured business data with metadata-filtered vector search, knowledge-graph relationship mapping, grounded generation with citations, and a golden eval harness for retrieval and answer quality.

Built by Conrad CJ Wilson.

## What It Demonstrates

| Capability | Implementation |
|---|---|
| Document ingestion | Chunking with entity extraction and source-type metadata |
| Embedding pipeline | Swappable embedder (mock deterministic for zero-dependency mode, `sentence-transformers` for production) |
| Vector search | Cosine similarity with metadata filtering (author, source type, date range) |
| Knowledge graph | Entity relationship mapping with graph path retrieval |
| Grounded generation | LLM answers with mandatory citation gates and faithfulness checks |
| Evaluation harness | Golden eval measuring MRR, grounding rate, and faithfulness |
| Swappable backends | SQLite (local/development) and pgvector (production) with identical contracts |
| Dual LLM providers | OpenAI and Amazon Bedrock Converse API adapters |
| Zero-dependency mode | Deterministic hashed embeddings and templated answers — no model, GPU, or cloud required |

## Stack

| Layer | Tooling |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + uvicorn |
| Vector store | SQLite (local) / PostgreSQL + pgvector (production) |
| Embeddings | Mock (deterministic hash) / `sentence-transformers` (all-MiniLM-L6-v2) |
| LLM | OpenAI Chat Completions / Amazon Bedrock Converse |
| Evaluation | Custom golden eval harness (MRR, grounding, faithfulness) |
| Deployment | Procfile, railway.toml, Dockerfile, Docker Compose |
| Testing | pytest with contract fixtures |

## Architecture

```
                    ingest ──▶ chunk ──▶ embed ──▶ store
                                                   │
                    query ──▶ embed ──▶ retrieve ◄──┘
                                                   │
                                         graph_paths
                                                   │
                                         grounded generate
                                                   │
                                   eval ◄── answer + citations
```

### Data flow

1. **Ingest**: Raw document text is chunked, entities are extracted, and embeddings are generated.
2. **Store**: Chunks and embeddings are persisted to SQLite or pgvector with metadata filters.
3. **Retrieve**: Queries are embedded and matched against the vector store with optional metadata constraints.
4. **Graph**: Related entities are traversed to enrich retrieval context.
5. **Generate**: The LLM produces a grounded answer with required citations from retrieved sources.
6. **Eval**: The golden eval harness measures retrieval recall (MRR) and answer quality (faithfulness, grounding).

## Quick Start

### Zero-dependency mode (default)

```bash
git clone https://github.com/cjps4linux-creator/ragpilot.git
cd ragpilot/backend
pip install -r requirements.txt
python -m uvicorn ragpilot.main:app --port 8000
```

No API keys, models, or databases required. `MOCK_MODE=true` uses deterministic hashed embeddings.

### Production mode

```bash
export MOCK_MODE=false
export VECTOR_BACKEND=pgvector
export DATABASE_URL=postgresql://user:password@host:5432/ragpilot
export EMBED_MODEL=all-MiniLM-L6-v2
export OPENAI_API_KEY=sk-...
# OR for Bedrock:
export AWS_REGION=us-east-1
export BEDROCK_MODEL=anthropic.claude-v2

cd backend && python -m uvicorn ragpilot.main:app --port 8000
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Ingest a document (meeting note, CRM record, report) |
| POST | `/query` | RAG query (returns grounded answer + citations + graph paths) |
| GET | `/graph?entity=` | Knowledge-graph relations for an entity |
| GET | `/eval` | Retrieval and answer quality metrics |
| GET | `/eval/golden` | Run the full golden eval suite |
| POST | `/reset` | Clear all ingested state |

## Evaluation

The golden eval harness runs a held-out question/answer set and reports:

- **MRR (Mean Reciprocal Rank)**: How high the correct source appears in retrieval results
- **Grounding rate**: Percentage of answers backed by retrieved citations
- **Faithfulness**: Whether the generated answer is supported by the cited sources (lexical-overlap proxy in this version)

Run evaluation:

```bash
curl http://localhost:8000/eval/golden
```

## Mapping to the JD

| JD Requirement | ragpilot Implementation |
|---|---|
| RAG pipelines for knowledge retrieval | `retrieval.query` with vector + metadata search |
| Extract insights from unstructured data | `ingest` with chunking and entity extraction |
| Hybrid search / re-ranking | `store.retrieve` with cosine similarity and metadata filter |
| Hallucination control / grounded generation | Citation gate + `evalset` faithfulness check |
| Agent / eval frameworks | `evalset.run_golden` with MRR and grounding metrics |
| Cloud deployment readiness | Docker, Procfile, railway.toml, adapter factory for OpenAI/Bedrock |
| Knowledge graph / relationship mapping | `store.add_edge` and `graph_paths` |
| Reliable source attribution | `Answer.citations` on every response |

## Honest Limitations

- Mock-mode embeddings are **lexical hashes**, not semantic vectors. The golden MRR reflects that ceiling. Real `sentence-transformers` embeddings substantially improve retrieval quality and are a one-line environment change (`MOCK_MODE=false`).
- Faithfulness scoring uses lexical overlap as a proxy. Production systems should integrate an NLI/entailment scorer for higher accuracy. The harness is structured so that swap is local to `evalset.faithfulness`.
- Knowledge-graph relationships are simple entity co-occurrence in this version. Production deployments should add typed relationships and graph traversal depth limits.
- No authentication or authorization is implemented. The platform is designed to integrate with an API gateway or reverse proxy for production access control.

## Deployment

### Railway / Render

Procfile and railway.toml are included. Set `MOCK_MODE`, `VECTOR_BACKEND`, `DATABASE_URL`, and LLM provider keys as platform environment variables. The health check endpoint `/health` is used for deployment verification.

### Docker

```bash
docker build -f backend/Dockerfile -t ragpilot:latest .
docker run --rm -p 8000:8000 ragpilot:latest
```

## Current State

Functional reference implementation with a complete RAG pipeline, dual-mode embeddings, knowledge graph, eval harness, and production deployment configuration. The platform runs in zero-dependency mock mode for evaluation and switches to real embeddings and LLM providers with environment configuration.

## License

MIT — use, modify, and ship freely.

**Author:** Conrad CJ Wilson
**GitHub:** https://github.com/cjps4linux-creator
**LinkedIn:** https://www.linkedin.com/in/conradcjwilson
