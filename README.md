# ragpilot — RAG & Knowledge-Systems reference implementation

Built to demonstrate the exact capability set in an Applied AI Engineer (RAG &
Knowledge Systems) brief: retrieval-augmented generation over unstructured
business data (meeting notes, CRM records), metadata-filtered vector search,
knowledge-graph relationship mapping, grounded generation with citations, a
**golden eval harness** (faithfulness / grounding / MRR), and a swappable
vector backend (SQLite default, **pgvector** for production).

Two modes, one contract:
- **MOCK_MODE=true** (default): deterministic hashed embeddings + templated
  grounded answers. Zero dependencies, no model/GPU, no cloud — runs anywhere.
- **MOCK_MODE=false**: real `sentence-transformers` embeddings + an LLM
  (OpenAI or **Amazon Bedrock** Converse API). The retrieval / graph /
  grounding / eval contracts stay identical — only the adapters change.

## Run (zero-dependency, mock mode)
```bash
pip install -r backend/requirements.txt
cd backend && python -m uvicorn ragpilot.main:app --port 8000
curl -X POST http://localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"title":"Q3 investor sync","raw_text":"Acme Capital led the Series B. Northwind Partners co-invested.","source_type":"meeting_note","author":"Carlos"}'
curl -X POST http://localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Who led the Series B?","entity":"Acme Capital"}'
curl http://localhost:8000/eval/golden   # runs the golden eval set
```

## Real embeddings + pgvector (production)
```bash
export MOCK_MODE=false
export VECTOR_BACKEND=pgvector
export DATABASE_URL=postgresql://user:pass@host:5432/ragpilot
export EMBED_MODEL=all-MiniLM-L6-v2
# optional LLM providers:
export OPENAI_API_KEY=sk-...            # OR
export AWS_REGION=us-east-1             # + BEDROCK_MODEL=anthropic.claude-v2
cd backend && python -m uvicorn ragpilot.main:app --port 8000
```
pgvector requires the `vector` extension on the DB (auto-created on `init()`).

## Deploy (Railway / Render / any PaaS)
- `Procfile` and `railway.toml` are included. Set `MOCK_MODE`, `VECTOR_BACKEND`,
  `DATABASE_URL` as platform env vars. `railway up` builds the Dockerfile and
  health-checks `/health`.
- Default `sqlite` backend needs no database — deploys with zero config.

## Architecture
```
ingest ─▶ chunk ─▶ embed (mock | sentence-transformers) ─▶ store (sqlite | pgvector)
                                      │
   query ─▶ embed ─▶ retrieve (vector sim + metadata filter)
                                      │
                            graph_paths (entity relations)
                                      │
                       generate (grounded, citations)
                                      │
         eval (MRR, grounding rate, faithfulness)  +  /eval/golden (held-out set)
```

## Maps to the JD
- RAG pipelines for investor intelligence .......... `retrieval.query`
- Extract insights from unstructured ............... `ingest` (chunk + entity extraction)
- Hybrid search / re-ranking ....................... `store.retrieve` (cosine + metadata)
- Hallucination control / grounding ............... `retrieval` citation gate + `evalset`
- Agent / eval frameworks ......................... `evalset.run_golden` (MRR/faithfulness)
- Cloud deploy (Bedrock-class) ................... `adapters._llm_complete` (OpenAI/Bedrock)

## Honest limitations
- MOCK_MODE embeddings are **lexical**, not semantic — the golden MRR reflects
  that ceiling. Real `sentence-transformers` embeddings raise retrieval quality
  substantially (swap is a one-line env change).
- Faithfulness is a lexical-overlap proxy; production swaps in an NLI/entailment
  scorer. The harness is built so that swap is local to `evalset.faithfulness`.
- Embeddings + vector search + metadata filter .. `store.retrieve`
- Knowledge graph / relationship mapping .......... `store.add_edge` / `graph_paths`
- Reliable source attribution .................... `Answer.citations`
- Minimize hallucination ........................ `grounded` flag + citation requirement
- Evaluate retrieval + answer quality ........... `evaluation.EvalMetrics`
- CRM integration pattern ....................... see sibling repo `leadpilot`

## Endpoints
- POST /ingest — ingest a document (meeting note / CRM record / report)
- POST /query  — RAG query (returns grounded Answer + citations + graph paths)
- GET  /graph?entity= — knowledge-graph relations for an entity
- GET  /eval   — retrieval/answer quality metrics
- POST /reset  — clear state

Swap in real adapters (`sentence-transformers` embedder, `pgvector` store, LLM client) by implementing the slots in `ragpilot/adapters.py` — the retrieval, graph, grounding, and eval contracts stay identical.
