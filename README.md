# ragpilot — RAG & Knowledge-Systems reference implementation

Built to demonstrate the exact capability set in an Applied AI Engineer (RAG &
Knowledge Systems) brief: retrieval-augmented generation over unstructured
business data (meeting notes, CRM records), metadata-filtered vector search,
knowledge-graph relationship mapping, grounded generation with citations, and an
evaluation harness for retrieval/answer quality + hallucination control.

Runs headless in MOCK_MODE (deterministic hashed embeddings, no model/GPU).
Swap real adapters (sentence-transformers embedder, pgvector store, LLM client)
into the marked slots — the retrieval, graph, grounding, and eval contracts
stay identical.

## Run
```bash
docker build -f backend/Dockerfile -t ragpilot:latest .
docker run --rm -p 8000:8000 ragpilot:latest
curl -X POST http://localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"title":"Q3 investor sync","raw_text":"Acme Capital led the Series B. Northwind Partners co-invested. Jane Doe represents Acme Capital.","source_type":"meeting_note","author":"Carlos"}'
curl -X POST http://localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Who led the Series B?","entity":"Acme Capital"}'
curl http://localhost:8000/eval
```

## Architecture
```
ingest ─▶ chunk ─▶ embed ─▶ store (chunks+embeddings)
                                      │
   query ─▶ embed ─▶ retrieve (vector sim + metadata filter)
                                      │
                            graph_paths (entity relations)
                                      │
                       generate (grounded, citations)
                                      │
                       eval (MRR, grounding rate, citations)
```

## Maps to the JD
- RAG pipelines for investor intelligence .......... `retrieval.query`
- Extract insights from unstructured ............... `ingest` (chunk + entity extraction)
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
