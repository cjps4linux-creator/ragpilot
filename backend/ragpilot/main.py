from fastapi import FastAPI
from pydantic import BaseModel

from ragpilot import store, metrics
from ragpilot.config import MOCK_MODE
from ragpilot.models import Document, Answer
from ragpilot.ingest import ingest
from ragpilot.retrieval import query
from ragpilot import evalset

app = FastAPI(title="ragpilot", version="0.2.0")


class IngestReq(BaseModel):
    title: str
    raw_text: str
    source_type: str = "meeting_note"
    author: str | None = None
    meta: dict = {}


class QueryReq(BaseModel):
    question: str
    source_types: list[str] | None = None
    entity: str | None = None


@app.on_event("startup")
def _startup():
    store.init()


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE,
            "vector_backend": store.BACKEND}


@app.post("/ingest")
def ingest_doc(req: IngestReq):
    doc = Document(title=req.title, raw_text=req.raw_text,
                   source_type=req.source_type, author=req.author, meta=req.meta)
    return ingest(doc)


@app.post("/seed-demo")
def seed_demo():
    """Ingest the demo corpus that the golden eval set expects."""
    docs = [
        Document(title="Acme kickoff", source_type="meeting_note", author="Alice",
                 raw_text="Acme meeting: discussed lead investor Globex and follow-on from Initech. "
                          "Globex confirmed participation in the Series A."),
        Document(title="Northwind CRM", source_type="crm_record", author="Bob",
                 raw_text="Northwind partnered with Initech on the distribution deal. "
                          "Initech is the counterparty for Q3."),
        Document(title="Q3 Report", source_type="report", author="Carol",
                 raw_text="Q3 report shows revenue up 18% quarter over quarter. "
                          "Globex remains the largest external stakeholder."),
    ]
    return {"seeded": [ingest(d)["doc_id"] for d in docs]}


@app.post("/query", response_model=Answer)
def query_doc(req: QueryReq):
    return query(req.question, req.source_types, req.entity)


@app.get("/graph")
def graph(entity: str):
    return {"entity": entity, "paths": store.graph_paths(entity)}


@app.get("/eval")
def eval_metrics():
    return metrics.metrics.snapshot()


@app.get("/eval/golden")
def eval_golden():
    # ensure demo corpus is present for a reproducible golden run
    seed_demo()
    return evalset.run_golden(lambda q, st, e: query(q, st, e))


@app.post("/reset")
def reset():
    store.reset()
    metrics.metrics.reset()
    return {"status": "reset"}
