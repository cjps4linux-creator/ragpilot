from fastapi import FastAPI
from pydantic import BaseModel

from ragpilot import store, metrics
from ragpilot.config import MOCK_MODE
from ragpilot.models import Document, Answer
from ragpilot.ingest import ingest
from ragpilot.retrieval import query

app = FastAPI(title="ragpilot", version="0.1.0")


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
    return {"status": "ok", "mock_mode": MOCK_MODE}


@app.post("/ingest")
def ingest_doc(req: IngestReq):
    doc = Document(title=req.title, raw_text=req.raw_text,
                   source_type=req.source_type, author=req.author, meta=req.meta)
    return ingest(doc)


@app.post("/query", response_model=Answer)
def query_doc(req: QueryReq):
    return query(req.question, req.source_types, req.entity)


@app.get("/graph")
def graph(entity: str):
    return {"entity": entity, "paths": store.graph_paths(entity)}


@app.get("/eval")
def eval_metrics():
    return metrics.metrics.snapshot()


@app.post("/reset")
def reset():
    store.reset()
    metrics.metrics.reset()
    return {"status": "reset"}
