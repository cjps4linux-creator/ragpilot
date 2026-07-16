"""Contract tests for ragpilot (headless, mock mode)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# remove every other product 'backend' from path to avoid 'app' collision
sys.path = [p for p in sys.path
            if "jobhunter" not in p and "leadpilot" not in p
            and ("products" not in p or "ragpilot" in p)]
# drop any cached 'app' modules from a prior collision
for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
    del sys.modules[mod]
os.environ["MOCK_MODE"] = "true"
os.environ["DB_PATH"] = os.path.join(os.path.dirname(__file__), "test_ragpilot.db")

from ragpilot import store, metrics
from ragpilot.metrics import metrics as metrics_mod
from ragpilot.models import Document
from ragpilot.ingest import ingest
from ragpilot.retrieval import query


def setup_module(_):
    store.init()
    store.reset()


def _seed():
    store.reset()
    metrics_mod.reset()
    ingest(Document(title="Q3 investor sync",
                    raw_text="Acme Capital led the Series B. Northwind Partners co-invested. Jane Doe represents Acme Capital.",
                    source_type="meeting_note", author="Carlos"))
    ingest(Document(title="CRM note",
                    raw_text="Northwind Partners focuses on supply-chain startups. Series B closed in Q3.",
                    source_type="crm_record", author="Mei"))


def test_ingestion_chunks_entities():
    _seed()
    chunks = store.all_chunks()
    assert len(chunks) > 0
    ents = store.graph_paths("Acme Capital")
    assert any("Acme Capital" in p for p in ents), "entity edge missing"
    print(f"PASS ingestion: {len(chunks)} chunks, graph edges built")


def test_retrieval_metadata_filter():
    _seed()
    # entity-scoped retrieval should surface the Acme chunk, not the CRM one
    ans = query("Who led the Series B?", entity="Acme Capital")
    assert ans.citations, "no citations returned"
    assert ans.grounded is True
    assert any("Acme" in (c.entity or "") for c in ans.citations)
    # metadata filter: restrict to crm_record
    ans2 = query("Series B timing", source_types=["crm_record"])
    assert ans2.citations, "metadata filter returned nothing"
    print(f"PASS retrieval+filter: grounded={ans.grounded} citations={len(ans.citations)}")


def test_grounding_citations():
    _seed()
    ans = query("Which firms co-invested?", entity="Northwind Partners")
    assert ans.citations
    for c in ans.citations:
        assert c.chunk_id and c.snippet
    assert ans.grounded is True
    print(f"PASS grounding: {len(ans.citations)} cited chunks, grounded={ans.grounded}")


def test_eval_metrics():
    _seed()
    query("Who led the Series B?", entity="Acme Capital")
    query("Series B timing", source_types=["crm_record"])
    snap = metrics.metrics.snapshot()
    assert snap["queries"] == 2
    assert snap["grounding_rate"] == 1.0
    assert snap["mrr"] > 0
    print(f"PASS eval: {snap}")


def test_graph_resolution():
    _seed()
    paths = store.graph_paths("Acme Capital")
    assert paths, "no graph paths"
    assert any("Acme Capital" in p for p in paths)
    print(f"PASS graph: {len(paths)} relation(s) for Acme Capital")


if __name__ == "__main__":
    setup_module(None)
    test_ingestion_chunks_entities()
    test_retrieval_metadata_filter()
    test_grounding_citations()
    test_eval_metrics()
    test_graph_resolution()
    print("\nALL TESTS PASSED")
