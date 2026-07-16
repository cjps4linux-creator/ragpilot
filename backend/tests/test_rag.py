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

from ragpilot import store, metrics  # noqa: E402
from ragpilot.metrics import metrics as metrics_mod  # noqa: E402
from ragpilot.models import Document  # noqa: E402
from ragpilot.ingest import ingest  # noqa: E402
from ragpilot.retrieval import query  # noqa: E402


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


def test_golden_eval():
    """Golden-set eval must report non-trivial retrieval + grounding."""
    from ragpilot import evalset
    store.reset()
    metrics_mod.reset()
    ingest(Document(title="Acme kickoff", source_type="meeting_note", author="Alice",
                    raw_text="Acme meeting: discussed lead investor Globex and follow-on from Initech. "
                             "Globex confirmed participation in the Series A."))
    ingest(Document(title="Northwind CRM", source_type="crm_record", author="Bob",
                    raw_text="Northwind partnered with Initech on the distribution deal."))
    ingest(Document(title="Q3 Report", source_type="report", author="Carol",
                    raw_text="Q3 report shows revenue up 18% quarter over quarter. "
                             "Globex remains the largest external stakeholder."))
    res = evalset.run_golden(lambda q, st, e: query(q, st, e))
    assert res["cases"] == len(evalset.GOLDEN)
    assert res["mean_grounding"] >= 0.5, f"grounding too low: {res}"
    assert res["mean_mrr"] > 0, f"retrieval MRR zero: {res}"
    print(f"PASS golden-eval: {res}")


def test_pgvector_backend_skippable():
    """Runs only when a real Postgres/pgvector DATABASE_URL is provided."""
    import pytest
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set — pgvector backend not exercised")
    os.environ["VECTOR_BACKEND"] = "pgvector"
    from ragpilot import store as pgstore
    pgstore.init()
    pgstore.reset()
    pgstore.insert_document(Document(title="PG test", source_type="report",
                                      raw_text="Postgres pgvector backend works."))
    assert pgstore.all_chunks()
    print("PASS pgvector backend: insert + list")


if __name__ == "__main__":
    setup_module(None)
    test_ingestion_chunks_entities()
    test_retrieval_metadata_filter()
    test_grounding_citations()
    test_eval_metrics()
    test_graph_resolution()
    test_golden_eval()
    test_pgvector_backend_skippable()
    print("\nALL TESTS PASSED")
