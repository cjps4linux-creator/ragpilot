import os

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("1", "true", "yes")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.2-3b")

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "ragpilot.db"))

# Retrieval / generation knobs
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))
CITATION_REQUIRED = os.getenv("CITATION_REQUIRED", "true").lower() in ("1", "true", "yes")
