import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CHAT_LOGS_DIR = DATA_DIR / "chat_logs"
USERS_DIR = DATA_DIR / "users"

for _d in (VECTOR_DB_DIR, CHAT_LOGS_DIR, USERS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
_default_embedding = (
    "all-MiniLM-L6-v2" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _default_embedding)

# --- Memory behavior ---
TOP_K = _int("TOP_K", 5)
WINDOW_SIZE = _int("WINDOW_SIZE", 10)
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.3)
DUPLICATE_THRESHOLD = _float("DUPLICATE_THRESHOLD", 0.95)
CONSOLIDATION_CANDIDATE_THRESHOLD = _float("CONSOLIDATION_CANDIDATE_THRESHOLD", 0.5)
MIN_IMPORTANCE_TO_STORE = _int("MIN_IMPORTANCE_TO_STORE", 3)
EXTRACT_EVERY_N_TURNS = _int("EXTRACT_EVERY_N_TURNS", 1)
ASYNC_EXTRACTION = _bool("ASYNC_EXTRACTION", True)

# --- Scoring ---
WEIGHT_RELEVANCE = _float("WEIGHT_RELEVANCE", 1.0)
WEIGHT_RECENCY = _float("WEIGHT_RECENCY", 0.3)
WEIGHT_IMPORTANCE = _float("WEIGHT_IMPORTANCE", 0.3)
RECENCY_HALF_LIFE_DAYS = _float("RECENCY_HALF_LIFE_DAYS", 30)