from datetime import datetime, timezone

from app import config
from app.memory.memory_model import Memory


class MemoryRetriever:
    """Similarity search + recency/importance scoring."""

    def __init__(self, store, embedder, top_k=None, min_similarity=None):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k or config.TOP_K
        self.min_similarity = (
            config.MIN_SIMILARITY if min_similarity is None else min_similarity
        )

    def retrieve(self, user_id: str, query: str, k: int | None = None) -> list[Memory]:
        k = k or self.top_k
        embedding = self.embedder.embed_one(query)
        # Over-fetch, then re-rank with recency and importance.
        candidates = self.store.search(user_id, embedding, k * 3)

        scored = []
        for memory, similarity in candidates:
            if similarity < self.min_similarity:
                continue
            scored.append((self._score(memory, similarity), memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    def _score(self, memory: Memory, similarity: float) -> float:
        age_days = max(
            (datetime.now(timezone.utc) - memory.timestamp).total_seconds() / 86400, 0.0
        )
        recency = 0.5 ** (age_days / config.RECENCY_HALF_LIFE_DAYS)
        importance = memory.importance / 10
        return (
            config.WEIGHT_RELEVANCE * similarity
            + config.WEIGHT_RECENCY * recency
            + config.WEIGHT_IMPORTANCE * importance
        )