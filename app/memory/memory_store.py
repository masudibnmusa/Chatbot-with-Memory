import hashlib
from datetime import datetime, timezone

import chromadb

from app import config
from app.memory.memory_model import Memory


class MemoryStore:
    """Vector DB wrapper. Each user gets a separate Chroma collection,
    so isolation is enforced here rather than in calling code."""

    def __init__(self, path=None):
        self._client = chromadb.PersistentClient(path=str(path or config.VECTOR_DB_DIR))

    @staticmethod
    def _name(user_id: str) -> str:
        return "user_" + hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:24]

    def _collection(self, user_id: str):
        return self._client.get_or_create_collection(
            name=self._name(user_id), metadata={"hnsw:space": "cosine"}
        )

    def add(self, memory: Memory, embedding: list[float]) -> None:
        self._collection(memory.user_id).add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.text],
            metadatas=[memory.to_metadata()],
        )

    def update(self, user_id, memory_id, text, embedding, importance=None) -> bool:
        col = self._collection(user_id)
        existing = col.get(ids=[memory_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = dict(existing["metadatas"][0])
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()
        if importance is not None:
            meta["importance"] = int(importance)
        col.update(
            ids=[memory_id], embeddings=[embedding], documents=[text], metadatas=[meta]
        )
        return True

    def delete(self, user_id: str, memory_id: str) -> None:
        self._collection(user_id).delete(ids=[memory_id])

    def delete_all(self, user_id: str) -> None:
        try:
            self._client.delete_collection(name=self._name(user_id))
        except Exception:
            pass

    def count(self, user_id: str) -> int:
        return self._collection(user_id).count()

    def list_all(self, user_id: str) -> list[Memory]:
        res = self._collection(user_id).get(include=["documents", "metadatas"])
        memories = [
            Memory.from_metadata(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories

    def search(self, user_id: str, embedding: list[float], k: int) -> list[tuple[Memory, float]]:
        col = self._collection(user_id)
        n = col.count()
        if n == 0:
            return []
        res = col.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append((Memory.from_metadata(id_, doc, meta), 1.0 - dist))  # cosine sim
        return out