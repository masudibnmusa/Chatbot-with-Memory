from datetime import datetime, timezone

from app import config
from app.llm.prompt_templates import CONSOLIDATION_SYSTEM_PROMPT
from app.memory.memory_model import Memory
from app.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ACTIONS = {"ADD", "UPDATE", "DELETE", "IGNORE"}


class MemoryConsolidator:
    """Dedupes new facts and resolves contradictions before they hit the store."""

    def __init__(self, store, embedder, llm):
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def consolidate(self, user_id, candidates, source_session="", source_message=""):
        results = []
        for cand in candidates:
            text, importance = cand["text"], cand["importance"]
            embedding = self.embedder.embed_one(text)

            similar = [
                (m, s)
                for m, s in self.store.search(user_id, embedding, 5)
                if s >= config.CONSOLIDATION_CANDIDATE_THRESHOLD
            ]

            # Nothing related: just add.
            if not similar:
                self._add(user_id, text, importance, embedding, source_session, source_message)
                results.append({"action": "ADD", "text": text})
                continue

            # Near-exact duplicate: skip without an LLM call.
            if similar[0][1] >= config.DUPLICATE_THRESHOLD:
                results.append({"action": "IGNORE", "text": text})
                continue

            decision = self._decide(text, similar)
            action = decision["action"]

            if action == "IGNORE":
                results.append({"action": "IGNORE", "text": text})

            elif action == "DELETE":
                self.store.delete(user_id, decision["target_id"])
                results.append({"action": "DELETE", "target_id": decision["target_id"]})

            elif action == "UPDATE":
                target_id = decision["target_id"]
                new_text = decision["text"] or text
                old = next(m for m, _ in similar if m.id == target_id)
                self.store.update(
                    user_id,
                    target_id,
                    new_text,
                    self.embedder.embed_one(new_text),
                    importance=max(importance, old.importance),
                )
                results.append({"action": "UPDATE", "target_id": target_id, "text": new_text})

            else:  # ADD
                new_text = decision["text"] or text
                self._add(
                    user_id, new_text, importance,
                    self.embedder.embed_one(new_text), source_session, source_message,
                )
                results.append({"action": "ADD", "text": new_text})

        return results

    def _decide(self, new_text: str, similar) -> dict:
        existing = "\n".join(f"[{m.id}] {m.text}" for m, _ in similar)
        prompt = f"NEW fact:\n{new_text}\n\nEXISTING memories:\n{existing}"
        data = self.llm.chat_json(CONSOLIDATION_SYSTEM_PROMPT, prompt)

        fallback = {"action": "ADD", "target_id": None, "text": new_text}
        if not isinstance(data, dict):
            return fallback

        action = str(data.get("action", "")).upper()
        if action not in VALID_ACTIONS:
            return fallback

        valid_ids = {m.id for m, _ in similar}
        target_id = data.get("target_id")
        if action in ("UPDATE", "DELETE") and target_id not in valid_ids:
            return fallback

        return {"action": action, "target_id": target_id, "text": (data.get("text") or "").strip()}

    def _add(self, user_id, text, importance, embedding, session, message):
        memory = Memory(
            text=text,
            user_id=user_id,
            importance=importance,
            source_session=session,
            source_message=message,
            embedding_model=getattr(self.embedder, "model_name", ""),
            timestamp=datetime.now(timezone.utc),
        )
        self.store.add(memory, embedding)