from app.llm.prompt_templates import (
    MERGE_SUMMARIES_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    format_transcript,
)
from app.memory.memory_model import Memory


class MemorySummarizer:
    """Compresses conversations into summaries to keep the store small."""

    MIN_MESSAGES = 4

    def __init__(self, store, embedder, llm):
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def summarize_session(self, user_id, session_id, messages):
        if len(messages) < self.MIN_MESSAGES:
            return None
        summary = self.llm.chat(
            SUMMARY_SYSTEM_PROMPT,
            [{"role": "user", "content": format_transcript(messages)}],
            max_tokens=300,
        ).strip()
        if not summary:
            return None
        return self._store_summary(user_id, summary, session_id, importance=4)

    def compress_summaries(self, user_id, keep_recent: int = 5):
        """Merge older session summaries into one longer-term summary."""
        summaries = [m for m in self.store.list_all(user_id) if m.kind == "summary"]
        old = summaries[keep_recent:]      # list_all is newest-first
        if len(old) < 3:
            return None

        joined = "\n".join(f"- {m.text}" for m in old)
        merged = self.llm.chat(
            MERGE_SUMMARIES_SYSTEM_PROMPT,
            [{"role": "user", "content": joined}],
            max_tokens=400,
        ).strip()
        if not merged:
            return None

        new_memory = self._store_summary(user_id, merged, "merged", importance=5)
        for m in old:
            self.store.delete(user_id, m.id)
        return new_memory

    def _store_summary(self, user_id, text, session_id, importance):
        memory = Memory(
            text=text,
            user_id=user_id,
            importance=importance,
            source_session=session_id,
            kind="summary",
            embedding_model=getattr(self.embedder, "model_name", ""),
        )
        self.store.add(memory, self.embedder.embed_one(text))
        return memory