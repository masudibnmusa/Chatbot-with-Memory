from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    text: str
    user_id: str
    importance: int = 5                 # 1 (trivial) .. 10 (core)
    source_session: str = ""
    source_message: str = ""            # provenance: message it came from
    kind: str = "fact"                  # "fact" or "summary"
    embedding_model: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=_now)

    def to_metadata(self) -> dict:
        return {
            "user_id": self.user_id,
            "importance": int(self.importance),
            "timestamp": self.timestamp.isoformat(),
            "source_session": self.source_session,
            "source_message": self.source_message,
            "kind": self.kind,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_metadata(cls, id: str, text: str, meta: dict) -> "Memory":
        return cls(
            id=id,
            text=text,
            user_id=meta.get("user_id", ""),
            importance=int(meta.get("importance", 5)),
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            source_session=meta.get("source_session", ""),
            source_message=meta.get("source_message", ""),
            kind=meta.get("kind", "fact"),
            embedding_model=meta.get("embedding_model", ""),
        )