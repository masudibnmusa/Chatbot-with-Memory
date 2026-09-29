import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import config


class SessionManager:
    """Tracks the current session and the short-term sliding window."""

    def __init__(self, user_id: str, window_size: int | None = None, session_id: str | None = None):
        self.user_id = user_id
        self.window_size = window_size or config.WINDOW_SIZE
        self.session_id = session_id or (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        self.messages: list[dict] = []

    @property
    def safe_user_id(self) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", self.user_id)

    def add(self, role: str, content: str) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def recent(self) -> list[dict]:
        """Last N messages, in the format LLM APIs expect."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages[-self.window_size:]
        ]

    def last_messages(self, n: int) -> list[dict]:
        return [
            {"role": m["role"], "content": m["content"]} for m in self.messages[-n:]
        ]

    def save_log(self) -> Path | None:
        if not self.messages:
            return None
        folder = Path(config.CHAT_LOGS_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.safe_user_id}_{self.session_id}.json"
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": self.messages,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path