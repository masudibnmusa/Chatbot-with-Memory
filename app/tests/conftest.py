import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.memory_model import Memory  # noqa: E402
from app.memory.memory_store import MemoryStore  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words => higher similarity."""

    model_name = "fake"
    DIM = 128

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class FakeLLM:
    def __init__(self, chat_replies=None, json_replies=None):
        self.chat_replies = list(chat_replies or [])
        self.json_replies = list(json_replies or [])
        self.chat_calls = []

    def chat(self, system, messages, max_tokens=1024):
        self.chat_calls.append((system, messages))
        return self.chat_replies.pop(0) if self.chat_replies else "ok"

    def chat_json(self, system, user, max_tokens=1024):
        return self.json_replies.pop(0) if self.json_replies else None


def add_memory(store, embedder, user_id, text, importance=5):
    m = Memory(text=text, user_id=user_id, importance=importance, embedding_model="fake")
    store.add(m, embedder.embed_one(text))
    return m


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path):
    return MemoryStore(path=tmp_path / "db")