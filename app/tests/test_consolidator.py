from app.memory.memory_consolidator import MemoryConsolidator
from tests.conftest import FakeLLM, add_memory


def test_adds_when_nothing_similar(store, embedder):
    c = MemoryConsolidator(store, embedder, FakeLLM())
    c.consolidate("alice", [{"text": "User is learning Rust", "importance": 6}])
    assert store.count("alice") == 1


def test_exact_duplicate_is_ignored(store, embedder):
    add_memory(store, embedder, "alice", "User is learning Rust")
    c = MemoryConsolidator(store, embedder, FakeLLM())
    results = c.consolidate("alice", [{"text": "User is learning Rust", "importance": 6}])
    assert results[0]["action"] == "IGNORE"
    assert store.count("alice") == 1


def test_contradiction_updates_existing(store, embedder):
    old = add_memory(store, embedder, "alice", "User lives in Paris")
    llm = FakeLLM(json_replies=[
        {"action": "UPDATE", "target_id": old.id, "text": "User lives in Berlin"}
    ])
    c = MemoryConsolidator(store, embedder, llm)
    c.consolidate("alice", [{"text": "User lives in Berlin now", "importance": 7}])

    memories = store.list_all("alice")
    assert len(memories) == 1
    assert memories[0].text == "User lives in Berlin"


def test_bad_llm_output_falls_back_to_add(store, embedder):
    add_memory(store, embedder, "alice", "User lives in Paris")
    c = MemoryConsolidator(store, embedder, FakeLLM(json_replies=[None]))
    c.consolidate("alice", [{"text": "User lives in Berlin now", "importance": 7}])
    assert store.count("alice") == 2