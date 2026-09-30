from app.memory.memory_retriever import MemoryRetriever
from tests.conftest import add_memory


def test_retrieves_relevant_memory_first(store, embedder):
    add_memory(store, embedder, "alice", "User's dog is named Max")
    add_memory(store, embedder, "alice", "User likes hiking mountains")

    retriever = MemoryRetriever(store, embedder, top_k=3, min_similarity=0.3)
    results = retriever.retrieve("alice", "What is my dog's name?")

    assert results
    assert results[0].text == "User's dog is named Max"
    assert all("hiking" not in m.text for m in results)   # below threshold


def test_users_are_isolated(store, embedder):
    add_memory(store, embedder, "alice", "User's dog is named Max")

    retriever = MemoryRetriever(store, embedder, top_k=3, min_similarity=0.1)
    assert retriever.retrieve("bob", "What is my dog's name?") == []