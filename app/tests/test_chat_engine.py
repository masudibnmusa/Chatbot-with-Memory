from app import config
from app.chat.chat_engine import ChatEngine
from tests.conftest import FakeLLM


def test_remembers_across_sessions(store, embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASYNC_EXTRACTION", False)
    monkeypatch.setattr(config, "CHAT_LOGS_DIR", tmp_path / "logs")

    # Session 1: user shares a fact, which gets extracted and stored.
    llm1 = FakeLLM(
        chat_replies=["Nice to meet Max!"],
        json_replies=[[{"text": "User's dog is named Max", "importance": 6}]],
    )
    engine1 = ChatEngine("alice", store=store, embedder=embedder, llm=llm1)
    assert engine1.respond("My dog is named Max")
    assert store.count("alice") == 1

    # Session 2: a new engine retrieves the memory and injects it into the prompt.
    llm2 = FakeLLM(chat_replies=["Your dog is Max."])
    engine2 = ChatEngine("alice", store=store, embedder=embedder, llm=llm2)
    engine2.respond("What is my dog's name?")

    system_prompt = llm2.chat_calls[-1][0]
    assert "User's dog is named Max" in system_prompt


def test_end_session_saves_log(store, embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASYNC_EXTRACTION", False)
    monkeypatch.setattr(config, "CHAT_LOGS_DIR", tmp_path / "logs")

    engine = ChatEngine("alice", store=store, embedder=embedder, llm=FakeLLM())
    engine.respond("hello")
    engine.end_session()

    assert list((tmp_path / "logs").glob("alice_*.json"))