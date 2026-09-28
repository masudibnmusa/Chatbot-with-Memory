# Chatbot with Memory

Persistent conversation memory using embeddings.

A chatbot that remembers past conversations across sessions. Instead of stuffing the full chat history into every prompt (which hits context limits and gets expensive), it stores memories as embeddings in a vector database and retrieves only the relevant ones for each new message.

**Core idea:** after each exchange, extract and store what's worth remembering. When the user sends a new message, embed it, retrieve the most relevant past memories, and inject them into the prompt alongside recent chat history, so the LLM responds as if it "remembers."

---

## Features

- **Short-term memory:** a sliding window of the last N messages from the current session
- **Long-term memory:** facts stored as embeddings in a vector DB with metadata (timestamp, importance, source session)
- **Memory extraction:** an LLM call pulls out preferences, personal details, goals, and decisions (e.g. "User is learning Rust", "User's dog is named Max")
- **Relevance retrieval:** top-k similarity search on every new message
- **Memory scoring (optional):** rank by a mix of relevance, recency, and importance, similar to the "generative agents" approach
- **Memory maintenance:** deduplicate, merge conflicting facts ("moved to Berlin" replaces "lives in Paris"), and summarize old conversations
- **User control:** view, edit, and delete what the bot remembers
- **Per-user namespaces:** each user's memories are kept separate

---

## How It Works

```
User message
     ↓
embedder.py → memory_retriever.py        (top-k relevant memories)
     ↓
prompt_builder.py                        (system prompt + memories + last N messages)
     ↓
llm_client.py                            (response shown to user)
     ↓
memory_extractor.py                      (what's worth remembering from this turn?)
     ↓
memory_consolidator.py                   (dedupe / resolve conflicts)
     ↓
memory_store.py                          (embed + save to vector DB)
```

1. **Retrieve:** the incoming message is embedded and used to search the vector DB for relevant memories.
2. **Build the prompt:** retrieved memories are inserted into the system prompt under a section such as *"Things you know about this user"*, followed by the recent chat window.
3. **Respond:** the LLM generates a reply using that context.
4. **Extract:** after the turn (or at session end), a second LLM call identifies facts worth remembering.
5. **Consolidate:** new facts are compared against existing memories to skip duplicates and resolve contradictions.
6. **Store:** the surviving memories are embedded and saved with metadata.

---

## Project Structure

```
memory-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── main.py                        # Entry point (Streamlit/CLI/FastAPI)
│   ├── config.py                      # API keys, top-k, window size, thresholds
│   │
│   ├── chat/
│   │   ├── chat_engine.py             # Main loop: message -> retrieve -> respond -> store
│   │   ├── session_manager.py         # Track current session + short-term history
│   │   └── prompt_builder.py          # Combine system prompt + memories + recent history
│   │
│   ├── memory/
│   │   ├── memory_extractor.py        # LLM: pull memorable facts from conversation
│   │   ├── memory_store.py            # Vector DB wrapper (Chroma/Qdrant/Pinecone)
│   │   ├── memory_retriever.py        # Similarity search + recency/importance scoring
│   │   ├── memory_consolidator.py     # Merge duplicates, resolve contradictions
│   │   ├── memory_summarizer.py       # Compress old conversations into summaries
│   │   └── memory_model.py            # Data model: text, timestamp, importance, user_id
│   │
│   ├── embeddings/
│   │   └── embedder.py                # OpenAI / sentence-transformers wrapper
│   │
│   ├── llm/
│   │   ├── llm_client.py              # Claude/GPT API wrapper
│   │   └── prompt_templates.py        # Chat + extraction + consolidation prompts
│   │
│   ├── user_control/
│   │   └── memory_manager_ui.py       # View/edit/delete stored memories
│   │
│   └── utils/
│       └── logger.py
│
├── data/
│   ├── vector_db/                     # Local Chroma files
│   ├── chat_logs/                     # Raw conversation transcripts (JSON)
│   └── users/                         # Per-user memory namespaces
│
├── tests/
│   ├── test_memory_extractor.py
│   ├── test_memory_retriever.py
│   ├── test_consolidator.py
│   └── test_chat_engine.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── run.sh
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An API key for your chosen LLM provider (Claude or OpenAI)
- An API key for embeddings, if using a hosted model (or a local `sentence-transformers` model)

### Installation

```bash
git clone https://github.com/masudibnmusa/Chatbot-with-Memory.git
cd memory-chatbot

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Typical settings (see `app/config.py`):

| Setting | Description |
|---|---|
| `LLM_API_KEY` | API key for the chat/extraction LLM |
| `EMBEDDING_MODEL` | Embedding model name (stored with each memory) |
| `TOP_K` | Number of memories retrieved per message |
| `WINDOW_SIZE` | Number of recent messages kept in the prompt |
| `MIN_SIMILARITY` | Minimum similarity score for a memory to be injected |
| `VECTOR_DB_PATH` | Location of local vector DB files |

### Run

```bash
./run.sh
```

Or start the entry point directly:

```bash
python -m app.main
```

---

## Memory Model

Each stored memory carries:

| Field | Purpose |
|---|---|
| `text` | The memory itself, e.g. "User's dog is named Max" |
| `user_id` | Owner of the memory (used for isolation) |
| `timestamp` | When it was created or last updated |
| `importance` | Score used in ranking |
| `source_session` | Session the memory came from |
| `embedding_model` | Model used to embed it (needed if you ever re-embed) |

---

## Retrieval and Scoring

Memories are retrieved by embedding similarity. Optionally, results are re-ranked with a weighted combination:

```
score = w_relevance * relevance + w_recency * recency + w_importance * importance
```

Memories below `MIN_SIMILARITY` are dropped, so unrelated facts are never injected just because they landed in the top-k.

---

## Memory Maintenance

- **Deduplication:** near-identical facts are merged instead of stored twice.
- **Conflict resolution:** for each new fact, the closest existing memories are checked, and the consolidator decides whether to add, update, delete, or ignore.
- **Summarization:** old conversations are compressed into summaries to keep the store small and clean.

---

## User Control and Privacy

Users can view, edit, and delete anything the bot remembers through `memory_manager_ui.py`.

Design principles:

- Memories are isolated per user, enforced at the store level (metadata filter or separate collections).
- Sensitive categories (health, finances, credentials) should not be stored unless the user opts in.
- Raw transcripts in `data/chat_logs/` can be deleted independently of extracted memories.

---

## Testing

```bash
pytest tests/
```

| Test file | Covers |
|---|---|
| `test_memory_extractor.py` | Extraction quality and format |
| `test_memory_retriever.py` | Similarity search and scoring |
| `test_consolidator.py` | Deduplication and conflict handling |
| `test_chat_engine.py` | End-to-end message flow |

---

## Tech Stack

- **LLM:** Claude or GPT via API
- **Embeddings:** OpenAI or `sentence-transformers`
- **Vector DB:** Chroma (local), with Qdrant or Pinecone as alternatives
- **Interface:** Streamlit, CLI, or FastAPI

---

## License

MIT