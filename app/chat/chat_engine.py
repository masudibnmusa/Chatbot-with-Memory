from concurrent.futures import ThreadPoolExecutor

from app import config
from app.chat.prompt_builder import PromptBuilder
from app.chat.session_manager import SessionManager
from app.embeddings.embedder import Embedder
from app.llm.llm_client import LLMClient
from app.memory.memory_consolidator import MemoryConsolidator
from app.memory.memory_extractor import MemoryExtractor
from app.memory.memory_retriever import MemoryRetriever
from app.memory.memory_store import MemoryStore
from app.memory.memory_summarizer import MemorySummarizer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChatEngine:
    """message -> retrieve -> respond -> extract -> consolidate -> store"""

    def __init__(self, user_id, store=None, embedder=None, llm=None):
        self.user_id = user_id
        self.embedder = embedder if embedder is not None else Embedder()
        self.store = store if store is not None else MemoryStore()
        self.llm = llm if llm is not None else LLMClient()

        self.retriever = MemoryRetriever(self.store, self.embedder)
        self.extractor = MemoryExtractor(self.llm)
        self.consolidator = MemoryConsolidator(self.store, self.embedder, self.llm)
        self.summarizer = MemorySummarizer(self.store, self.embedder, self.llm)
        self.prompt_builder = PromptBuilder()
        self.session = SessionManager(user_id)

        self._turns = 0
        # A single worker keeps memory writes serialized.
        self._executor = ThreadPoolExecutor(max_workers=1) if config.ASYNC_EXTRACTION else None

    def respond(self, user_message: str) -> str:
        memories = self.retriever.retrieve(self.user_id, user_message)

        self.session.add("user", user_message)
        system, messages = self.prompt_builder.build(memories, self.session.recent())
        reply = self.llm.chat(system, messages)
        self.session.add("assistant", reply)

        self._turns += 1
        if self._turns % config.EXTRACT_EVERY_N_TURNS == 0:
            self._schedule_extraction()
        return reply

    def _schedule_extraction(self) -> None:
        batch = self.session.last_messages(config.EXTRACT_EVERY_N_TURNS * 2)
        if self._executor:
            self._executor.submit(self._extract_and_store, batch)
        else:
            self._extract_and_store(batch)

    def _extract_and_store(self, batch: list[dict]) -> None:
        try:
            candidates = self.extractor.extract(batch)
            if not candidates:
                return
            first_user = next((m["content"] for m in batch if m["role"] == "user"), "")
            self.consolidator.consolidate(
                self.user_id,
                candidates,
                source_session=self.session.session_id,
                source_message=first_user[:200],
            )
        except Exception:
            logger.exception("Memory extraction failed")

    def flush(self) -> None:
        """Wait for pending background extraction to finish."""
        if self._executor:
            self._executor.submit(lambda: None).result()

    def end_session(self) -> None:
        self.flush()
        try:
            self.summarizer.summarize_session(
                self.user_id, self.session.session_id, self.session.messages
            )
            self.summarizer.compress_summaries(self.user_id)
        except Exception:
            logger.exception("Session summarization failed")
        self.session.save_log()
        if self._executor:
            self._executor.shutdown(wait=True)