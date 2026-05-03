"""Matching agent that queries the embedding vector store."""

from __future__ import annotations

from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.database.embedding_store import EmbeddingStore
from caretta_reid.schemas.messages import EmbeddingResult, MatchResult


class MatchingAgent(BaseAgent[EmbeddingResult, MatchResult]):
    """Retrieves the closest stored identities using cosine similarity."""

    def __init__(self, store: EmbeddingStore, top_k: int) -> None:
        self._store = store
        self._top_k = top_k

    def execute(self, message: EmbeddingResult) -> MatchResult:
        """Queries the vector database for the closest identity candidates."""

        try:
            candidates = self._store.query(message.embedding, self._top_k)
            return MatchResult(image_path=message.image_path, candidates=candidates)
        except (RuntimeError, ValueError) as error:
            logger.exception("Failed to match embedding for {}", message.image_path)
            raise RuntimeError(f"Matching failed for {message.image_path}: {error}") from error
