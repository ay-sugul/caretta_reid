"""Identity decision agent that applies the score threshold."""

from __future__ import annotations

from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.config.settings import Settings
from caretta_reid.schemas.messages import IdentityDecision, MatchResult


class IdentityAgent(BaseAgent[MatchResult, IdentityDecision]):
    """Converts ranked matches into a known or new-individual decision."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def execute(self, message: MatchResult) -> IdentityDecision:
        """Returns the best identity or a new-individual label."""

        try:
            top_candidates = message.candidates[: self._settings.top_k_matches]
            if top_candidates and top_candidates[0].score >= self._settings.similarity_threshold:
                return IdentityDecision(
                    image_path=message.image_path,
                    predicted_identity=top_candidates[0].identity,
                    is_known_individual=True,
                    confidence=top_candidates[0].score,
                    top_candidates=top_candidates,
                )
            return IdentityDecision(
                image_path=message.image_path,
                predicted_identity="new_individual",
                is_known_individual=False,
                confidence=top_candidates[0].score if top_candidates else 0.0,
                top_candidates=top_candidates,
            )
        except (IndexError, ValueError, TypeError) as error:
            logger.exception("Failed to decide identity for {}", message.image_path)
            raise RuntimeError(f"Identity decision failed for {message.image_path}: {error}") from error
