"""Orchestrates the full turtle Re-ID pipeline."""

from __future__ import annotations

from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.agents.detection_agent import DetectionAgent
from caretta_reid.agents.embedding_agent import EmbeddingAgent
from caretta_reid.agents.identity_agent import IdentityAgent
from caretta_reid.agents.matching_agent import MatchingAgent
from caretta_reid.agents.preprocessing_agent import PreprocessingAgent
from caretta_reid.schemas.messages import DetectionRequest, PipelineResponse


class OrchestratorAgent(BaseAgent[DetectionRequest, PipelineResponse]):
    """Runs each agent in order and converts failures into structured output."""

    def __init__(
        self,
        detection_agent: DetectionAgent,
        preprocessing_agent: PreprocessingAgent,
        embedding_agent: EmbeddingAgent,
        matching_agent: MatchingAgent,
        identity_agent: IdentityAgent,
    ) -> None:
        self._detection_agent = detection_agent
        self._preprocessing_agent = preprocessing_agent
        self._embedding_agent = embedding_agent
        self._matching_agent = matching_agent
        self._identity_agent = identity_agent

    def execute(self, message: DetectionRequest) -> PipelineResponse:
        """Executes the detection-to-identity sequence with error handling."""

        try:
            detection = self._detection_agent.execute(message)
            preprocessing = self._preprocessing_agent.execute(detection)
            embedding = self._embedding_agent.execute(preprocessing)
            matches = self._matching_agent.execute(embedding)
            identity = self._identity_agent.execute(matches)
            return PipelineResponse(
                image_path=message.image_path,
                detection=detection,
                preprocessing=preprocessing,
                embedding=embedding,
                matches=matches,
                identity=identity,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as error:
            logger.exception("Pipeline failed for {}", message.image_path)
            return PipelineResponse(image_path=message.image_path, error_message=str(error))
