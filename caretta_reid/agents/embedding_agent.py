"""Embedding agent that projects image tensors into a 512-D vector space."""

from __future__ import annotations

import torch
from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.config.settings import Settings
from caretta_reid.models.backbone import EfficientNetBackbone
from caretta_reid.schemas.messages import EmbeddingResult, PreprocessingResult


class EmbeddingAgent(BaseAgent[PreprocessingResult, EmbeddingResult]):
    """Produces L2-normalized embeddings from preprocessed turtle crops."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backbone = EfficientNetBackbone(
            embedding_dimension=settings.embedding_dimension,
            pretrained=settings.pretrained_backbone,
            device=settings.device,
            seed=settings.embedding_seed,
        )
        self._backbone.eval()

    def execute(self, message: PreprocessingResult) -> EmbeddingResult:
        """Runs the backbone in inference mode and returns a list embedding."""

        try:
            tensor = message.tensor
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Preprocessing tensor must be a torch.Tensor.")
            batch = tensor.unsqueeze(0).to(self._settings.device)
            with torch.no_grad():
                embedding = self._backbone(batch).squeeze(0).detach().cpu().tolist()
            return EmbeddingResult(image_path=message.image_path, embedding=embedding, dimension=len(embedding))
        except (RuntimeError, TypeError, ValueError) as error:
            logger.exception("Failed to embed image {}", message.image_path)
            raise RuntimeError(f"Embedding failed for {message.image_path}: {error}") from error
