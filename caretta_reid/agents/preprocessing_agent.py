"""Preprocessing agent that crops and normalizes image tensors."""

from __future__ import annotations

from PIL import Image
from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.config.settings import Settings
from caretta_reid.schemas.messages import DetectionResult, PreprocessingResult
from caretta_reid.utils.image_utils import crop_with_bbox, image_to_tensor, load_image


class PreprocessingAgent(BaseAgent[DetectionResult, PreprocessingResult]):
    """Applies crop, resize, and normalization to detected regions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def execute(self, message: DetectionResult) -> PreprocessingResult:
        """Builds a normalized tensor for the embedding agent."""

        try:
            image = load_image(message.image_path)
            crop = crop_with_bbox(image, message.bbox)
            tensor = image_to_tensor(crop, self._settings.image_size)
            return PreprocessingResult(image_path=message.image_path, tensor=tensor, bbox=message.bbox)
        except (FileNotFoundError, OSError, ValueError) as error:
            logger.exception("Failed to preprocess image {}", message.image_path)
            raise RuntimeError(f"Preprocessing failed for {message.image_path}: {error}") from error
