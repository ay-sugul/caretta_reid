"""Backbone abstractions for embedding generation."""

from __future__ import annotations

from typing import Protocol

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class BackboneStrategy(Protocol):
    """Defines the interface for interchangeable backbone strategies."""

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of images into embeddings."""


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor with a projection head."""

    def __init__(self, embedding_dimension: int, pretrained: bool, device: torch.device, seed: int = 42) -> None:
        super().__init__()
        self._device = device
        self._model = self._build_model(pretrained)
        in_features = self._model.classifier[1].in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._projection = nn.Linear(in_features, embedding_dimension)
        self.to(device)

    def _build_model(self, pretrained: bool) -> nn.Module:
        """Builds the torchvision EfficientNet model with download fallback."""

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            return efficientnet_b0(weights=weights)
        except (OSError, RuntimeError) as error:
            logger.warning("Falling back to randomly initialized EfficientNet-B0: {}", error)
            return efficientnet_b0(weights=None)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encodes images and returns L2-normalized embeddings."""

        features = self._model.features(image_batch)
        pooled = self._model.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        projected = self._projection(flattened)
        return F.normalize(projected, p=2, dim=1)
