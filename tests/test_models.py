"""Tests for the model layer and utility metrics."""

from __future__ import annotations

import torch
from torch import nn

from caretta_reid.config.settings import Settings
from caretta_reid.models.backbone import EfficientNetBackbone
from caretta_reid.models.siamese import TripletSiameseModel
from caretta_reid.utils.metrics import roc_auc_score_binary, top_k_accuracy


class _FakeEfficientNet(nn.Module):
    """Tiny stand-in for torchvision's EfficientNet during tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 1280, kernel_size=1), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.ModuleList([nn.Identity(), nn.Linear(1280, 1280)])


def test_efficientnet_backbone_projects_to_expected_dimension(monkeypatch) -> None:
    """The backbone wrapper should emit the requested embedding size."""

    monkeypatch.setattr("caretta_reid.models.backbone.efficientnet_b0", lambda weights=None: _FakeEfficientNet())
    backbone = EfficientNetBackbone(embedding_dimension=512, pretrained=False, device=torch.device("cpu"))
    output = backbone(torch.randn(2, 3, 16, 16))

    assert output.shape == (2, 512)


class _FakeBackbone(nn.Module):
    """Produces deterministic embeddings for the Siamese model tests."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.ones(image_batch.shape[0], 512) * self.scale, p=2, dim=1)


def test_triplet_siamese_model_computes_loss(monkeypatch) -> None:
    """The Siamese model should compute a finite triplet loss."""

    monkeypatch.setattr("caretta_reid.models.siamese.EfficientNetBackbone", lambda **_: _FakeBackbone())
    model = TripletSiameseModel(Settings(pretrained_backbone=False))
    loss = model.compute_triplet_loss(torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16))

    assert torch.isfinite(loss)


def test_top_k_accuracy_and_roc_auc() -> None:
    """Metric helpers should behave deterministically on simple inputs."""

    predictions = [["t001", "t002"], ["t003", "t004"]]
    targets = ["t001", "t004"]
    assert top_k_accuracy(predictions, targets, k=2) == 1.0
    assert roc_auc_score_binary([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0]) > 0.5
