"""Tests for the pipeline agents."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from caretta_reid.agents.detection_agent import DetectionAgent
from caretta_reid.agents.embedding_agent import EmbeddingAgent
from caretta_reid.agents.identity_agent import IdentityAgent
from caretta_reid.agents.orchestrator_agent import OrchestratorAgent
from caretta_reid.agents.preprocessing_agent import PreprocessingAgent
from caretta_reid.config.settings import Settings
from caretta_reid.schemas.messages import (
    BoundingBox,
    DetectionRequest,
    DetectionResult,
    EmbeddingResult,
    IdentityDecision,
    MatchCandidate,
    MatchResult,
    PipelineResponse,
    PreprocessingResult,
)


class _FakeBackbone(nn.Module):
    """Returns a deterministic embedding for embedding-agent tests."""

    def __init__(self) -> None:
        super().__init__()

    def eval(self) -> "_FakeBackbone":
        return self

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return torch.full((image_batch.shape[0], 512), 0.25, dtype=torch.float32)


class _PassthroughAgent:
    """Simple agent stub for orchestration tests."""

    def __init__(self, output: object) -> None:
        self._output = output

    def execute(self, message: object) -> object:
        return self._output


def _write_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    image = Image.new("RGB", size, color="white")
    image.save(path)


def _build_settings(tmp_path: Path) -> Settings:
    annotations_path = tmp_path / "annotations.json"
    image_rel_path = "images/t001/sample.jpg"
    annotations_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image_rel_path, "width": 32, "height": 32}],
                "annotations": [
                    {"id": 10, "image_id": 1, "bbox": [1, 2, 20, 18], "area": 360.0, "segmentation": {}},
                    {"id": 11, "image_id": 1, "bbox": [3, 4, 8, 6], "area": 48.0, "segmentation": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        raw_data_dir=tmp_path,
        processed_data_dir=tmp_path / "processed",
        annotations_path=annotations_path,
        metadata_csv_path=tmp_path / "metadata.csv",
        metadata_splits_path=tmp_path / "metadata_splits.csv",
        embeddings_persist_dir=tmp_path / "chroma",
    )


def test_detection_agent_selects_smallest_annotation(tmp_path: Path) -> None:
    """Detection should choose the smallest annotation as the head proxy."""

    settings = _build_settings(tmp_path)
    image_path = tmp_path / "images" / "t001" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _write_image(image_path)

    result = DetectionAgent(settings).execute(DetectionRequest(image_path=image_path))

    assert result.annotation_id == 11
    assert result.bbox == BoundingBox(left=3, top=4, width=8, height=6)


def test_preprocessing_agent_returns_tensor(tmp_path: Path) -> None:
    """Preprocessing should return a 3x224x224 tensor."""

    settings = _build_settings(tmp_path)
    image_path = tmp_path / "images" / "t001" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _write_image(image_path)
    detection = DetectionResult(image_path=image_path, bbox=BoundingBox(left=0, top=0, width=16, height=16))

    result = PreprocessingAgent(settings).execute(detection)

    assert result.tensor.shape == (3, settings.image_size, settings.image_size)


def test_embedding_agent_uses_backbone(tmp_path: Path, monkeypatch) -> None:
    """Embedding should project the tensor into the configured dimension."""

    settings = _build_settings(tmp_path)
    monkeypatch.setattr("caretta_reid.agents.embedding_agent.EfficientNetBackbone", lambda **_: _FakeBackbone())
    image_path = tmp_path / "images" / "t001" / "sample.jpg"
    detection = PreprocessingResult(image_path=image_path, tensor=torch.zeros(3, 224, 224), bbox=BoundingBox(left=0, top=0, width=10, height=10))

    result = EmbeddingAgent(settings).execute(detection)

    assert result.dimension == 512
    assert len(result.embedding) == 512
    assert result.embedding[0] == 0.25


def test_identity_agent_applies_threshold(tmp_path: Path) -> None:
    """Identity should switch to the new-individual label below the threshold."""

    settings = _build_settings(tmp_path)
    matches = MatchResult(
        image_path=tmp_path / "images" / "t001" / "sample.jpg",
        candidates=[MatchCandidate(identity="t001", score=0.7), MatchCandidate(identity="t002", score=0.6)],
    )

    decision = IdentityAgent(settings).execute(matches)

    assert decision.is_known_individual is True
    assert decision.predicted_identity == "t001"


def test_orchestrator_agent_runs_pipeline(tmp_path: Path) -> None:
    """Orchestrator should return a complete response when every stage succeeds."""

    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    detection = DetectionResult(image_path=image_path, bbox=BoundingBox(left=0, top=0, width=10, height=10))
    preprocessing = PreprocessingResult(image_path=image_path, tensor=torch.zeros(3, 224, 224), bbox=detection.bbox)
    embedding = EmbeddingResult(image_path=image_path, embedding=[0.1] * 512, dimension=512)
    matches = MatchResult(image_path=image_path, candidates=[MatchCandidate(identity="t001", score=0.9)])
    identity = IdentityDecision(
        image_path=image_path,
        predicted_identity="t001",
        is_known_individual=True,
        confidence=0.9,
        top_candidates=matches.candidates,
    )
    orchestrator = OrchestratorAgent(
        detection_agent=_PassthroughAgent(detection),
        preprocessing_agent=_PassthroughAgent(preprocessing),
        embedding_agent=_PassthroughAgent(embedding),
        matching_agent=_PassthroughAgent(matches),
        identity_agent=_PassthroughAgent(identity),
    )

    response = orchestrator.execute(DetectionRequest(image_path=image_path))

    assert isinstance(response, PipelineResponse)
    assert response.identity is not None
    assert response.identity.predicted_identity == "t001"
