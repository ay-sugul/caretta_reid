"""Pydantic message models exchanged between agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Represents an image crop in pixel coordinates."""

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectionRequest(BaseModel):
    """Carries the image path and optional dataset identifiers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    image_id: int | None = None
    file_name: str | None = None


class DetectionResult(BaseModel):
    """Contains the detected turtle head region and annotation metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    bbox: BoundingBox
    annotation_id: int | None = None
    image_id: int | None = None
    segmentation: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: str = "annotations"


class PreprocessingResult(BaseModel):
    """Stores the cropped, resized, normalized tensor for embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    tensor: Any
    bbox: BoundingBox


class EmbeddingResult(BaseModel):
    """Contains a normalized embedding vector."""

    image_path: Path
    embedding: list[float]
    dimension: int


class EmbeddingRecord(BaseModel):
    """Represents an embedding persisted in the vector store."""

    image_path: Path
    identity: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchCandidate(BaseModel):
    """Represents a single match retrieved from the vector store."""

    identity: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchResult(BaseModel):
    """Contains the ranked nearest neighbors for a query embedding."""

    image_path: Path
    candidates: list[MatchCandidate] = Field(default_factory=list)


class IdentityDecision(BaseModel):
    """Final identity decision produced by the pipeline."""

    image_path: Path
    predicted_identity: str
    is_known_individual: bool
    confidence: float
    top_candidates: list[MatchCandidate] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    """Aggregates the outputs of the complete pipeline."""

    image_path: Path
    detection: DetectionResult | None = None
    preprocessing: PreprocessingResult | None = None
    embedding: EmbeddingResult | None = None
    matches: MatchResult | None = None
    identity: IdentityDecision | None = None
    error_message: str | None = None
