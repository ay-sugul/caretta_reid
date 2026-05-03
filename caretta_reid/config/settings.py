"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _workspace_root() -> Path:
    """Returns the repository root computed from this module path."""

    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Centralizes every runtime hyperparameter and filesystem path."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CARETTA_",
        extra="ignore",
    )

    app_name: str = "Caretta Re-ID"
    raw_data_dir: Path = Field(default_factory=lambda: _workspace_root() / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: _workspace_root() / "data" / "processed")
    annotations_path: Path = Field(default_factory=lambda: _workspace_root() / "data" / "raw" / "annotations.json")
    metadata_csv_path: Path = Field(default_factory=lambda: _workspace_root() / "data" / "raw" / "metadata.csv")
    metadata_splits_path: Path = Field(default_factory=lambda: _workspace_root() / "data" / "raw" / "metadata_splits.csv")
    embeddings_persist_dir: Path = Field(default_factory=lambda: _workspace_root() / "data" / "processed" / "chroma")
    demo_output_dir: Path = Field(default_factory=lambda: _workspace_root() / "data" / "processed" / "demo")
    split_column: str = "split_closed"
    chroma_collection_name: str = "caretta_embeddings"
    embedding_dimension: int = 512
    similarity_threshold: float = 0.65
    top_k_matches: int = 5
    batch_size: int = 8
    image_size: int = 224
    num_workers: int = 0
    pretrained_backbone: bool = True
    embedding_seed: int = 42
    use_gpu_if_available: bool = True
    dev_turtle_ids: list[str] = Field(default_factory=lambda: [f"t{i:03d}" for i in range(1, 11)])
    debug_mode: bool = False

    @field_validator("dev_turtle_ids", mode="before")
    @classmethod
    def parse_dev_turtle_ids(cls, value: object) -> list[str]:
        """Parses a comma-separated environment value into a list."""

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [f"t{i:03d}" for i in range(1, 11)]

    @property
    def device(self) -> torch.device:
        """Returns the preferred torch device with GPU auto-selection."""

        if self.use_gpu_if_available and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Builds a cached settings instance for dependency injection."""

    return Settings()
