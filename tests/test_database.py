"""Tests for the embedding store."""

from __future__ import annotations

from pathlib import Path

from caretta_reid.config.settings import Settings
from caretta_reid.database.embedding_store import EmbeddingStore
from caretta_reid.schemas.messages import EmbeddingRecord


def test_embedding_store_add_query_and_clear(tmp_path: Path) -> None:
    """The store should persist, query, and delete embeddings."""

    settings = Settings(
        embeddings_persist_dir=tmp_path / "chroma",
        raw_data_dir=tmp_path,
        processed_data_dir=tmp_path / "processed",
        annotations_path=tmp_path / "annotations.json",
        metadata_csv_path=tmp_path / "metadata.csv",
        metadata_splits_path=tmp_path / "metadata_splits.csv",
    )
    store = EmbeddingStore(settings)
    record = EmbeddingRecord(image_path=tmp_path / "sample.jpg", identity="t001", embedding=[1.0, 0.0, 0.0], metadata={"split": "train"})

    store.add_records([record])
    matches = store.query([1.0, 0.0, 0.0], top_k=1)

    assert store.count() == 1
    assert matches[0].identity == "t001"
    assert matches[0].score > 0.99

    store.clear()
    assert store.count() == 0
