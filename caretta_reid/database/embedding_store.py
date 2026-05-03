"""ChromaDB-backed persistence for turtle embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection
from loguru import logger

from caretta_reid.agents.detection_agent import DetectionAgent
from caretta_reid.agents.embedding_agent import EmbeddingAgent
from caretta_reid.agents.preprocessing_agent import PreprocessingAgent
from caretta_reid.config.settings import Settings, get_settings
from caretta_reid.schemas.messages import DetectionRequest, EmbeddingRecord, MatchCandidate


class EmbeddingStore:
    """Encapsulates CRUD operations against a persistent Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.embeddings_persist_dir))
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Creates the cosine-similarity collection on first use."""

        return self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_records(self, records: Sequence[EmbeddingRecord]) -> None:
        """Upserts a batch of embeddings into ChromaDB."""

        if not records:
            return
        self._collection.upsert(
            ids=[record.image_path.as_posix() for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[{"identity": record.identity, **record.metadata} for record in records],
            documents=[record.image_path.as_posix() for record in records],
        )

    def query(self, embedding: Sequence[float], top_k: int) -> list[MatchCandidate]:
        """Returns the nearest stored identities ranked by cosine similarity."""

        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        candidates: list[MatchCandidate] = []
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for metadata, distance in zip(metadatas, distances, strict=False):
            similarity = 1.0 - float(distance)
            candidates.append(
                MatchCandidate(
                    identity=str(metadata.get("identity", "unknown")),
                    score=similarity,
                    metadata=dict(metadata),
                )
            )
        return candidates

    def count(self) -> int:
        """Returns the number of stored embeddings."""

        return int(self._collection.count())

    def clear(self) -> None:
        """Deletes every record from the collection."""

        ids = self._collection.get().get("ids", [])
        if ids:
            self._collection.delete(ids=ids)


def _main() -> int:
    """Entry point used to populate the embedding database."""

    settings = get_settings()
    try:
        store = EmbeddingStore(settings)
        records = _build_embedding_records(settings)
        store.add_records(records)
        logger.info("Stored {} embeddings at {}", len(records), settings.embeddings_persist_dir)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Failed to initialize embedding store")
        return 1


def _build_embedding_records(settings: Settings) -> list[EmbeddingRecord]:
    """Creates embeddings for the development identities using the pipeline agents."""

    dataframe = pd.read_csv(settings.metadata_splits_path)
    selected_rows = dataframe[
        (dataframe[settings.split_column] == "train")
        & (dataframe["identity"].isin(settings.dev_turtle_ids))
    ]
    detection_agent = DetectionAgent(settings)
    preprocessing_agent = PreprocessingAgent(settings)
    embedding_agent = EmbeddingAgent(settings)
    records: list[EmbeddingRecord] = []
    for _, row in selected_rows.iterrows():
        image_path = settings.raw_data_dir / str(row["file_name"])
        request = DetectionRequest(image_path=image_path, file_name=str(row["file_name"]))
        try:
            detection = detection_agent.execute(request)
            preprocessing = preprocessing_agent.execute(detection)
            embedding = embedding_agent.execute(preprocessing)
            records.append(
                EmbeddingRecord(
                    image_path=image_path,
                    identity=str(row["identity"]),
                    embedding=embedding.embedding,
                    metadata=_row_metadata(row),
                )
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            logger.exception("Skipping {} because embedding generation failed", image_path)
            logger.debug("Failure details: {}", error)
    return records


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    """Converts a dataframe row into compact record metadata."""

    metadata: dict[str, Any] = {}
    for column in ("split_closed", "split_closed_random", "split_open", "clarity", "date", "year"):
        if column in row and pd.notna(row[column]):
            metadata[column] = row[column].item() if hasattr(row[column], "item") else row[column]
    return metadata


if __name__ == "__main__":
    raise SystemExit(_main())
