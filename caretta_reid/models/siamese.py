"""Triplet-loss training utilities for identity learning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from caretta_reid.config.settings import Settings, get_settings
from caretta_reid.models.backbone import EfficientNetBackbone
from caretta_reid.utils.image_utils import load_image


class _TripletImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset that yields anchor, positive, and negative image tensors."""

    def __init__(self, triplets: list[tuple[Path, Path, Path]], image_size: int) -> None:
        self._triplets = triplets
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, positive_path, negative_path = self._triplets[index]
        return (
            self._transform(load_image(anchor_path)),
            self._transform(load_image(positive_path)),
            self._transform(load_image(negative_path)),
        )


class TripletSiameseModel(nn.Module):
    """Learns turtle embeddings using triplet margin loss."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._backbone = EfficientNetBackbone(
            embedding_dimension=settings.embedding_dimension,
            pretrained=settings.pretrained_backbone,
            device=settings.device,
        )
        self._loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Returns normalized embeddings for an image batch."""

        return self._backbone(image_batch)

    def compute_triplet_loss(
        self,
        anchor_batch: torch.Tensor,
        positive_batch: torch.Tensor,
        negative_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Computes triplet margin loss from three batches."""

        anchor_embedding = self.forward(anchor_batch)
        positive_embedding = self.forward(positive_batch)
        negative_embedding = self.forward(negative_batch)
        return self._loss(anchor_embedding, positive_embedding, negative_embedding)

    def fit(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> list[float]:
        """Runs a simple triplet-loss training loop and returns epoch losses."""

        history: list[float] = []
        self.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for anchor_batch, positive_batch, negative_batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss = self.compute_triplet_loss(
                    anchor_batch.to(self._settings.device),
                    positive_batch.to(self._settings.device),
                    negative_batch.to(self._settings.device),
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
            history.append(epoch_loss)
        return history


def _dataframe_to_triplets(settings: Settings, dataframe: pd.DataFrame) -> list[tuple[Path, Path, Path]]:
    """Builds deterministic triplets from the development identities."""

    grouped = dataframe.groupby("identity")
    identities = [identity for identity in settings.dev_turtle_ids if identity in grouped.groups]
    triplets: list[tuple[Path, Path, Path]] = []
    for index, identity in enumerate(identities):
        paths = [settings.raw_data_dir / path for path in grouped.get_group(identity)["file_name"].tolist()]
        if len(paths) < 2:
            continue
        negative_identity = identities[(index + 1) % len(identities)] if identities else identity
        negative_paths = [settings.raw_data_dir / path for path in grouped.get_group(negative_identity)["file_name"].tolist()]
        if not negative_paths:
            continue
        for anchor_index in range(len(paths) - 1):
            triplets.append((paths[anchor_index], paths[anchor_index + 1], negative_paths[0]))
    return triplets


def _build_triplet_loader(settings: Settings) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Creates a dataloader from the prepared metadata split."""

    dataframe = pd.read_csv(settings.metadata_splits_path)
    train_rows = dataframe[(dataframe[settings.split_column] == "train") & dataframe["identity"].isin(settings.dev_turtle_ids)]
    triplets = _dataframe_to_triplets(settings, train_rows)
    dataset = _TripletImageDataset(triplets, settings.image_size)
    return DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)


def _main() -> int:
    """Entry point used by ``python -m caretta_reid.models.siamese``."""

    settings = get_settings()
    try:
        model = TripletSiameseModel(settings).to(settings.device)
        dataloader = _build_triplet_loader(settings)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        history = model.fit(dataloader, optimizer, epochs=1)
        logger.info("Training finished with history: {}", history)
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        logger.exception("Triplet training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
