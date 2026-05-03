"""Image helpers used across detection and preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from caretta_reid.schemas.messages import BoundingBox


def load_image(image_path: Path) -> Image.Image:
    """Loads an RGB image from disk."""

    with Image.open(image_path) as image:
        return image.convert("RGB")


def crop_with_bbox(image: Image.Image, bbox: BoundingBox, padding: int = 0) -> Image.Image:
    """Crops the image around the provided bounding box."""

    left = max(bbox.left - padding, 0)
    top = max(bbox.top - padding, 0)
    right = min(bbox.left + bbox.width + padding, image.width)
    bottom = min(bbox.top + bbox.height + padding, image.height)
    return image.crop((left, top, right, bottom))


def build_preprocessing_transform(image_size: int) -> transforms.Compose:
    """Builds the resizing and normalization pipeline."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Converts a PIL image to a normalized tensor."""

    return build_preprocessing_transform(image_size)(image)
