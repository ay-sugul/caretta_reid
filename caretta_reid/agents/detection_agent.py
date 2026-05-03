"""Detection agent that resolves turtle head crops from annotations."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image
from loguru import logger

from caretta_reid.agents.base_agent import BaseAgent
from caretta_reid.config.settings import Settings
from caretta_reid.schemas.messages import BoundingBox, DetectionRequest, DetectionResult


class DetectionAgent(BaseAgent[DetectionRequest, DetectionResult]):
	"""Selects the best annotation for a turtle head crop."""

	def __init__(self, settings: Settings) -> None:
		self._settings = settings
		self._images_by_file_name, self._annotations_by_image_id = self._load_annotation_index()
		self._hash_to_file_name = self._build_hash_index()

	def execute(self, message: DetectionRequest) -> DetectionResult:
		"""Finds the most plausible head annotation for the requested image."""

		try:
			file_name = self._resolve_file_name(message)
			image_id = message.image_id or self._images_by_file_name.get(file_name, {}).get("id")
			annotation = self._select_annotation(image_id)
			if annotation is None:
				return self._fallback_result(message.image_path)
			bbox = self._annotation_bbox(annotation)
			return DetectionResult(
				image_path=message.image_path,
				bbox=bbox,
				annotation_id=int(annotation.get("id")) if annotation.get("id") is not None else None,
				image_id=image_id,
				segmentation=dict(annotation.get("segmentation", {})) or None,
				confidence=1.0,
				source="annotation",
			)
		except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
			logger.exception("Failed to detect region for {}", message.image_path)
			raise RuntimeError(f"Detection failed for {message.image_path}: {error}") from error

	def _load_annotation_index(self) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
		"""Loads the COCO-style annotation file into lookup dictionaries."""

		with self._settings.annotations_path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
		images_by_file_name = {str(image["file_name"]): image for image in payload.get("images", [])}
		annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
		for annotation in payload.get("annotations", []):
			image_id = int(annotation["image_id"])
			annotations_by_image_id.setdefault(image_id, []).append(annotation)
		return images_by_file_name, annotations_by_image_id

	def _resolve_file_name(self, message: DetectionRequest) -> str:
		"""Returns the dataset-relative file name used by the annotation index."""

		if message.file_name:
			return message.file_name
		hashed_file_name = self._lookup_file_name_by_hash(message.image_path)
		if hashed_file_name is not None:
			return hashed_file_name
		try:
			return message.image_path.relative_to(self._settings.raw_data_dir).as_posix()
		except ValueError:
			return message.image_path.name

	def _build_hash_index(self) -> dict[str, str]:
		"""Builds a SHA-256 lookup table for raw dataset images."""

		index: dict[str, str] = {}
		for image_path in self._settings.raw_data_dir.rglob("*"):
			if not image_path.is_file():
				continue
			if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
				continue
			index[self._sha256(image_path)] = image_path.relative_to(self._settings.raw_data_dir).as_posix()
		return index

	def _lookup_file_name_by_hash(self, image_path: Path) -> str | None:
		"""Maps uploaded images back to their dataset-relative path when possible."""

		try:
			return self._hash_to_file_name.get(self._sha256(image_path))
		except (FileNotFoundError, OSError):
			return None

	def _sha256(self, image_path: Path) -> str:
		"""Computes a stable hash for an image file."""

		digest = hashlib.sha256()
		with image_path.open("rb") as handle:
			for chunk in iter(lambda: handle.read(1024 * 1024), b""):
				digest.update(chunk)
		return digest.hexdigest()

	def _select_annotation(self, image_id: int | None) -> dict[str, Any] | None:
		"""Chooses the smallest annotation as a proxy for the head region."""

		if image_id is None:
			return None
		candidates = self._annotations_by_image_id.get(image_id, [])
		if not candidates:
			return None
		return min(candidates, key=lambda annotation: float(annotation.get("area", float("inf"))))

	def _annotation_bbox(self, annotation: dict[str, Any]) -> BoundingBox:
		"""Converts the annotation bbox to the typed message model."""

		bbox = annotation.get("bbox")
		if not isinstance(bbox, list) or len(bbox) != 4:
			return self._full_image_bbox(annotation)
		left, top, width, height = bbox
		return BoundingBox(left=int(left), top=int(top), width=int(width), height=int(height))

	def _full_image_bbox(self, annotation: dict[str, Any]) -> BoundingBox:
		"""Falls back to the full image size when no bbox is available."""

		image_info = self._images_by_file_name.get(str(annotation.get("file_name", "")), {})
		image_path = self._settings.raw_data_dir / str(image_info.get("file_name", ""))
		if image_path.exists():
			with Image.open(image_path) as image:
				return BoundingBox(left=0, top=0, width=image.width, height=image.height)
		return BoundingBox(left=0, top=0, width=self._settings.image_size, height=self._settings.image_size)

	def _fallback_result(self, image_path: Path) -> DetectionResult:
		"""Creates a full-frame detection when annotations cannot be matched."""

		with Image.open(image_path) as image:
			bbox = BoundingBox(left=0, top=0, width=image.width, height=image.height)
		return DetectionResult(image_path=image_path, bbox=bbox, confidence=0.0, source="fallback")
