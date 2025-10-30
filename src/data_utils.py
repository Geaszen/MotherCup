"""Dataset helpers for YOLO-formatted container damage data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], Image.Image]
DetectionTransform = Callable[[Image.Image, Dict[str, Tensor]], Tuple[Image.Image, Dict[str, Tensor]]]


@dataclass
class YoloSample:
    """Container for a single YOLO record."""

    boxes: Tensor
    labels: Tensor
    masks: Optional[Tensor]


def list_image_files(root: Path) -> List[Path]:
    """Return all image files (jpg/png/jpeg) under ``root``."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in root.glob("**/*") if p.suffix.lower() in exts])


def yolo_txt_path_rel(image_path: Path, images_root: Path, labels_dir: Path) -> Path:
    """Map an image path under images_root to its label txt under labels_dir, preserving relative subpath."""
    rel = image_path.relative_to(images_root)
    return labels_dir / rel.with_suffix(".txt")


def read_yolo_annotations(label_path: Path) -> Tuple[Tensor, Tensor]:
    """Read YOLO annotations and return tensors of boxes (cx,cy,w,h) and labels."""
    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    boxes: List[List[float]] = []
    labels: List[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed label line (expect 5 fields) at {label_path}:{lineno} -> {line}")
            cls_id_s, cx_s, cy_s, w_s, h_s = parts
            # 允许 "1.0" 这类写法，转为 int
            try:
                cls_val = int(float(cls_id_s))
            except Exception as e:
                raise ValueError(f"Class id must be integer-like at {label_path}:{lineno}, got '{cls_id_s}'") from e
            try:
                cx = float(cx_s); cy = float(cy_s); w = float(w_s); h = float(h_s)
            except Exception as e:
                raise ValueError(f"Box values must be numeric at {label_path}:{lineno}, got '{cx_s} {cy_s} {w_s} {h_s}'") from e
            labels.append(cls_val)
            boxes.append([cx, cy, w, h])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def cxcywh_to_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    """Convert normalized YOLO (cx, cy, w, h) to pixel-space (x1, y1, x2, y2)."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def yolo_to_mask(boxes_xyxy: Tensor, height: int, width: int) -> Tensor:
    """Create rectangular masks from bounding boxes for Mask R-CNN training."""
    masks = torch.zeros((boxes_xyxy.shape[0], height, width), dtype=torch.uint8)
    for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        x1i = max(int(torch.floor(x1).item()), 0)
        y1i = max(int(torch.floor(y1).item()), 0)
        x2i = min(int(torch.ceil(x2).item()), width)
        y2i = min(int(torch.ceil(y2).item()), height)
        if x2i > x1i and y2i > y1i:
            masks[idx, y1i:y2i, x1i:x2i] = 1
    return masks


class YOLOClassificationDataset(Dataset):
    """Image-level dataset derived from YOLO detection labels (no augmentation)."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        cache_labels: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self._image_paths = list_image_files(self.images_dir)
        if cache_labels:
            self._label_cache = {p: self._read_label(p) for p in self._image_paths}
        else:
            self._label_cache = None

    def _read_label(self, image_path: Path) -> int:
        label_path = yolo_txt_path_rel(image_path, self.images_dir, self.labels_dir)
        if not label_path.exists():
            return 0
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return 1
        return 0

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        image_path = self._image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self._label_cache[image_path] if self._label_cache is not None else self._read_label(image_path)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image_tensor, torch.tensor(label, dtype=torch.float32)


class YOLOMaskDataset(Dataset):
    """Detection + segmentation dataset that converts YOLO boxes into rectangular masks."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms: Optional[DetectionTransform] = None,
        class_offset: int = 0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._image_paths = list_image_files(self.images_dir)
        self.class_offset = class_offset

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:  # type: ignore[override]
        image_path = self._image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = yolo_txt_path_rel(image_path, self.images_dir, self.labels_dir)
        boxes_cxcywh, labels = read_yolo_annotations(label_path)
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh, width, height)
        masks = yolo_to_mask(boxes_xyxy, height, width)

        num_objs = boxes_xyxy.shape[0]
        target: Dict[str, Tensor] = {
            "boxes": boxes_xyxy,
            "labels": labels + self.class_offset,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        target["masks"] = target["masks"].float()
        return image_tensor, target


def split_dataset(dataset: Dataset, val_fraction: float, seed: int = 42) -> Tuple[Iterable[int], Iterable[int]]:
    """Return train/validation indices for a dataset."""
    import random
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_fraction))
    return indices[:split], indices[split:]


__all__ = [
    "YOLOClassificationDataset",
    "YOLOMaskDataset",
    "split_dataset",
    "read_yolo_annotations",
    "cxcywh_to_xyxy",
    "yolo_to_mask",
    "list_image_files", 
]