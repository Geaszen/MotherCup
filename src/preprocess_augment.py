"""Offline data augmentation script for YOLO-style datasets."""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover - surface a clear message when the dependency is missing
    raise SystemExit(
        "albumentations is required for offline preprocessing. Install it via `pip install albumentations opencv-python`"
    ) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python is required for offline preprocessing. Install it via `pip install opencv-python`"
    ) from exc

from config import DATASET_ROOT


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_COPIES_PER_IMAGE = 6
DEFAULT_RANDOM_SEED = 42
AUG_SUFFIX = "_aug"


@dataclass
class AugmentationPlan:
    """Configuration for preprocessing augmentation."""

    copies_per_image: int = DEFAULT_COPIES_PER_IMAGE
    seed: int = DEFAULT_RANDOM_SEED
    min_visibility: float = 0.15


def list_train_images(images_root: Path) -> list[Path]:
    """Return all training image paths, skipping files that were already augmented."""

    return sorted(
        [
            p
            for p in images_root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and AUG_SUFFIX not in p.stem
        ]
    )


def read_yolo_label(label_path: Path) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    """Read YOLO labels, returning class ids and (cx, cy, w, h) tuples."""

    if not label_path.exists():
        return [], []

    classes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            classes.append(int(cls))
            boxes.append((float(cx), float(cy), float(w), float(h)))
    return classes, boxes


def write_yolo_label(label_path: Path, classes: Sequence[int], boxes: Sequence[Sequence[float]]) -> None:
    """Persist YOLO labels to disk."""

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for cls, (cx, cy, w, h) in zip(classes, boxes):
            handle.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_augmentations(plan: AugmentationPlan) -> A.Compose:
    """Create the albumentations pipeline used for offline preprocessing."""

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.4),
            A.ColorJitter(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT101,
                p=0.6,
            ),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=plan.min_visibility),
    )


def augment_once(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    augmenter: A.Compose,
) -> bool:
    """Apply an augmentation to a single image + label pair."""

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return False
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    classes, boxes = read_yolo_label(label_path)
    result = augmenter(image=image, bboxes=boxes, class_labels=classes)

    aug_image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    aug_boxes = result["bboxes"]
    aug_classes = result["class_labels"]

    if not aug_boxes:
        return False

    cv2.imwrite(str(output_image_path), aug_image)
    write_yolo_label(output_label_path, aug_classes, aug_boxes)
    return True


def run_augmentation(
    images_root: Path,
    labels_root: Path,
    plan: AugmentationPlan,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Augment the training split and persist the new samples."""

    augmenter = build_augmentations(plan)

    processed = 0
    created = 0

    for image_path in list_train_images(images_root):
        rel_path = image_path.relative_to(images_root)
        label_path = labels_root / rel_path.with_suffix(".txt")

        if not label_path.exists():
            # Skip images without labels; offline augmentation is for supervised samples.
            continue

        for copy_idx in range(plan.copies_per_image):
            suffix = f"{AUG_SUFFIX}{copy_idx + 1}"
            output_image_path = (images_root / rel_path).with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
            output_label_path = labels_root / rel_path.with_name(f"{image_path.stem}{suffix}.txt")

            if not overwrite and output_image_path.exists() and output_label_path.exists():
                continue

            # Use deterministic params per copy to keep reproducibility across runs.
            seed = plan.seed + processed * 997 + copy_idx * 57
            random.seed(seed)
            np.random.seed(seed)
            try:
                A.set_seed(seed)
            except AttributeError:
                pass

            success = augment_once(image_path, label_path, output_image_path, output_label_path, augmenter)
            if success:
                created += 1
        processed += 1

    return processed, created


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline augmentation for the training split.")
    parser.add_argument("--images", type=Path, default=DATASET_ROOT / "images" / "train", help="Path to training images.")
    parser.add_argument("--labels", type=Path, default=DATASET_ROOT / "labels" / "train", help="Path to training labels.")
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES_PER_IMAGE, help="Number of augmented copies per image.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate augmented files even if they already exist.",
    )
    return parser.parse_args(args)


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args if cli_args is not None else sys.argv[1:])
    plan = AugmentationPlan(copies_per_image=max(1, args.copies), seed=args.seed)

    images_root = args.images.resolve()
    labels_root = args.labels.resolve()

    if not images_root.exists() or not labels_root.exists():
        raise SystemExit("Training image/label directories must exist before preprocessing.")

    processed, created = run_augmentation(images_root, labels_root, plan, overwrite=args.overwrite)
    print(f"Processed {processed} base images; created {created} augmented samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
