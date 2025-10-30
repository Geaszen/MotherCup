"""Configuration dataclasses for the container damage project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset3713"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
DEFAULT_CLASSES = ("dent", "hole", "rusty")
DEFAULT_CLASSIFIER_WEIGHTS = WEIGHTS_ROOT / "resnet50_imagenet.pth"
DEFAULT_DETECTOR_WEIGHTS = WEIGHTS_ROOT / "maskrcnn_resnet50_fpn_v2_coco.pth"


@dataclass
class BaseConfig:
    """Common configuration shared across training stages."""

    dataset_root: Path = DATASET_ROOT
    classes: tuple[str, ...] = DEFAULT_CLASSES
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)


@dataclass
class ClassifierConfig(BaseConfig):
    """Hyper parameters for the image-level classifier."""

    # splits
    train_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "train")
    test_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "test")
    valid_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "valid")  # reserved, not used now

    classifier_weights: Optional[Path] = DEFAULT_CLASSIFIER_WEIGHTS
    val_split: float = 0.1  # fallback only when test_dir not found
    batch_size: int = 16
    max_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 3
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.train_dir = Path(self.train_dir)
        self.test_dir = Path(self.test_dir)
        self.valid_dir = Path(self.valid_dir)
        if self.classifier_weights is not None:
            self.classifier_weights = Path(self.classifier_weights)


@dataclass
class DetectorConfig(BaseConfig):
    """Hyper parameters for the detection and instance segmentation model."""

    # splits
    train_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "train")
    labels_dir: Path = field(default_factory=lambda: DATASET_ROOT / "labels" / "train")

    test_images_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "test")
    test_labels_dir: Path = field(default_factory=lambda: DATASET_ROOT / "labels" / "test")

    valid_images_dir: Path = field(default_factory=lambda: DATASET_ROOT / "images" / "valid")  # reserved
    valid_labels_dir: Path = field(default_factory=lambda: DATASET_ROOT / "labels" / "valid")  # reserved

    detector_weights: Optional[Path] = DEFAULT_DETECTOR_WEIGHTS
    val_split: float = 0.1  # fallback only when test split not found
    batch_size: int = 4
    max_epochs: int = 25
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    score_threshold: float = 0.1  # more permissive for early evaluation
    mask_threshold: float = 0.5
    grad_clip_norm: Optional[float] = 5.0
    resume_from: Optional[Path] = None
    use_point_rend: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.train_dir = Path(self.train_dir)
        self.labels_dir = Path(self.labels_dir)
        self.test_images_dir = Path(self.test_images_dir)
        self.test_labels_dir = Path(self.test_labels_dir)
        self.valid_images_dir = Path(self.valid_images_dir)
        self.valid_labels_dir = Path(self.valid_labels_dir)
        if self.detector_weights is not None:
            self.detector_weights = Path(self.detector_weights)


__all__ = [
    "BaseConfig",
    "ClassifierConfig",
    "DetectorConfig",
    "PROJECT_ROOT",
    "DATASET_ROOT",
    "WEIGHTS_ROOT",
    "DEFAULT_CLASSES",
    "DEFAULT_CLASSIFIER_WEIGHTS",
    "DEFAULT_DETECTOR_WEIGHTS",
]