"""Image augmentation helpers."""
from __future__ import annotations

import math
import random
from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F


def _gaussian_noise(image: Tensor, std: float = 0.03) -> Tensor:
    """Additive Gaussian noise applied to a float tensor in [0, 1]."""

    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, 1.0)


class GaussianNoise:
    """Simple random noise transform."""

    def __init__(self, p: float = 0.5, std: float = 0.03) -> None:
        self.p = p
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return _gaussian_noise(img, self.std)
        return img


def build_classification_transform(image_size: int = 256, is_train: bool = True) -> T.Compose:
    """Create torchvision transforms for the classifier."""

    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomApply([T.RandomRotation(degrees=10, interpolation=F.InterpolationMode.BILINEAR)], p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                T.ToTensor(),
                GaussianNoise(p=0.5, std=0.015),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


class DetectionAugmentor:
    """Apply paired augmentations to an image and the corresponding detection target."""

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.1,
        brightness_delta: float = 0.2,
        contrast_delta: float = 0.2,
        noise_prob: float = 0.3,
        noise_std: float = 5.0,
    ) -> None:
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(self, image: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        width, height = image.size
        boxes = target["boxes"].clone()
        masks = target["masks"].clone()

        if random.random() < self.horizontal_flip_prob:
            image = ImageOps.mirror(image)
            boxes = self._flip_boxes_horizontal(boxes, width)
            if masks.numel():
                masks = torch.flip(masks, dims=[2])

        if random.random() < self.vertical_flip_prob:
            image = ImageOps.flip(image)
            boxes = self._flip_boxes_vertical(boxes, height)
            if masks.numel():
                masks = torch.flip(masks, dims=[1])

        # Photometric distortions keep geometry intact but improve robustness.
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
            image = F.adjust_brightness(image, factor)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast_delta, self.contrast_delta)
            image = F.adjust_contrast(image, factor)

        if random.random() < self.noise_prob:
            np_img = np.array(image).astype(np.float32)
            noise = np.random.normal(0.0, self.noise_std, size=np_img.shape).astype(np.float32)
            np_img = np.clip(np_img + noise, 0.0, 255.0).astype(np.uint8)
            image = Image.fromarray(np_img)

        target = target.copy()
        target["boxes"] = boxes
        target["masks"] = masks
        return image, target

    @staticmethod
    def _flip_boxes_horizontal(boxes: Tensor, width: int) -> Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        return flipped

    @staticmethod
    def _flip_boxes_vertical(boxes: Tensor, height: int) -> Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 1] = height - boxes[:, 3]
        flipped[:, 3] = height - boxes[:, 1]
        return flipped


def build_detection_transforms(is_train: bool = True) -> Callable[[Image.Image, dict], Tuple[Image.Image, dict]]:
    """Factory for detection/segmentation augmentations."""

    if is_train:
        return DetectionAugmentor()

    def identity(image: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        return image, target

    return identity


__all__ = [
    "build_classification_transform",
    "build_detection_transforms",
    "DetectionAugmentor",
]
