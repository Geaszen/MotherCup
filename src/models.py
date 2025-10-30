"""Model factory helpers."""
from __future__ import annotations

import torch
from pathlib import Path

from torch import nn
from torchvision import models


def _load_state_dict_from_path(model: nn.Module, weight_path: Path | str | None, device: torch.device) -> None:
    if weight_path is None:
        return
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weight file not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)


def build_classifier(device: torch.device, weight_path: Path | None = None) -> nn.Module:
    model = models.resnet50(weights=None)
    _load_state_dict_from_path(model, weight_path, device)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model.to(device)


def build_svdd_backbone(
    device: torch.device,
    backbone: str = "resnet50",
    weight_path: Path | None = None,
) -> nn.Module:
    """Return a backbone whose final classification head is removed for DeepSVDD."""
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet50'.")
    _load_state_dict_from_path(model, weight_path, device)
    model.fc = nn.Identity()  # output embedding
    return model.to(device)


def build_mask_rcnn(
    num_classes: int,
    use_point_rend: bool = False,
    weight_path: Path | None = None,
) -> nn.Module:
    model = models.detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    _load_state_dict_from_path(model, weight_path, torch.device("cpu"))
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    if use_point_rend:
        try:
            from torchvision.models.detection import PointRend
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PointRend requires torchvision >= 0.15") from exc
        model = PointRend(model, num_classes=num_classes)
    return model


__all__ = ["build_classifier", "build_mask_rcnn", "build_svdd_backbone"]