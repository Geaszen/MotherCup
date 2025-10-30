"""Train a DeepSVDD one-class classifier using only positive (damaged) samples."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from config import ClassifierConfig
from data_utils import YOLOClassificationDataset
from models import build_svdd_backbone
from transforms import build_classification_transform


@torch.no_grad()
def _gather_positive_indices(ds: YOLOClassificationDataset) -> List[int]:
    """Return indices whose image has at least one YOLO object (label=1)."""
    pos_indices: List[int] = []
    for i in range(len(ds)):
        # leverage cached labels if available
        label = ds._label_cache[ds._image_paths[i]] if getattr(ds, "_label_cache", None) is not None else ds[i][1].item()
        if int(label) == 1:
            pos_indices.append(i)
    return pos_indices


@torch.no_grad()
def init_center_c(loader: DataLoader, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Compute feature center over the positive training set."""
    model.eval()
    c_sum = None
    n = 0
    for x, _ in loader:
        x = x.to(device)
        z = model(x)  # [B, D]
        if z.ndim == 4:
            z = torch.mean(z, dim=(2, 3))  # safety for unexpected feature maps
        if c_sum is None:
            c_sum = torch.zeros(z.shape[1], device=device)
        c_sum += z.sum(dim=0)
        n += z.shape[0]
    c = c_sum / max(n, 1)
    return c


def compute_threshold(loader: DataLoader, model: nn.Module, c: torch.Tensor, device: torch.device, pct: float = 95.0) -> float:
    """Compute distance threshold T as pct-percentile of train distances."""
    model.eval()
    dists: List[float] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model(x)
            if z.ndim == 4:
                z = torch.mean(z, dim=(2, 3))
            dist = torch.linalg.norm(z - c.unsqueeze(0), dim=1)  # [B]
            dists.extend(dist.detach().cpu().tolist())
    if not dists:
        return 0.0
    return float(np.percentile(np.array(dists, dtype=np.float32), pct))


def train_svdd(
    cfg: ClassifierConfig,
    backbone: str = "resnet50",
    pretrained_weights: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float = 1e-4,
    wd: float = 1e-4,
    pct_threshold: float = 95.0,
) -> Path:
    """Train DeepSVDD on positives only and save checkpoint with center and threshold."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    transform = build_classification_transform(is_train=False)
    labels_train_dir = cfg.dataset_root / "labels" / "train"

    base_train = YOLOClassificationDataset(cfg.train_dir, labels_train_dir, transform=transform, cache_labels=True)
    pos_indices = _gather_positive_indices(base_train)
    if len(pos_indices) == 0:
        raise RuntimeError("No positive samples found under images/train + labels/train. DeepSVDD needs positives.")
    train_ds = Subset(base_train, pos_indices)

    bs = batch_size or cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    model = build_svdd_backbone(device=device, backbone=backbone, weight_path=pretrained_weights)

    # optimizer with weight decay (equivalent to ||W||_2^2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    max_epochs = epochs or cfg.max_epochs

    # initialize feature center
    c = init_center_c(train_loader, model, device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = math.inf
    best_path = Path("checkpoints/classifier_svdd")
    best_path.mkdir(parents=True, exist_ok=True)
    best_file = best_path / "best2.pt"

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z = model(x)
                if z.ndim == 4:
                    z = torch.mean(z, dim=(2, 3))
                # DeepSVDD objective: minimize mean squared distance to center
                loss = torch.mean(torch.sum((z - c.unsqueeze(0)) ** 2, dim=1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{max_epochs} | Train DeepSVDD Loss {avg_loss:.6f}")

        # keep best by lowest train objective
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Recompute threshold each time we improve (optional)
            T = compute_threshold(train_loader, model, c, device, pct=pct_threshold)
            torch.save(
                {
                    "type": "svdd",
                    "backbone": backbone,
                    "model": model.state_dict(),
                    "center": c.detach().cpu(),
                    "threshold": float(T),
                    "cfg": cfg.__dict__,
                },
                best_file,
            )
            print(f"Saved new best DeepSVDD classifier to {best_file} | T(pct={pct_threshold:.1f})={T:.6f}")

    return best_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepSVDD one-class classifier for damage presence.")
    p.add_argument("--device", default="cuda:0", help="cuda or cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    p.add_argument("--weights", type=str, default="checkpoints/classifier_svdd/best.pt", help="Optional pretrained backbone weights path")
    p.add_argument("--pct-threshold", type=float, default=95.0, help="Percentile to set T on train distances")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClassifierConfig(device=args.device)
    best = train_svdd(
        cfg=cfg,
        backbone=args.backbone,
        pretrained_weights=Path(args.weights) if args.weights else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        pct_threshold=args.pct_threshold,
    )
    print(f"DeepSVDD training finished. Best model stored at {best}")


if __name__ == "__main__":
    main()