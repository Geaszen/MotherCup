"""Training script for the image-level damage classifier."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from config import ClassifierConfig
from data_utils import YOLOClassificationDataset, split_dataset
from models import build_classifier
from transforms import build_classification_transform
device = torch.device("cuda:1")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(cfg: ClassifierConfig) -> Tuple[DataLoader, DataLoader, float]:
    # Deterministic preprocessing only (no augmentation)
    train_transform = build_classification_transform(is_train=False)
    val_transform = build_classification_transform(is_train=False)

    labels_train_dir = cfg.dataset_root / "labels" / "train"
    labels_test_dir = cfg.dataset_root / "labels" / "test"

    train_dataset = YOLOClassificationDataset(
        cfg.train_dir,
        labels_train_dir,
        transform=train_transform,
    )

    # Prefer explicit test split; fallback to split from train if test split is missing.
    if cfg.test_dir.exists() and labels_test_dir.exists():
        val_dataset = YOLOClassificationDataset(
            cfg.test_dir,
            labels_test_dir,
            transform=val_transform,
            cache_labels=True,
        )
        # Compute class weights from train split
        train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
        pos_count = train_labels.sum().item()
        neg_count = len(train_labels) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)])

        sampler = None
        if pos_count > 0 and neg_count > 0:
            class_counts = torch.tensor([neg_count, pos_count], dtype=torch.float32)
            weights = 1.0 / class_counts
            sample_weights = weights[train_labels.long()]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, pos_weight.item()

    # Fallback: split train into train/val
    train_indices, val_indices = split_dataset(train_dataset, cfg.val_split, seed=cfg.seed)
    train_subset = Subset(train_dataset, train_indices)
    val_dataset_fallback = YOLOClassificationDataset(
        cfg.train_dir,
        labels_train_dir,
        transform=val_transform,
        cache_labels=True,
    )
    val_subset = Subset(val_dataset_fallback, val_indices)

    # Compute class weights for imbalance handling on the train subset.
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    pos_count = train_labels.sum().item()
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)])

    sampler = None
    if pos_count > 0 and neg_count > 0:
        class_counts = torch.tensor([neg_count, pos_count], dtype=torch.float32)
        weights = 1.0 / class_counts
        sample_weights = weights[train_labels.long()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, pos_weight.item()


def train_classifier(cfg: ClassifierConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, pos_weight = make_dataloaders(cfg)
    model = build_classifier(device, cfg.classifier_weights)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_auc = -math.inf
    best_path = Path("checkpoints/classifier")
    best_path.mkdir(parents=True, exist_ok=True)
    best_path = best_path / "best.pt"

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(cfg.max_epochs):
        freeze_backbone = epoch < cfg.freeze_backbone_epochs
        for name, param in model.named_parameters():
            if name.startswith("layer") or name.startswith("conv1") or name.startswith("bn1"):
                param.requires_grad = not freeze_backbone

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            preds = (logits.sigmoid() >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, val_recall, val_precision, val_auc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{cfg.max_epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Rec {val_recall:.4f} "
            f"Prec {val_precision:.4f} AUC {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print(f"Saved new best classifier to {best_path}")

    return best_path


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = logits.sigmoid()
        preds = (probs >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        all_probs.extend(probs.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    val_loss = total_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    auc = compute_auc(all_labels, all_probs)
    return val_loss, val_acc, recall, precision, auc


def compute_auc(labels: list[int], probs: list[float]) -> float:
    """Compute ROC-AUC with a simple trapezoidal integration."""
    if len(set(labels)) < 2:
        return 0.0
    pairs = sorted(zip(probs, labels), reverse=True)
    tps = 0.0
    fps = 0.0
    tp_prev = 0.0
    fp_prev = 0.0
    auc = 0.0
    pos_total = sum(labels)
    neg_total = len(labels) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.0

    for prob, label in pairs:
        if label == 1:
            tps += 1
        else:
            fps += 1
        auc += (fps - fp_prev) * (tps + tp_prev) / 2.0
        tp_prev, fp_prev = tps, fps

    auc /= (pos_total * neg_total)
    return auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the damage classifier")
    parser.add_argument("--device", default="cuda:1", help="Device to use (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="Override maximum epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Override batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClassifierConfig(device=args.device)
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    best_path = train_classifier(cfg)
    print(f"Training finished. Best model stored at {best_path}")


if __name__ == "__main__":
    main()