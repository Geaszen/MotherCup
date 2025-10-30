"""Fine-tune Mask R-CNN for damage detection and segmentation."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from config import DetectorConfig
from data_utils import YOLOMaskDataset, split_dataset
from models import build_mask_rcnn
from transforms import build_detection_transforms
device = torch.device("cuda:0")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def make_dataloaders(cfg: DetectorConfig) -> Tuple[DataLoader, DataLoader]:
    train_transforms = build_detection_transforms(is_train=False)
    val_transforms = build_detection_transforms(is_train=False)

    train_dataset = YOLOMaskDataset(
        cfg.train_dir,
        cfg.labels_dir,
        transforms=train_transforms,
        class_offset=1,
    )

    # Prefer explicit test split; fallback to split from train if test split is missing.
    if cfg.test_images_dir.exists() and cfg.test_labels_dir.exists():
        val_dataset = YOLOMaskDataset(
            cfg.test_images_dir,
            cfg.test_labels_dir,
            transforms=val_transforms,
            class_offset=1,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return train_loader, val_loader

    # Fallback: split train into train/val
    train_indices, val_indices = split_dataset(train_dataset, cfg.val_split, seed=cfg.seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(YOLOMaskDataset(cfg.train_dir, cfg.labels_dir, transforms=val_transforms, class_offset=1), val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_detector(cfg: DetectorConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(cfg)
    num_classes = len(cfg.classes) + 1  # background
    model = build_mask_rcnn(num_classes, use_point_rend=cfg.use_point_rend, weight_path=cfg.detector_weights)

    if cfg.resume_from is not None and cfg.resume_from.exists():
        state = torch.load(cfg.resume_from, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"Resumed detector weights from {cfg.resume_from}")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_map = -float("inf")
    best_path = Path("checkpoints/detector")
    best_path.mkdir(parents=True, exist_ok=True)
    best_path = best_path / "best2.pt"

    for epoch in range(cfg.max_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            running_loss += losses.item()
        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)

        metrics = evaluate(model, val_loader, device, cfg)
        print(
            f"Epoch {epoch+1}/{cfg.max_epochs} | Train Loss {avg_loss:.4f} | "
            f"Precision {metrics['precision']:.4f} Recall {metrics['recall']:.4f} "
            f"mAP@0.5 {metrics['map50']:.4f} mIoU {metrics['miou']:.4f}"
        )

        if metrics["map50"] > best_map:
            best_map = metrics["map50"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print(f"Saved new best detector to {best_path}")

    return best_path


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DetectorConfig,
) -> Dict[str, float]:
    model.eval()
    total_targets = 0
    total_predictions = 0
    true_positive_masks: List[float] = []

    from collections import defaultdict
    per_class_scores: Dict[int, List[float]] = defaultdict(list)
    per_class_tp: Dict[int, List[int]] = defaultdict(list)
    per_class_total: Dict[int, int] = defaultdict(int)

    from torchvision.ops import box_iou

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            gt_labels = target["labels"].to(device)
            gt_boxes = target["boxes"].to(device)
            gt_masks = target["masks"].to(device)
            total_targets += gt_labels.numel()
            for label in gt_labels.tolist():
                if label > 0:
                    per_class_total[int(label)] += 1

            scores = output["scores"].to(device)
            # 评估 AP 不应在这里筛阈值；保留所有预测进入 PR 曲线
            pred_boxes = output["boxes"].to(device)
            pred_labels = output["labels"].to(device)
            pred_masks = output["masks"].to(device)
            total_predictions += pred_labels.numel()

            if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                # 没有GT或没有框，所有预测都算作该类的FP，TP=0
                for label, score in zip(pred_labels.tolist(), scores.tolist()):
                    if label == 0:
                        continue
                    per_class_scores[label].append(float(score))
                    per_class_tp[label].append(0)
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            # 每次评估按分数从高到低贪心匹配
            order = torch.argsort(scores, descending=True)
            matched_gt: Dict[int, set] = {int(c): set() for c in range(1, len(cfg.classes) + 1)}

            for idx in order.tolist():
                label = int(pred_labels[idx].item())
                if label == 0:
                    continue
                per_class_scores[label].append(float(scores[idx].item()))
                gt_candidates = torch.where(gt_labels == label)[0]
                if gt_candidates.numel() == 0:
                    per_class_tp[label].append(0)
                    continue
                ious_with_gt = ious[idx, gt_candidates]
                best_iou, best_idx = torch.max(ious_with_gt, dim=0)
                best_gt = int(gt_candidates[best_idx].item())
                if best_iou >= 0.5 and best_gt not in matched_gt[label]:
                    per_class_tp[label].append(1)
                    matched_gt[label].add(best_gt)
                    # 掩码 mIoU 仅在 TP 上计算
                    gt_mask = (gt_masks[best_gt] > 0.5).float()
                    pred_mask = (pred_masks[idx] > cfg.mask_threshold).float()
                    inter = (gt_mask * pred_mask).sum().item()
                    union = gt_mask.sum().item() + pred_mask.sum().item() - inter
                    if union > 0:
                        true_positive_masks.append(inter / union)
                else:
                    per_class_tp[label].append(0)

    precision, recall, map50 = compute_pr_map(per_class_scores, per_class_tp, per_class_total, num_classes=len(cfg.classes))
    miou = float(sum(true_positive_masks) / max(len(true_positive_masks), 1))
    return {"precision": precision, "recall": recall, "map50": map50, "miou": miou}


def compute_pr_map(
    per_class_scores: Dict[int, List[float]],
    per_class_tp: Dict[int, List[int]],
    per_class_total: Dict[int, int],
    num_classes: int,
) -> Tuple[float, float, float]:
    total_tp = 0
    total_fp = 0
    total_gt = sum(per_class_total.values())
    ap_list: List[float] = []

    # 仅对验证集中“有GT的类别”计算 AP；若该类没有任何预测，则 AP=0
    classes_with_gt = [cls_id for cls_id in range(1, num_classes + 1) if per_class_total.get(cls_id, 0) > 0]
    for cls in classes_with_gt:
        scores = per_class_scores.get(cls, [])
        if not scores:
            ap_list.append(0.0)
            continue
        order = np.argsort(scores)[::-1]
        tps = np.array(per_class_tp[cls])[order]
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / max(per_class_total.get(cls, 1), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        total_tp += int(tp_cum[-1])
        total_fp += int(fp_cum[-1])
        ap = average_precision(recalls, precisions)
        ap_list.append(ap)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    map50 = float(np.mean(ap_list)) if ap_list else 0.0
    return precision, recall, map50


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using the Common 101-point interpolation."""
    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0
    for recall_level in recall_levels:
        precisions_above = precisions[recalls >= recall_level]
        if precisions_above.size == 0:
            precision = 0.0
        else:
            precision = np.max(precisions_above)
        ap += precision
    return ap / len(recall_levels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mask R-CNN detector")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", type=str, default="checkpoints/detector/best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DetectorConfig(device=args.device)
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.resume:
        cfg.resume_from = Path(args.resume)
    best_path = train_detector(cfg)
    print(f"Training finished. Best model stored at {best_path}")


if __name__ == "__main__":
    main()