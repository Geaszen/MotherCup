"""Evaluate classifier + detector on a validation set and export predictions."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.ops import box_iou

from data_utils import (
    cxcywh_to_xyxy,
    list_image_files,
    read_yolo_annotations,
    yolo_to_mask,
    yolo_txt_path_rel,
)
from inference import (
    load_classifier,
    load_detector,
    load_svdd_classifier,
    mask_to_rle,
    write_results_csv,
)
from transforms import build_classification_transform


def compute_binary_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute Accuracy/Precision/Recall/F1 and ranking metrics when possible."""
    assert labels.shape == probs.shape, "labels and probs must align"
    preds = (probs >= threshold).astype(np.int32)
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    total = labels.size

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))

    if positives > 0 and negatives > 0:
        metrics["roc_auc"] = compute_roc_auc(labels, probs)
        metrics["auprc"] = compute_auprc(labels, probs)

    return metrics


def compute_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal integration."""
    order = np.argsort(probs)[::-1]
    labels_sorted = labels[order]
    probs_sorted = probs[order]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / tps[-1] if tps[-1] else np.zeros_like(tps, dtype=np.float64)
    fpr = fps / fps[-1] if fps[-1] else np.zeros_like(fps, dtype=np.float64)

    # prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def compute_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute area under the precision-recall curve."""
    order = np.argsort(probs)[::-1]
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / tp[-1] if tp[-1] else np.zeros_like(tp, dtype=np.float64)

    # prepend starting point
    precision = np.concatenate([[precision[0]], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def compute_detection_metrics(
    per_class_scores: Dict[int, List[float]],
    per_class_tp: Dict[int, List[int]],
    per_class_total: Dict[int, int],
    num_classes: int,
) -> Tuple[float, float, float, Dict[int, float]]:
    """Return precision, recall, mAP@0.5 and per-class AP."""
    total_tp = 0.0
    total_fp = 0.0
    total_gt = float(sum(per_class_total.values()))
    ap_per_class: Dict[int, float] = {}

    for cls in range(1, num_classes + 1):
        total_gt_cls = float(per_class_total.get(cls, 0))
        scores = per_class_scores.get(cls, [])
        tps = per_class_tp.get(cls, [])
        if not scores or not tps:
            ap_per_class[cls] = 0.0
            continue
        scores_np = np.array(scores)
        tps_np = np.array(tps)
        order = np.argsort(scores_np)[::-1]
        tps_sorted = tps_np[order]
        fps_sorted = 1 - tps_sorted
        tp_cum = np.cumsum(tps_sorted)
        fp_cum = np.cumsum(fps_sorted)
        if total_gt_cls:
            recalls = tp_cum / total_gt_cls
        else:
            recalls = np.zeros_like(tp_cum, dtype=np.float64)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        total_tp += float(tp_cum[-1])
        total_fp += float(fp_cum[-1])
        ap_per_class[cls] = average_precision(recalls, precisions)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    map50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return precision, recall, map50, ap_per_class


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Common 101-point interpolated average precision."""
    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_levels:
        mask = recalls >= r
        precision_at_r = np.max(precisions[mask]) if np.any(mask) else 0.0
        ap += precision_at_r
    return ap / len(recall_levels)


def evaluate_pipeline(
    images_dir: Path,
    labels_dir: Path,
    classifier_path: Path,
    detector_path: Path,
    output_csv: Path,
    device: str = "cuda",
    cls_type: str = "bce",
    cls_threshold: float = 0.5,
    det_score_threshold: float = 0.4,
    mask_threshold: float = 0.5,
    max_detections: int = 4,
) -> None:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = build_classification_transform(is_train=False)

    if cls_type == "svdd":
        classifier, svdd_center, svdd_T = load_svdd_classifier(classifier_path, device_t)
        svdd_center = svdd_center.detach()
    else:
        classifier = load_classifier(classifier_path, device_t)
        svdd_center = None
        svdd_T = 0.0

    detector, class_names, _ = load_detector(detector_path, device_t)
    num_classes = len(class_names)

    image_paths = list_image_files(images_dir)

    classification_labels: List[int] = []
    classification_probs: List[float] = []
    results_for_csv: List[Dict] = []

    per_class_scores: Dict[int, List[float]] = defaultdict(list)
    per_class_tp: Dict[int, List[int]] = defaultdict(list)
    per_class_total: Dict[int, int] = defaultdict(int)
    true_positive_masks: List[float] = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = yolo_txt_path_rel(Path(image_path), images_dir, labels_dir)
        boxes_cxcywh, labels = read_yolo_annotations(label_path)
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh, width, height)
        masks = yolo_to_mask(boxes_xyxy, height, width).float()
        gt_labels = (labels + 1).to(device_t)
        gt_boxes = boxes_xyxy.to(device_t)
        gt_masks = masks.to(device_t)

        classification_labels.append(1 if gt_labels.numel() > 0 else 0)

        cls_tensor = transform(image).unsqueeze(0).to(device_t)
        with torch.no_grad():
            if cls_type == "svdd":
                features = classifier(cls_tensor)
                if features.ndim == 4:
                    features = torch.mean(features, dim=(2, 3))
                dist = torch.linalg.norm(features.squeeze(0) - svdd_center, ord=2).item()
                k = 10.0 / max(svdd_T, 1e-6)
                prob = 1.0 / (1.0 + math.exp(k * (dist - svdd_T)))
            else:
                prob = torch.sigmoid(classifier(cls_tensor).squeeze()).item()
        classification_probs.append(prob)

        det_tensor = F.to_tensor(image).unsqueeze(0).to(device_t)
        with torch.no_grad():
            output = detector(det_tensor)[0]

        scores = output["scores"].to(device_t)
        pred_boxes = output["boxes"].to(device_t)
        pred_labels = output["labels"].to(device_t)
        pred_masks = output.get("masks")
        if pred_masks is not None:
            pred_masks = pred_masks.to(device_t)

        for label in gt_labels.tolist():
            if label > 0:
                per_class_total[int(label)] += 1

        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            for label, score in zip(pred_labels.tolist(), scores.tolist()):
                if label == 0:
                    continue
                per_class_scores[label].append(float(score))
                per_class_tp[label].append(0)
        else:
            ious = box_iou(pred_boxes, gt_boxes)
            order = torch.argsort(scores, descending=True)
            matched_gt = {cls: set() for cls in range(1, num_classes + 1)}

            for idx in order.tolist():
                label = int(pred_labels[idx].item())
                if label == 0:
                    continue
                score = float(scores[idx].item())
                per_class_scores[label].append(score)
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
                    if pred_masks is not None and gt_masks.numel() > 0:
                        gt_mask = (gt_masks[best_gt] > 0.5).float()
                        pred_mask = pred_masks[idx]
                        if pred_mask.ndim == 3:
                            pred_mask = pred_mask[0]
                        pred_mask = (pred_mask > mask_threshold).float()
                        inter = float((gt_mask * pred_mask).sum().item())
                        union = float(gt_mask.sum().item() + pred_mask.sum().item() - inter)
                        if union > 0:
                            true_positive_masks.append(inter / union)
                else:
                    per_class_tp[label].append(0)

        # Prepare predictions for CSV following inference ordering
        entry = {
            "image": str(image_path.relative_to(images_dir)),
            "damage_probability": float(prob),
            "predictions": [],
        }
        if prob >= cls_threshold:
            keep = scores >= det_score_threshold
            if torch.any(keep):
                indices = torch.where(keep)[0]
                boxes_np = pred_boxes[indices].cpu().numpy()
                scores_np = scores[indices].cpu().numpy()
                areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
                severity = scores_np * np.sqrt(np.maximum(areas, 1.0))
                order_csv = indices[np.argsort(severity)[::-1]][:max_detections]

                for idx in order_csv.tolist():
                    label_idx = int(pred_labels[idx].item())
                    if label_idx == 0 or label_idx - 1 >= num_classes:
                        continue
                    bbox = pred_boxes[idx].cpu().numpy()
                    score = float(scores[idx].item())
                    if pred_masks is not None:
                        mask_arr = pred_masks[idx].cpu().numpy()
                        binary_mask = (mask_arr[0] if mask_arr.ndim == 3 else mask_arr) >= mask_threshold
                    else:
                        binary_mask = np.zeros((height, width), dtype=np.uint8)
                    entry["predictions"].append(
                        {
                            "label": class_names[label_idx - 1],
                            "score": score,
                            "bbox_xyxy": [float(x) for x in bbox.tolist()],
                            "mask_rle": mask_to_rle(binary_mask.astype(np.uint8)),
                        }
                    )
        results_for_csv.append(entry)

    # Aggregate metrics
    cls_labels_np = np.array(classification_labels, dtype=np.int32)
    cls_probs_np = np.array(classification_probs, dtype=np.float64)
    cls_metrics = compute_binary_metrics(cls_labels_np, cls_probs_np, cls_threshold)

    precision, recall, map50, ap_per_class = compute_detection_metrics(
        per_class_scores, per_class_tp, per_class_total, num_classes
    )
    miou = float(sum(true_positive_masks) / len(true_positive_masks)) if true_positive_masks else 0.0

    print("Classification metrics:")
    for key, value in cls_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nDetection metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  mAP@0.5: {map50:.4f}")
    for cls_idx, ap in ap_per_class.items():
        name = class_names[cls_idx - 1] if 1 <= cls_idx <= num_classes else str(cls_idx)
        print(f"    AP[{name}]: {ap:.4f}")
    print(f"  mIoU: {miou:.4f}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_results_csv(results_for_csv, output_csv)
    print(f"Saved inference predictions to {output_csv}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cascade pipeline on a validation set")
    parser.add_argument("--images", type=str, default="dataset3713/images/valid")
    parser.add_argument("--labels", type=str, default="dataset3713/labels/valid")
    parser.add_argument("--classifier", type=str, default="checkpoints/classifier/best.pt")
    parser.add_argument("--detector", type=str, default="checkpoints/detector/best.pt")
    parser.add_argument("--output", type=str, default="test_result.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cls-type", type=str, choices=["bce", "svdd"], default="bce")
    parser.add_argument("--cls-th", type=float, default=0.5)
    parser.add_argument("--det-th", type=float, default=0.4)
    parser.add_argument("--mask-th", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    evaluate_pipeline(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        classifier_path=Path(args.classifier),
        detector_path=Path(args.detector),
        output_csv=Path(args.output),
        device=args.device,
        cls_type=args.cls_type,
        cls_threshold=args.cls_th,
        det_score_threshold=args.det_th,
        mask_threshold=args.mask_th,
        max_detections=args.max_det,
    )


if __name__ == "__main__":
    main()
