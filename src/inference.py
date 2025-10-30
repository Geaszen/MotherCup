"""Run the cascaded classifier + detector pipeline on a folder of images."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import pathlib
# import torch.serialization

# torch.serialization.add_safe_globals([pathlib.PosixPath])

from config import DEFAULT_CLASSES
from models import build_classifier, build_mask_rcnn, build_svdd_backbone
from transforms import build_classification_transform


def load_classifier(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_classifier(device)
    # torch.serialization.add_safe_globals([pathlib.PosixPath])
    state = torch.load(checkpoint_path, map_location=device)
    # support {"model": state_dict, ...} or raw state_dict
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_svdd_classifier(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor, float]:
    """Load DeepSVDD classifier along with center c and threshold T."""
    # torch.serialization.add_safe_globals([pathlib.PosixPath])
    state = torch.load(checkpoint_path, map_location=device)
    backbone = state.get("backbone", "resnet50")
    model = build_svdd_backbone(device=device, backbone=backbone, weight_path=None)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()
    center = state.get("center", None)
    if center is None:
        raise RuntimeError("SVDD checkpoint missing 'center'.")
    if not torch.is_tensor(center):
        center = torch.tensor(center, device=device, dtype=torch.float32)
    else:
        center = center.to(device)
    T = float(state.get("threshold", 0.0))
    return model, center, T


def load_detector(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, List[str], bool]:
    # torch.serialization.add_safe_globals([pathlib.PosixPath])
    state = torch.load(checkpoint_path, map_location=device)
    cfg_dict = state.get("cfg", {})
    class_names = list(cfg_dict.get("classes", DEFAULT_CLASSES))
    use_point_rend = bool(cfg_dict.get("use_point_rend", False))
    model = build_mask_rcnn(len(class_names) + 1, use_point_rend=use_point_rend)
    # support {"model": state_dict, ...} or raw state
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, use_point_rend


def run_inference(
    images_dir: Path,
    classifier_path: Path,
    detector_path: Path,
    output_path: Path,
    device: str = "cuda",
    cls_type: str = "bce",           # "bce" or "svdd"
    cls_threshold: float = 0.5,
    det_score_threshold: float = 0.4,
    mask_threshold: float = 0.5,
) -> Path:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    val_transform = build_classification_transform(is_train=False)

    if cls_type == "svdd":
        classifier, svdd_center, svdd_T = load_svdd_classifier(classifier_path, device_t)
        svdd_center = svdd_center.detach()
    else:
        classifier = load_classifier(classifier_path, device_t)
        svdd_center = None
        svdd_T = 0.0

    detector, class_names, _ = load_detector(detector_path, device_t)

    results: List[Dict] = []
    image_paths = sorted([p for p in Path(images_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        cls_tensor = val_transform(image).unsqueeze(0).to(device_t)
        with torch.no_grad():
            if cls_type == "svdd":
                z = classifier(cls_tensor)
                if z.ndim == 4:
                    z = torch.mean(z, dim=(2, 3))
                dist = torch.linalg.norm(z.squeeze(0) - svdd_center, ord=2).item()
                # map to probability in [0,1]: probability of "has damage" decreases with distance
                k = 10.0 / max(svdd_T, 1e-6)
                prob = 1.0 / (1.0 + math.exp(k * (dist - svdd_T)))
            else:
                prob = torch.sigmoid(classifier(cls_tensor).squeeze()).item()

        entry = {
            "image": str(image_path),
            "damage_probability": float(prob),
            "predictions": [],
        }

        if prob >= cls_threshold:
            det_tensor = F.to_tensor(image).unsqueeze(0).to(device_t)
            with torch.no_grad():
                detections = detector(det_tensor)[0]
            scores = detections["scores"].cpu().numpy()
            boxes = detections["boxes"].cpu().numpy()
            labels = detections["labels"].cpu().numpy()
            masks = detections["masks"].cpu().numpy()
            keep = scores >= det_score_threshold
            if np.any(keep):
                filtered_indices = np.where(keep)[0]
                # severity sorting (area * score) approximates "severity"
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                severity = scores * np.sqrt(np.maximum(areas, 1.0))
                order = filtered_indices[np.argsort(severity[filtered_indices])[::-1]]
                max_detections = 4
                detections_added = 0
                for idx in order:
                    label_idx = labels[idx]
                    if label_idx == 0 or label_idx - 1 >= len(class_names):
                        continue
                    box = boxes[idx]
                    score = scores[idx]
                    mask = masks[idx]
                    binary_mask = (mask[0] if mask.ndim == 3 else mask) >= mask_threshold
                    entry["predictions"].append(
                        {
                            "label": class_names[int(label_idx) - 1],
                            "score": float(score),
                            "bbox_xyxy": [float(x) for x in box.tolist()],
                            "mask_rle": mask_to_rle(binary_mask.astype(np.uint8)),
                        }
                    )
                    detections_added += 1
                    if detections_added >= max_detections:
                        break
        results.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        write_results_csv(results, output_path)
    print(f"Wrote inference results for {len(results)} images to {output_path}")
    return output_path


def mask_to_rle(mask: np.ndarray) -> Dict[str, List[int]]:
    """Convert a binary mask to run-length encoding for compact storage."""
    pixels = mask.flatten(order="F")
    rle = []
    run_start = None
    for idx, pixel in enumerate(pixels, start=1):
        if pixel and run_start is None:
            run_start = idx
        elif not pixel and run_start is not None:
            rle.extend([run_start, idx - run_start])
            run_start = None
    if run_start is not None:
        rle.extend([run_start, len(pixels) - run_start + 1])
    return {"counts": rle, "size": list(mask.shape)}


def write_results_csv(results: List[Dict], output_path: Path) -> None:
    """Save inference results into CSV file in the required competition format."""
    fieldnames = [
        "image",
        "damage_probability",
        "label",
        "score",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_rle",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            image_path = entry["image"]
            damage_prob = entry["damage_probability"]
            predictions = entry["predictions"] or [None]
            for pred in predictions:
                row = {
                    "image": image_path,
                    "damage_probability": f"{damage_prob:.6f}",
                    "label": "",
                    "score": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "mask_rle": "",
                }
                if pred is not None:
                    bbox = pred["bbox_xyxy"]
                    row.update(
                        {
                            "label": pred["label"],
                            "score": f"{pred['score']:.6f}",
                            "bbox_x1": f"{bbox[0]:.2f}",
                            "bbox_y1": f"{bbox[1]:.2f}",
                            "bbox_x2": f"{bbox[2]:.2f}",
                            "bbox_y2": f"{bbox[3]:.2f}",
                            "mask_rle": " ".join(map(str, pred["mask_rle"]["counts"])),
                        }
                    )
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cascade classifier + detector inference")
    parser.add_argument("--images", type=str, default="mathorcup/dataset3713/images/test")
    parser.add_argument("--classifier", type=str, default="checkpoints/classifier_svdd/best.pt")
    parser.add_argument("--detector", type=str, default="checkpoints/detector/best.pt")
    parser.add_argument("--output", type=str, default="mathorcup/test_result.csv")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cls-type", type=str, choices=["bce", "svdd"], default="svdd", help="Use BCE classifier or DeepSVDD")
    parser.add_argument("--cls-th", type=float, default=0.5)
    parser.add_argument("--det-th", type=float, default=0.4)
    parser.add_argument("--mask-th", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        images_dir=Path(args.images),
        classifier_path=Path(args.classifier),
        detector_path=Path(args.detector),
        output_path=Path(args.output),
        device=args.device,
        cls_type=args.cls_type,
        cls_threshold=args.cls_th,
        det_score_threshold=args.det_th,
        mask_threshold=args.mask_th,
    )


if __name__ == "__main__":
    main()