"""Evaluate a trained YOLOv8 model on the synthetic test set.

Prints per-class precision / recall / F1, overall mAP, and a
counting-accuracy spot-check across 5 random test images.
Saves a plain-text summary to runs/evaluation_report.txt.

Usage:
    python scripts/evaluate_model.py --weights runs/synthetic/yolov8n_*/weights/best.pt
    python scripts/evaluate_model.py --weights path/to/best.pt --split val
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.logger import get_logger
    log = get_logger("evaluate_model")
except Exception:
    import logging
    log = logging.getLogger("evaluate_model")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATASET_YAML = PROJECT_ROOT / "data" / "synthetic_dataset" / "dataset.yaml"
REPORT_PATH  = PROJECT_ROOT / "runs" / "evaluation_report.txt"

# Ground-truth label ids of the synthetic dataset (DATASET_YAML above) only.
# Model predictions are named via model.names, never this list.
CLASS_NAMES = [
    "magnum_classic", "magnum_almond", "magnum_white",
    "cornetto_vanilla", "cornetto_chocolate", "cornetto_strawberry",
    "popsicle_fruit", "popsicle_chocolate",
    "sandwich_ice_cream", "cup_ice_cream", "empty_slot",
]


def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)


def _parse_gt(label_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                counts[name] += 1
    return dict(counts)


def _counting_spot_check(model, n: int = 5) -> list[dict]:
    """Run prediction on *n* random test images and compare counts to ground truth."""
    import yaml

    with open(DATASET_YAML) as f:
        ds = yaml.safe_load(f)

    test_img_dir = Path(ds["path"]) / ds.get("test", "images/test")
    test_lbl_dir = Path(ds["path"]) / "labels" / "test"
    img_files = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    if not img_files:
        return []

    sample = random.sample(img_files, min(n, len(img_files)))
    rows = []

    for img_path in sample:
        lbl_path = test_lbl_dir / img_path.with_suffix(".txt").name
        gt = _parse_gt(lbl_path)

        preds = model.predict(str(img_path), conf=0.5, verbose=False)
        pred_counts: dict[str, int] = defaultdict(int)
        if preds and preds[0].boxes is not None:
            for cls_id in preds[0].boxes.cls.tolist():
                cid = int(cls_id)
                name = model.names.get(cid, str(cid))
                pred_counts[name] += 1

        all_cats = set(gt) | set(pred_counts)
        exact_matches = sum(
            1 for c in all_cats if gt.get(c, 0) == pred_counts.get(c, 0)
        )
        acc = exact_matches / len(all_cats) if all_cats else 1.0

        rows.append({
            "image": img_path.name,
            "gt": dict(gt),
            "pred": dict(pred_counts),
            "category_accuracy": round(acc, 3),
        })

    return rows


def evaluate(weights_path: Path, split: str) -> None:
    from ultralytics import YOLO

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        sys.exit(1)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML not found: {DATASET_YAML}")
        sys.exit(1)

    log.info("Loading model: %s", weights_path)
    model = YOLO(str(weights_path))

    log.info("Running validation on '%s' split…", split)
    metrics = model.val(data=str(DATASET_YAML), split=split, verbose=False)

    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp       = float(metrics.box.mp)
    mr       = float(metrics.box.mr)

    per_class_p = (metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else list(metrics.box.p))
    per_class_r = (metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else list(metrics.box.r))

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"  Evaluation Report — {weights_path.name}")
    lines.append(f"  Split: {split}")
    lines.append("=" * 64)
    lines.append(f"\n  mAP@0.5        : {map50:.4f}")
    lines.append(f"  mAP@0.5:0.95   : {map50_95:.4f}")
    lines.append(f"  Mean Precision : {mp:.4f}")
    lines.append(f"  Mean Recall    : {mr:.4f}")

    lines.append(f"\n{'─'*64}")
    lines.append(f"  {'Class':<28} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    lines.append(f"{'─'*64}")
    for i in sorted(model.names):
        name = model.names[i]
        p = per_class_p[i] if i < len(per_class_p) else 0.0
        r = per_class_r[i] if i < len(per_class_r) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"  {name:<28} {p:>9.4f} {r:>9.4f} {f1:>9.4f}")

    lines.append(f"\n{'─'*64}")
    lines.append("  Counting Accuracy Spot-Check (5 random test images)")
    lines.append(f"{'─'*64}")
    spot = _counting_spot_check(model, n=5)
    if spot:
        for row in spot:
            lines.append(f"  {row['image']}")
            lines.append(f"    Ground truth : {row['gt']}")
            lines.append(f"    Predicted    : {row['pred']}")
            lines.append(f"    Category acc : {row['category_accuracy']:.1%}")
        avg_acc = sum(r["category_accuracy"] for r in spot) / len(spot)
        lines.append(f"\n  Average counting accuracy: {avg_acc:.1%}")
    else:
        lines.append("  (No test images found)")

    lines.append("=" * 64)
    report = "\n".join(lines)

    print("\n" + report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    log.info("Report saved → %s", REPORT_PATH)


def main() -> None:
    _check_ultralytics()

    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 model on the synthetic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, type=Path, help="Path to best.pt")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    # Support glob patterns passed by the shell
    weights = args.weights
    if not weights.exists():
        candidates = sorted(PROJECT_ROOT.glob(str(args.weights)))
        if candidates:
            weights = candidates[0]
        else:
            print(f"ERROR: No weights found matching: {args.weights}")
            sys.exit(1)

    evaluate(weights, args.split)


if __name__ == "__main__":
    main()
