"""Train YOLOv8 on the synthetic ice cream shelf dataset.

Usage:
    python scripts/train_synthetic.py
    python scripts/train_synthetic.py --variant yolov8s --epochs 100 --batch 8
    python scripts/train_synthetic.py --variant yolov8m --epochs 50 --imgsz 960
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Project root so we can import src utilities
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.logger import get_logger
    log = get_logger("train_synthetic", log_dir=str(PROJECT_ROOT / "logs"))
except Exception:
    import logging
    log = logging.getLogger("train_synthetic")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


DATASET_YAML = PROJECT_ROOT / "data" / "synthetic_dataset" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "synthetic"
VALID_VARIANTS = ["yolov8n", "yolov8s", "yolov8m"]


def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("\nERROR: ultralytics is not installed.")
        print("Fix:   pip install ultralytics")
        sys.exit(1)


def _check_dataset() -> None:
    if not DATASET_YAML.exists():
        print(f"\nERROR: Dataset YAML not found at {DATASET_YAML}")
        print("Fix:   Extract synthetic_dataset.tar.gz into data/ and verify the path.")
        sys.exit(1)


def train(variant: str, epochs: int, batch: int, imgsz: int) -> None:
    """Run training and print a summary on completion."""
    from ultralytics import YOLO

    _check_dataset()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{variant}_{timestamp}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Training %s on synthetic dataset", variant)
    log.info("  epochs=%d  batch=%d  imgsz=%d", epochs, batch, imgsz)
    log.info("  dataset: %s", DATASET_YAML)
    log.info("  output:  runs/synthetic/%s", run_name)
    log.info("=" * 60)

    model = YOLO(f"{variant}.pt")  # downloads pretrained COCO weights on first use

    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=str(RUNS_DIR),
            name=run_name,
            patience=max(10, epochs // 5),
            verbose=True,
            exist_ok=True,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log.error("GPU out of memory. Try: --batch %d or --imgsz 416", batch // 2)
        raise

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    # Validation metrics on the val split
    log.info("\nRunning final validation…")
    metrics = model.val(data=str(DATASET_YAML), split="val", verbose=False)

    print("\n" + "=" * 60)
    print(f"  Training complete: {run_name}")
    print("=" * 60)
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    print(f"  Best weights   : {best_weights}")
    print("=" * 60)
    print(f"\nTo use in the dashboard, paste this path into the sidebar:")
    print(f"  {best_weights}")
    print(f"\nOr run evaluation:")
    print(f"  python scripts/evaluate_model.py --weights \"{best_weights}\"")


def main() -> None:
    _check_ultralytics()

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on the synthetic ice cream shelf dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default="yolov8n",
        help="YOLOv8 model variant (nano/small/medium)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch",  type=int, default=16, help="Batch size (reduce if OOM)")
    parser.add_argument("--imgsz",  type=int, default=640, help="Input image size")
    args = parser.parse_args()

    train(args.variant, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
