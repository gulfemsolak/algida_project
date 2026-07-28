"""Run YOLOv8 prediction on a single image or a folder of images.

Every detection is printed to stdout and the annotated image is saved
to runs/predictions/.

Usage:
    # Single image
    python scripts/predict_demo.py --weights path/to/best.pt \\
        --image data/synthetic_dataset/images/test/shelf_test_0001.jpg

    # Whole folder
    python scripts/predict_demo.py --weights path/to/best.pt \\
        --folder data/synthetic_dataset/images/test/

    # Custom confidence threshold
    python scripts/predict_demo.py --weights path/to/best.pt \\
        --image shelf.jpg --conf 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.logger import get_logger
    log = get_logger("predict_demo")
except Exception:
    import logging
    log = logging.getLogger("predict_demo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

OUTPUT_DIR = PROJECT_ROOT / "runs" / "predictions"


def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)


def _predict_one(model, img_path: Path, conf: float, out_dir: Path) -> None:
    """Run prediction on *img_path*, print results, save annotated image."""
    import cv2

    results = model.predict(str(img_path), conf=conf, verbose=False)
    res = results[0]

    print(f"\n{'─'*56}")
    print(f"  Image: {img_path.name}")
    print(f"{'─'*56}")

    if res.boxes is None or len(res.boxes) == 0:
        print("  No detections above confidence threshold.")
    else:
        for box in res.boxes:
            cid  = int(box.cls.item())
            name = model.names.get(cid, str(cid))
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0].tolist()]
            print(f"  {name:<28}  conf={conf_val:.3f}  bbox=[{x1},{y1},{x2},{y2}]")

        print(f"\n  Total detections: {len(res.boxes)}")

    # Save annotated image
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{img_path.stem}.jpg"
    annotated = res.plot()  # BGR numpy array with boxes drawn
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved → {out_path}")


def predict(weights_path: Path, img_path: Path | None, folder: Path | None, conf: float) -> None:
    from ultralytics import YOLO

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        sys.exit(1)

    log.info("Loading model: %s", weights_path)
    model = YOLO(str(weights_path))

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if folder is not None:
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in image_exts)
        if not images:
            print(f"No images found in {folder}")
            sys.exit(1)
        log.info("Processing %d images from %s", len(images), folder)
        for img in images:
            _predict_one(model, img, conf, OUTPUT_DIR)
    elif img_path is not None:
        if not img_path.exists():
            print(f"ERROR: Image not found: {img_path}")
            sys.exit(1)
        _predict_one(model, img_path, conf, OUTPUT_DIR)
    else:
        print("ERROR: Provide --image or --folder")
        sys.exit(1)

    print(f"\nAll annotated images saved to: {OUTPUT_DIR}")


def main() -> None:
    _check_ultralytics()

    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference and save annotated images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, type=Path, help="Path to best.pt")
    parser.add_argument("--image",  default=None, type=Path, help="Single image to process")
    parser.add_argument("--folder", default=None, type=Path, help="Folder of images to process")
    parser.add_argument("--conf",   default=0.5,  type=float, help="Confidence threshold")
    args = parser.parse_args()

    if args.image is None and args.folder is None:
        parser.error("Provide --image <path> or --folder <path>")

    predict(args.weights, args.image, args.folder, args.conf)


if __name__ == "__main__":
    main()
