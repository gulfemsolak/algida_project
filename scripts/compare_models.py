"""Compare trained YOLOv8 variants (nano / small / medium).

Discovers weights automatically under runs/synthetic/, evaluates each on
the test split, and prints a ranked comparison table. Saves results to
runs/model_comparison.json.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --split val
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.logger import get_logger
    log = get_logger("compare_models")
except Exception:
    import logging
    log = logging.getLogger("compare_models")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATASET_YAML   = PROJECT_ROOT / "data" / "synthetic_dataset" / "dataset.yaml"
SYNTHETIC_RUNS = PROJECT_ROOT / "runs" / "synthetic"
OUTPUT_JSON    = PROJECT_ROOT / "runs" / "model_comparison.json"


def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)


def _discover_weights() -> dict[str, Path]:
    """Return {variant_name: path_to_best.pt} for all runs found in runs/synthetic/."""
    found: dict[str, Path] = {}
    if not SYNTHETIC_RUNS.exists():
        return found
    for best in sorted(SYNTHETIC_RUNS.rglob("best.pt")):
        # Expect path: runs/synthetic/<variant>_<timestamp>/weights/best.pt
        exp_name = best.parent.parent.name          # e.g. "yolov8n_20260701_120000"
        variant  = exp_name.split("_")[0]           # e.g. "yolov8n"
        # Keep only the most-recently-trained run per variant (sorted → last wins)
        found[variant] = best
    return found


def _model_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_048_576, 2)


def _measure_speed(model, n_images: int = 20) -> float:
    """Return mean inference time in ms over *n_images* dummy passes."""
    import numpy as np
    dummy = (255 * __import__("numpy").random.rand(640, 640, 3)).astype(__import__("numpy").uint8)
    # Warmup
    for _ in range(3):
        model.predict(dummy, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n_images):
        model.predict(dummy, verbose=False)
    return round((time.perf_counter() - t0) / n_images * 1000, 2)


def compare(split: str) -> None:
    from ultralytics import YOLO

    weights_map = _discover_weights()

    if not weights_map:
        print("\nNo trained models found under runs/synthetic/")
        print("Train at least one model first:")
        print("  python scripts/train_synthetic.py --variant yolov8n --epochs 50")
        print("  python scripts/train_synthetic.py --variant yolov8s --epochs 50")
        print("  python scripts/train_synthetic.py --variant yolov8m --epochs 50")
        sys.exit(0)

    if len(weights_map) < 2:
        variant = list(weights_map)[0]
        print(f"\nOnly 1 model found ({variant}). Train at least 2 for a meaningful comparison.")
        print("Continuing with single-model evaluation…\n")

    results: list[dict] = []

    for variant, weights_path in sorted(weights_map.items()):
        log.info("Evaluating %s — %s", variant, weights_path)
        model = YOLO(str(weights_path))

        metrics = model.val(data=str(DATASET_YAML), split=split, verbose=False)
        speed_ms = _measure_speed(model)

        entry = {
            "variant":       variant,
            "weights_path":  str(weights_path),
            "model_size_mb": _model_size_mb(weights_path),
            "mAP50":         round(float(metrics.box.map50), 4),
            "mAP50_95":      round(float(metrics.box.map),   4),
            "precision":     round(float(metrics.box.mp),    4),
            "recall":        round(float(metrics.box.mr),    4),
            "inference_ms":  speed_ms,
        }
        results.append(entry)
        log.info(
            "  %s → mAP50=%.4f  speed=%.1f ms  size=%.1f MB",
            variant, entry["mAP50"], speed_ms, entry["model_size_mb"],
        )

    # Rank by mAP50
    results.sort(key=lambda x: x["mAP50"], reverse=True)

    # Recommendations
    fastest   = min(results, key=lambda x: x["inference_ms"])
    accurate  = results[0]   # already sorted by mAP50
    smallest  = min(results, key=lambda x: x["model_size_mb"])

    recommendations = {
        "edge_deployment":     f"{fastest['variant']} ({fastest['inference_ms']:.1f} ms/img)",
        "cloud_deployment":    f"{accurate['variant']} (mAP50={accurate['mAP50']:.4f})",
        "resource_constrained": f"{smallest['variant']} ({smallest['model_size_mb']:.1f} MB)",
        "balanced_choice":     accurate["variant"],
    }

    # Print table
    col_w = [14, 8, 8, 10, 10, 14, 10]
    headers = ["Variant", "mAP50", "mAP@.95", "Precision", "Recall", "Speed (ms)", "Size (MB)"]
    row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + "=" * 76)
    print("  Model Comparison — YOLOv8 Variants")
    print("=" * 76)
    print(row_fmt.format(*headers))
    print("  " + "─" * 72)
    for r in results:
        print(row_fmt.format(
            r["variant"], f"{r['mAP50']:.4f}", f"{r['mAP50_95']:.4f}",
            f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['inference_ms']:.1f}", f"{r['model_size_mb']:.1f}",
        ))
    print("=" * 76)
    print("\n  Deployment Recommendations:")
    for scenario, rec in recommendations.items():
        print(f"    {scenario:<25} → {rec}")

    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": results, "recommendations": recommendations}
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n  Comparison report saved → {OUTPUT_JSON}")


def main() -> None:
    _check_ultralytics()

    parser = argparse.ArgumentParser(
        description="Compare trained YOLOv8 variants on the synthetic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()
    compare(args.split)


if __name__ == "__main__":
    main()
