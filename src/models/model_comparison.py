"""Compare YOLOv8n vs yolov8s vs yolov8m on the same test set.

Measures mAP, precision, recall, inference speed, model size, and
GPU memory (if available), then produces a ranked comparison table.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _model_size_mb(weights_path: Path) -> float:
    return round(weights_path.stat().st_size / 1_048_576, 2) if weights_path.exists() else 0.0


def _measure_inference_speed(
    model,
    image: np.ndarray,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> float:
    """Return average inference time in milliseconds over *n_measure* runs."""
    for _ in range(n_warmup):
        model.predict(image, verbose=False)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model.predict(image, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 2)


def _gpu_memory_mb() -> float | None:
    """Return current GPU memory allocated in MB, or None if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1_048_576, 2)
    except ImportError:
        pass
    return None


def compare_models(
    weights_map: dict[str, str | Path],
    dataset_yaml: str | Path = "config/dataset.yaml",
    config_path: str | Path = "config/config.yaml",
    output_dir: str | Path = "reports/model_comparison",
    benchmark_image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate and compare multiple trained YOLOv8 models.

    Args:
        weights_map: Dict of ``{variant_name: path_to_best.pt}``.
        dataset_yaml: YOLO dataset YAML.
        config_path: Project config.yaml.
        output_dir: Directory to save comparison report and charts.
        benchmark_image_path: Image used for speed benchmarking.
            If None, a random image from the test split is used.

    Returns:
        List of per-model result dicts, sorted by mAP50 descending.
    """
    from ultralytics import YOLO

    cfg = _load_config(config_path)
    conf_threshold: float = cfg["inference"].get("conf_threshold", 0.5)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick a benchmark image
    if benchmark_image_path is None:
        ds_data: dict[str, Any]
        with open(dataset_yaml) as f:
            ds_data = yaml.safe_load(f)
        test_dir = Path(ds_data["path"]) / ds_data.get("test", "images/test")
        img_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if img_files:
            benchmark_image_path = img_files[0]
        else:
            benchmark_image_path = None

    bench_image = None
    if benchmark_image_path:
        bench_image = cv2.imread(str(benchmark_image_path))

    results: list[dict[str, Any]] = []

    for variant, weights_path in weights_map.items():
        weights_path = Path(weights_path)
        log.info("Evaluating %s …", variant)

        if not weights_path.exists():
            log.warning("Weights not found for %s at %s — skipping", variant, weights_path)
            continue

        model = YOLO(str(weights_path))

        # Validation metrics
        val_metrics = model.val(
            data=str(dataset_yaml),
            split="test",
            conf=conf_threshold,
            verbose=False,
            project=str(output_dir / "val_runs"),
            name=variant,
            exist_ok=True,
        )

        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        precision = float(np.mean(val_metrics.box.p)) if hasattr(val_metrics.box, "p") else 0.0
        recall = float(np.mean(val_metrics.box.r)) if hasattr(val_metrics.box, "r") else 0.0

        # Speed
        speed_ms = 0.0
        if bench_image is not None:
            try:
                speed_ms = _measure_inference_speed(model, bench_image)
            except Exception as exc:
                log.warning("Speed benchmark failed for %s: %s", variant, exc)

        gpu_mem = _gpu_memory_mb()

        entry: dict[str, Any] = {
            "variant": variant,
            "weights_path": str(weights_path),
            "model_size_mb": _model_size_mb(weights_path),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "inference_ms": speed_ms,
            "gpu_memory_mb": gpu_mem,
        }
        results.append(entry)
        log.info(
            "%s → mAP50=%.4f | speed=%.1f ms | size=%.1f MB",
            variant, map50, speed_ms, entry["model_size_mb"],
        )

    results.sort(key=lambda x: x["mAP50"], reverse=True)

    # Deployment recommendations
    recommendations = _generate_recommendations(results)

    report = {
        "models": results,
        "recommendations": recommendations,
    }
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Comparison report → %s", report_path)

    return results


def _generate_recommendations(results: list[dict[str, Any]]) -> dict[str, str]:
    if not results:
        return {}
    fastest = min(results, key=lambda x: x.get("inference_ms") or float("inf"))
    most_accurate = max(results, key=lambda x: x["mAP50"])
    smallest = min(results, key=lambda x: x.get("model_size_mb") or float("inf"))
    return {
        "edge_deployment": f"{fastest['variant']} (fastest: {fastest['inference_ms']:.1f} ms/image)",
        "cloud_deployment": f"{most_accurate['variant']} (highest mAP50: {most_accurate['mAP50']:.4f})",
        "resource_constrained": f"{smallest['variant']} (smallest: {smallest['model_size_mb']:.1f} MB)",
    }


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--nano-weights", default=None, type=click.Path())
    @click.option("--small-weights", default=None, type=click.Path())
    @click.option("--medium-weights", default=None, type=click.Path())
    @click.option("--dataset-yaml", default="config/dataset.yaml", show_default=True)
    @click.option("--config", default="config/config.yaml", show_default=True)
    @click.option("--output-dir", default="reports/model_comparison", show_default=True)
    def cli(nano_weights, small_weights, medium_weights, dataset_yaml, config, output_dir):
        """Compare YOLOv8 model variants."""
        weights_map: dict[str, str] = {}
        if nano_weights:
            weights_map["yolov8n"] = nano_weights
        if small_weights:
            weights_map["yolov8s"] = small_weights
        if medium_weights:
            weights_map["yolov8m"] = medium_weights
        if not weights_map:
            raise click.UsageError("Provide at least one weights file.")
        results = compare_models(weights_map, dataset_yaml, config, output_dir)
        print(json.dumps(results, indent=2))

    cli()
