"""Full model evaluation: mAP, precision, recall, confusion matrix, counting accuracy."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_label_file(label_path: Path) -> dict[int, int]:
    """Return class → count dict from a YOLO label file."""
    counts: dict[int, int] = defaultdict(int)
    if not label_path.exists():
        return counts
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(parts[0])] += 1
    return dict(counts)


def _count_accuracy(pred_counts: dict[int, int], gt_counts: dict[int, int]) -> float:
    """1.0 if every class count matches exactly, else fraction of classes correct."""
    all_classes = set(pred_counts) | set(gt_counts)
    if not all_classes:
        return 1.0
    correct = sum(1 for c in all_classes if pred_counts.get(c, 0) == gt_counts.get(c, 0))
    return correct / len(all_classes)


def _mean_absolute_count_error(pred_counts: dict[int, int], gt_counts: dict[int, int]) -> float:
    all_classes = set(pred_counts) | set(gt_counts)
    if not all_classes:
        return 0.0
    return sum(abs(pred_counts.get(c, 0) - gt_counts.get(c, 0)) for c in all_classes) / len(all_classes)


def evaluate_model(
    weights_path: str | Path,
    dataset_yaml: str | Path = "config/dataset.yaml",
    config_path: str | Path = "config/config.yaml",
    split: str = "test",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation on the test split.

    Args:
        weights_path: Path to trained ``best.pt`` weights.
        dataset_yaml: YOLO dataset YAML file.
        config_path: Project config.yaml for class names and thresholds.
        split: Dataset split to evaluate on (``train``, ``val``, or ``test``).
        output_dir: Directory to save the evaluation report and confusion matrix.

    Returns:
        Structured evaluation report dict.
    """
    cfg = _load_config(config_path)
    class_names: list[str] = cfg["classes"]
    conf_threshold: float = cfg["inference"].get("conf_threshold", 0.5)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_dir = Path(output_dir) if output_dir else weights_path.parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model from %s", weights_path)
    model = YOLO(str(weights_path))

    log.info("Running validation on '%s' split …", split)
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        conf=conf_threshold,
        save_json=True,
        plots=True,
        project=str(output_dir),
        name="eval",
        exist_ok=True,
    )

    # --- Standard metrics ----
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    per_class_p = metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else list(metrics.box.p)
    per_class_r = metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else list(metrics.box.r)
    per_class_f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(per_class_p, per_class_r)
    ]

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": round(per_class_p[i], 4) if i < len(per_class_p) else None,
            "recall": round(per_class_r[i], 4) if i < len(per_class_r) else None,
            "f1": round(per_class_f1[i], 4) if i < len(per_class_f1) else None,
        }

    # --- Counting accuracy on test images ---------------------------------
    ds_cfg_data: dict[str, Any]
    with open(dataset_yaml) as f:
        ds_cfg_data = yaml.safe_load(f)

    test_images_dir = Path(ds_cfg_data["path"]) / ds_cfg_data.get("test", "images/test")
    test_labels_dir = test_images_dir.parent.parent / "labels" / "test"

    image_paths = sorted(p for p in test_images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    count_accuracies: list[float] = []
    mace_values: list[float] = []

    for img_path in image_paths:
        gt_label = test_labels_dir / img_path.with_suffix(".txt").name
        gt_counts = _parse_label_file(gt_label)

        preds = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        pred_counts: dict[int, int] = defaultdict(int)
        if preds and preds[0].boxes is not None:
            for cls_id in preds[0].boxes.cls.tolist():
                pred_counts[int(cls_id)] += 1

        count_accuracies.append(_count_accuracy(dict(pred_counts), gt_counts))
        mace_values.append(_mean_absolute_count_error(dict(pred_counts), gt_counts))

    avg_count_accuracy = float(np.mean(count_accuracies)) if count_accuracies else 0.0
    avg_mace = float(np.mean(mace_values)) if mace_values else 0.0

    # --- Worst performing classes -----------------------------------------
    sorted_classes = sorted(per_class_metrics.items(), key=lambda x: x[1]["f1"] or 0)
    worst_classes = [name for name, _ in sorted_classes[:3]]

    report: dict[str, Any] = {
        "weights": str(weights_path),
        "split": split,
        "overall": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "counting_accuracy": round(avg_count_accuracy, 4),
            "mean_absolute_count_error": round(avg_mace, 4),
        },
        "per_class": per_class_metrics,
        "worst_classes": worst_classes,
        "output_dir": str(output_dir),
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Evaluation report → %s", report_path)

    log.info(
        "mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | Counting Acc=%.4f",
        map50, map50_95, avg_count_accuracy,
    )
    return report


def main():
    import click

    @click.command()
    @click.option("--weights", required=True, type=click.Path(exists=True))
    @click.option("--dataset-yaml", default="config/dataset.yaml", show_default=True)
    @click.option("--config", default="config/config.yaml", show_default=True)
    @click.option("--split", default="test", show_default=True, type=click.Choice(["train", "val", "test"]))
    @click.option("--output-dir", default=None, type=click.Path())
    def cli(weights, dataset_yaml, config, split, output_dir):
        """Evaluate a trained YOLOv8 model."""
        report = evaluate_model(weights, dataset_yaml, config, split, output_dir)
        print(json.dumps(report["overall"], indent=2))

    cli()


if __name__ == "__main__":
    main()
