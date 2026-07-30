"""YOLOv8 training wrapper with config-driven experiment management."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _experiment_name(model_variant: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_variant}_{ts}"


def _export_metrics_csv(results_dir: Path, output_csv: Path) -> None:
    """Convert Ultralytics results.csv into a clean training-metrics CSV."""
    src = results_dir / "results.csv"
    if not src.exists():
        log.warning("results.csv not found at %s", src)
        return
    shutil.copy(src, output_csv)
    log.info("Training metrics exported → %s", output_csv)


def train_model(
    model_variant: str = "yolov8m",
    config_path: str | Path = "config/config.yaml",
    resume: bool = False,
    resume_weights: str | Path | None = None,
) -> dict[str, Any]:
    """Train a YOLOv8 model with hyperparameters from config.yaml.

    Args:
        model_variant: One of ``yolov8n``, ``yolov8s``, ``yolov8m``.
        config_path: Path to config.yaml.
        resume: If True, resume an interrupted training run.
        resume_weights: Path to weights to resume from (required when resume=True).

    Returns:
        Dict with paths to best weights, training metrics CSV, and experiment dir.
    """
    cfg = _load_config(config_path)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    dataset_yaml = Path("config/dataset.yaml")

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml}. "
            "Run dataset_builder.build_dataset() first."
        )

    experiment = _experiment_name(model_variant)
    runs_dir = Path(paths_cfg.get("runs_dir", "runs"))
    exp_dir = runs_dir / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting experiment: %s", experiment)
    log.info("Model variant: %s", model_variant)
    log.info("Dataset YAML: %s", dataset_yaml)

    if resume and resume_weights:
        model = YOLO(str(resume_weights))
        log.info("Resuming from: %s", resume_weights)
    else:
        model = YOLO(f"{model_variant}.pt")

    results = model.train(
        data=str(dataset_yaml),
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch_size", 16),
        imgsz=train_cfg.get("img_size", 640),
        patience=train_cfg.get("patience", 20),
        lr0=train_cfg.get("lr0", 0.01),
        lrf=train_cfg.get("lrf", 0.01),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        box=train_cfg.get("box", 7.5),
        cls=train_cfg.get("cls", 0.5),
        dfl=train_cfg.get("dfl", 1.5),
        workers=train_cfg.get("workers", 8),
        seed=train_cfg.get("seed", 42),
        save_period=train_cfg.get("save_period", 10),
        val=train_cfg.get("val", True),
        plots=train_cfg.get("plots", True),
        project=str(runs_dir),
        name=experiment,
        exist_ok=True,
        resume=resume,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    last_weights = Path(results.save_dir) / "weights" / "last.pt"
    metrics_csv = exp_dir / "training_metrics.csv"
    _export_metrics_csv(Path(results.save_dir), metrics_csv)

    # Persist experiment meta
    meta = {
        "experiment": experiment,
        "model_variant": model_variant,
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "save_dir": str(results.save_dir),
        "metrics_csv": str(metrics_csv),
    }
    with open(exp_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Training complete. Best weights → %s", best_weights)
    return meta


def main():
    import click

    @click.command()
    @click.option("--model", "model_variant", default="yolov8m", show_default=True,
                  type=click.Choice(["yolov8n", "yolov8s", "yolov8m"]))
    @click.option("--config", default="config/config.yaml", show_default=True)
    @click.option("--resume", is_flag=True, default=False)
    @click.option("--resume-weights", default=None, type=click.Path())
    def cli(model_variant, config, resume, resume_weights):
        """Train YOLOv8 on the ice cream dataset."""
        result = train_model(model_variant, config, resume, resume_weights)
        print(json.dumps(result, indent=2))

    cli()


if __name__ == "__main__":
    main()
