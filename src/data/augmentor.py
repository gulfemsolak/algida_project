"""Albumentations-based augmentation pipeline for vending machine imagery.

Handles YOLO-format bounding boxes throughout so that augmented labels
remain perfectly aligned with augmented images.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict[str, Any]) -> A.Compose:
    """Construct the augmentation pipeline from config values.

    The pipeline is designed for vending machine conditions:
    - brightness/contrast → different machine lighting
    - slight rotation      → camera angle variance
    - Gaussian blur/noise  → low-quality cameras
    - random shadow        → glass reflections on vending machine door
    - hue/sat/val shift    → temperature-affected packaging
    - horizontal flip      → mirroring
    """
    aug = cfg.get("augmentation", {})
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.3),
                contrast_limit=aug.get("contrast_limit", 0.3),
                p=0.8,
            ),
            A.Rotate(
                limit=aug.get("rotation_limit", 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.GaussianBlur(
                blur_limit=(3, aug.get("blur_limit", 5)),
                p=0.3,
            ),
            A.GaussNoise(
                var_limit=tuple(aug.get("noise_var_limit", [10, 50])),
                p=0.3,
            ),
            A.RandomShadow(
                num_shadows_lower=aug.get("shadow_num_shadows_lower", 1),
                num_shadows_upper=aug.get("shadow_num_shadows_upper", 2),
                shadow_dimension=5,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=aug.get("hue_shift_limit", 20),
                sat_shift_limit=aug.get("sat_shift_limit", 30),
                val_shift_limit=aug.get("val_shift_limit", 20),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
        ),
    )


def _read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """Parse a YOLO label file → (class_ids, bboxes_xywh_norm)."""
    class_ids: list[int] = []
    bboxes: list[list[float]] = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return class_ids, bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            class_ids.append(cid)
            bboxes.append([cx, cy, w, h])
    return class_ids, bboxes


def _write_yolo_labels(label_path: Path, class_ids: list[int], bboxes: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cid, bbox in zip(class_ids, bboxes):
            f.write(f"{cid} {' '.join(f'{v:.6f}' for v in bbox)}\n")


def augment_dataset(
    raw_images_dir: str | Path,
    raw_labels_dir: str | Path,
    output_images_dir: str | Path,
    output_labels_dir: str | Path,
    config_path: str | Path = "config/config.yaml",
    copy_originals: bool = True,
) -> dict[str, int]:
    """Augment every image in *raw_images_dir* and write results.

    Args:
        raw_images_dir: Folder with original .jpg/.png images.
        raw_labels_dir: Folder with corresponding YOLO .txt label files.
        output_images_dir: Destination for augmented images.
        output_labels_dir: Destination for augmented label files.
        config_path: Path to config.yaml.
        copy_originals: If True, also copy originals to the output folder.

    Returns:
        Dict with ``{"total_augmented": N, "total_skipped": M}``.
    """
    cfg = _load_config(config_path)
    n_aug = cfg.get("augmentation", {}).get("num_augmented_per_image", 5)
    pipeline = build_pipeline(cfg)

    raw_images_dir = Path(raw_images_dir)
    raw_labels_dir = Path(raw_labels_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in raw_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    total_augmented = 0
    total_skipped = 0

    for img_path in image_paths:
        label_path = raw_labels_dir / img_path.with_suffix(".txt").name
        class_ids, bboxes = _read_yolo_labels(label_path)

        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("Cannot read image %s — skipping", img_path)
            total_skipped += 1
            continue

        if copy_originals:
            shutil.copy(img_path, output_images_dir / img_path.name)
            shutil.copy(label_path, output_labels_dir / label_path.name) if label_path.exists() else None

        for i in range(n_aug):
            try:
                result = pipeline(image=image, bboxes=bboxes, class_ids=class_ids)
            except Exception as exc:
                log.warning("Augmentation failed for %s (pass %d): %s", img_path.name, i, exc)
                continue

            stem = f"{img_path.stem}_aug{i:03d}"
            out_img_path = output_images_dir / f"{stem}.jpg"
            out_lbl_path = output_labels_dir / f"{stem}.txt"

            cv2.imwrite(str(out_img_path), result["image"])
            _write_yolo_labels(out_lbl_path, result["class_ids"], result["bboxes"])
            total_augmented += 1

        log.debug("Augmented %s → %d versions", img_path.name, n_aug)

    log.info(
        "Augmentation complete. Produced %d images, skipped %d.",
        total_augmented, total_skipped,
    )
    return {"total_augmented": total_augmented, "total_skipped": total_skipped}


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--raw-images", required=True, type=click.Path(exists=True))
    @click.option("--raw-labels", required=True, type=click.Path(exists=True))
    @click.option("--out-images", required=True, type=click.Path())
    @click.option("--out-labels", required=True, type=click.Path())
    @click.option("--config", default="config/config.yaml", show_default=True)
    def cli(raw_images, raw_labels, out_images, out_labels, config):
        """Run the augmentation pipeline."""
        stats = augment_dataset(raw_images, raw_labels, out_images, out_labels, config)
        print(json.dumps(stats, indent=2))

    cli()
