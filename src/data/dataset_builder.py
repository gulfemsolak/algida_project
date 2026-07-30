"""Converts raw images + YOLO annotations into train/val/test splits.

Performs stratified splitting based on the dominant class of each image
so that class distributions are approximately preserved across splits.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _image_dominant_class(label_path: Path) -> int | None:
    """Return the most frequent class id in a label file, or None if empty."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return None
    counts: Counter[int] = Counter()
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return counts.most_common(1)[0][0] if counts else None


def _stratified_split(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split (image, label) pairs into train/val/test with stratification."""
    rng = random.Random(seed)
    by_class: dict[int | None, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        dominant = _image_dominant_class(lbl)
        by_class[dominant].append((img, lbl))

    train, val, test = [], [], []
    for _, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))
        train.extend(group[:n_train])
        val.extend(group[n_train: n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _copy_pairs(
    pairs: list[tuple[Path, Path]],
    images_dest: Path,
    labels_dest: Path,
) -> None:
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy(img, images_dest / img.name)
        if lbl.exists():
            shutil.copy(lbl, labels_dest / lbl.name)
        else:
            (labels_dest / lbl.name).touch()


def _generate_dataset_yaml(
    splits_dir: Path,
    class_names: list[str],
    output_path: Path,
) -> None:
    cfg = {
        "path": str(splits_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("dataset.yaml written → %s", output_path)


def build_dataset(
    raw_images_dir: str | Path,
    raw_labels_dir: str | Path,
    config_path: str | Path = "config/config.yaml",
) -> dict[str, Any]:
    """Build YOLO-compatible train/val/test splits from raw data.

    Args:
        raw_images_dir: Folder containing original images.
        raw_labels_dir: Folder containing matching YOLO .txt label files.
        config_path: Path to config.yaml.

    Returns:
        Statistics dict.
    """
    cfg = _load_config(config_path)
    ds_cfg = cfg.get("dataset", {})
    train_ratio = ds_cfg.get("train_ratio", 0.70)
    val_ratio = ds_cfg.get("val_ratio", 0.20)
    seed = ds_cfg.get("random_seed", 42)
    class_names: list[str] = cfg["classes"]
    splits_dir = Path(cfg["paths"]["splits"])
    dataset_yaml_path = Path("config/dataset.yaml")

    raw_images_dir = Path(raw_images_dir)
    raw_labels_dir = Path(raw_labels_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in raw_images_dir.iterdir() if p.suffix.lower() in image_exts)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_images_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels = 0
    for img_path in image_paths:
        lbl_path = raw_labels_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            log.warning("Missing label for %s", img_path.name)
            missing_labels += 1
        pairs.append((img_path, lbl_path))

    log.info("Found %d images (%d missing labels)", len(pairs), missing_labels)

    train_pairs, val_pairs, test_pairs = _stratified_split(pairs, train_ratio, val_ratio, seed)

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        _copy_pairs(
            split_pairs,
            splits_dir / "images" / split_name,
            splits_dir / "labels" / split_name,
        )
        log.info("%-5s: %d images", split_name.upper(), len(split_pairs))

    _generate_dataset_yaml(splits_dir, class_names, dataset_yaml_path)

    # Class distribution across splits
    all_class_counts: Counter[int] = Counter()
    for _, lbl in pairs:
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    all_class_counts[int(parts[0])] += 1

    class_dist = {class_names[k]: v for k, v in sorted(all_class_counts.items())}

    stats = {
        "total_images": len(pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
        "missing_labels": missing_labels,
        "class_distribution": class_dist,
    }
    log.info("Dataset statistics:\n%s", json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--raw-images", required=True, type=click.Path(exists=True))
    @click.option("--raw-labels", required=True, type=click.Path(exists=True))
    @click.option("--config", default="config/config.yaml", show_default=True)
    def cli(raw_images, raw_labels, config):
        """Build train/val/test dataset splits."""
        stats = build_dataset(raw_images, raw_labels, config)
        print(json.dumps(stats, indent=2))

    cli()
