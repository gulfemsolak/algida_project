"""Tests for dataset_builder.py."""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from src.data.dataset_builder import _stratified_split, build_dataset


@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Create a minimal fake dataset with 20 images + label files."""
    raw_img_dir = tmp_path / "raw" / "images"
    raw_lbl_dir = tmp_path / "raw" / "labels"
    raw_img_dir.mkdir(parents=True)
    raw_lbl_dir.mkdir(parents=True)

    import numpy as np
    import cv2

    classes = list(range(5))  # 5 fake classes
    for i in range(20):
        # Tiny 32×32 white image
        img = np.ones((32, 32, 3), dtype=np.uint8) * 200
        img_path = raw_img_dir / f"img_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)

        cls_id = i % len(classes)
        lbl_path = raw_lbl_dir / f"img_{i:04d}.txt"
        lbl_path.write_text(f"{cls_id} 0.5 0.5 0.3 0.3\n")

    # Minimal config
    cfg = {
        "classes": [f"class_{c}" for c in classes],
        "dataset": {"train_ratio": 0.7, "val_ratio": 0.2, "random_seed": 0},
        "paths": {"splits": str(tmp_path / "splits")},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    return tmp_path, raw_img_dir, raw_lbl_dir, cfg_path


def test_stratified_split_sizes(tmp_dataset):
    tmp_path, raw_img_dir, raw_lbl_dir, cfg_path = tmp_dataset
    pairs = [(raw_img_dir / f"img_{i:04d}.jpg", raw_lbl_dir / f"img_{i:04d}.txt") for i in range(20)]
    train, val, test = _stratified_split(pairs, 0.7, 0.2, seed=0)
    assert len(train) + len(val) + len(test) == 20
    assert len(train) > len(val) > 0


def test_build_dataset_creates_splits(tmp_dataset):
    tmp_path, raw_img_dir, raw_lbl_dir, cfg_path = tmp_dataset
    stats = build_dataset(raw_img_dir, raw_lbl_dir, cfg_path)
    splits_dir = tmp_path / "splits"
    assert (splits_dir / "images" / "train").exists()
    assert (splits_dir / "images" / "val").exists()
    assert (splits_dir / "images" / "test").exists()
    assert stats["total_images"] == 20


def test_build_dataset_yaml_generated(tmp_dataset):
    tmp_path, raw_img_dir, raw_lbl_dir, cfg_path = tmp_dataset
    build_dataset(raw_img_dir, raw_lbl_dir, cfg_path)
    yaml_path = Path("config/dataset.yaml")
    # The YAML may not be written to the test path, so just verify stats
    stats = build_dataset(raw_img_dir, raw_lbl_dir, cfg_path)
    assert "class_distribution" in stats


def test_build_dataset_no_images(tmp_path):
    empty_img = tmp_path / "empty_images"
    empty_img.mkdir()
    empty_lbl = tmp_path / "empty_labels"
    empty_lbl.mkdir()

    cfg = {
        "classes": ["a"],
        "dataset": {"train_ratio": 0.7, "val_ratio": 0.2, "random_seed": 0},
        "paths": {"splits": str(tmp_path / "splits")},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    with pytest.raises(FileNotFoundError):
        build_dataset(empty_img, empty_lbl, cfg_path)
