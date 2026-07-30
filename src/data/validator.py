"""Dataset integrity validator.

Checks for missing labels, corrupt images, invalid bounding boxes,
and class imbalance. Outputs a JSON validation report.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _check_bbox(parts: list[str], num_classes: int) -> list[str]:
    """Validate a single YOLO annotation line. Returns list of error strings."""
    errors: list[str] = []
    if len(parts) < 5:
        errors.append("too few fields")
        return errors
    try:
        cid = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        errors.append("non-numeric fields")
        return errors
    if cid < 0 or cid >= num_classes:
        errors.append(f"invalid class id {cid} (num_classes={num_classes})")
    for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"{name}={val:.4f} out of [0,1]")
    if w <= 0 or h <= 0:
        errors.append("zero-area bbox")
    return errors


def validate_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    config_path: str | Path = "config/config.yaml",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Scan a dataset directory and report all integrity issues.

    Args:
        images_dir: Folder with image files.
        labels_dir: Folder with YOLO .txt label files.
        config_path: Path to config.yaml for class definitions.
        report_path: Optional JSON path to write the report.

    Returns:
        Validation report as a dict.
    """
    cfg = _load_config(config_path)
    class_names: list[str] = cfg["classes"]
    num_classes = len(class_names)
    imbalance_threshold = 0.1  # warn if any class < 10% of max class count

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}

    issues: dict[str, list[str]] = {}
    class_counts: Counter[int] = Counter()
    corrupt_count = 0
    missing_label_count = 0
    empty_label_count = 0

    for img_path in image_files:
        img_issues: list[str] = []

        # Corrupt image check
        img = cv2.imread(str(img_path))
        if img is None:
            img_issues.append("corrupt or unreadable image")
            corrupt_count += 1

        # Missing label
        lbl_path = labels_dir / img_path.with_suffix(".txt").name
        if img_path.stem not in label_files:
            img_issues.append("missing label file")
            missing_label_count += 1
        else:
            lbl_path = label_files[img_path.stem]
            lines = lbl_path.read_text().strip().splitlines()

            if not lines:
                img_issues.append("empty label file")
                empty_label_count += 1
            else:
                for ln_no, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    errs = _check_bbox(parts, num_classes)
                    if errs:
                        img_issues.append(f"line {ln_no}: {'; '.join(errs)}")
                    elif parts:
                        class_counts[int(parts[0])] += 1

        if img_issues:
            issues[img_path.name] = img_issues

    # Class imbalance analysis
    imbalance_warnings: list[str] = []
    if class_counts:
        max_count = max(class_counts.values())
        for cid, count in class_counts.items():
            ratio = count / max_count
            if ratio < imbalance_threshold:
                imbalance_warnings.append(
                    f"{class_names[cid]}: {count} samples ({ratio:.1%} of most common class)"
                )

    total = len(image_files)
    class_dist = {class_names[k]: v for k, v in sorted(class_counts.items())}

    report: dict[str, Any] = {
        "summary": {
            "total_images": total,
            "images_with_issues": len(issues),
            "corrupt_images": corrupt_count,
            "missing_labels": missing_label_count,
            "empty_labels": empty_label_count,
            "total_annotations": sum(class_counts.values()),
        },
        "class_distribution": class_dist,
        "imbalance_warnings": imbalance_warnings,
        "issues": issues,
    }

    if total > 0:
        pct_ok = (total - len(issues)) / total * 100
        log.info("Validation complete: %.1f%% of images OK (%d/%d)", pct_ok, total - len(issues), total)
    if imbalance_warnings:
        log.warning("Class imbalance detected:\n  " + "\n  ".join(imbalance_warnings))
    if issues:
        log.warning("%d image(s) have issues — see report for details", len(issues))

    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Validation report written → %s", report_path)

    return report


def main():
    import click

    @click.command()
    @click.option("--images-dir", required=True, type=click.Path(exists=True))
    @click.option("--labels-dir", required=True, type=click.Path(exists=True))
    @click.option("--config", default="config/config.yaml", show_default=True)
    @click.option("--report", default="reports/validation_report.json", show_default=True)
    def cli(images_dir, labels_dir, config, report):
        """Validate dataset integrity."""
        result = validate_dataset(images_dir, labels_dir, config, report)
        print(json.dumps(result["summary"], indent=2))

    cli()


if __name__ == "__main__":
    main()
