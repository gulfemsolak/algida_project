"""Custom metrics for shelf detection evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def counting_accuracy(
    predictions: dict[str, int],
    ground_truth: dict[str, int],
) -> float:
    """Fraction of categories where the predicted count exactly matches ground truth.

    Args:
        predictions: Dict mapping class_name → predicted count.
        ground_truth: Dict mapping class_name → true count.

    Returns:
        Float in [0, 1]; 1.0 means all counts are correct.
    """
    all_classes = set(predictions) | set(ground_truth)
    if not all_classes:
        return 1.0
    correct = sum(
        1 for c in all_classes
        if predictions.get(c, 0) == ground_truth.get(c, 0)
    )
    return correct / len(all_classes)


def mean_absolute_count_error(
    predictions: dict[str, int],
    ground_truth: dict[str, int],
) -> float:
    """Average absolute difference between predicted and true count per category.

    Args:
        predictions: Dict mapping class_name → predicted count.
        ground_truth: Dict mapping class_name → true count.

    Returns:
        Non-negative float; 0.0 means perfect counting.
    """
    all_classes = set(predictions) | set(ground_truth)
    if not all_classes:
        return 0.0
    return float(
        sum(abs(predictions.get(c, 0) - ground_truth.get(c, 0)) for c in all_classes)
        / len(all_classes)
    )


def shelf_fill_rate(detections: list[dict[str, Any]], total_slots: int) -> float:
    """Fraction of shelf slots occupied by products (non-empty detections).

    Args:
        detections: List of detection dicts with a ``category`` key.
        total_slots: Total number of shelf slots (products + empty).

    Returns:
        Float in [0, 1].
    """
    if total_slots <= 0:
        return 0.0
    product_count = sum(1 for d in detections if d.get("category") != "empty_slot")
    return product_count / total_slots


def per_category_accuracy(
    predictions_list: list[dict[str, int]],
    ground_truth_list: list[dict[str, int]],
) -> dict[str, float]:
    """Per-category count accuracy across multiple images.

    Args:
        predictions_list: List of per-image prediction count dicts.
        ground_truth_list: List of per-image ground truth count dicts.

    Returns:
        Dict mapping class_name → fraction of images where count was correct.
    """
    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)

    for preds, gt in zip(predictions_list, ground_truth_list):
        all_classes = set(preds) | set(gt)
        for c in all_classes:
            class_total[c] += 1
            if preds.get(c, 0) == gt.get(c, 0):
                class_correct[c] += 1

    return {
        c: round(class_correct[c] / class_total[c], 4)
        for c in class_total
    }


def detection_accuracy_at_thresholds(
    predictions_list: list[list[dict[str, Any]]],
    ground_truth_list: list[dict[str, int]],
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Counting accuracy at different confidence thresholds.

    Args:
        predictions_list: List of per-image detection dicts (with ``confidence`` and ``category``).
        ground_truth_list: List of per-image ground truth count dicts.
        thresholds: Confidence thresholds to evaluate; default [0.3, 0.5, 0.7, 0.9].

    Returns:
        Dict mapping threshold string → mean counting accuracy.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]

    results: dict[str, float] = {}
    for thresh in thresholds:
        accs: list[float] = []
        for dets, gt in zip(predictions_list, ground_truth_list):
            filtered = [d for d in dets if d.get("confidence", 0) >= thresh]
            pred_counts: dict[str, int] = Counter(d["category"] for d in filtered)
            accs.append(counting_accuracy(dict(pred_counts), gt))
        results[str(thresh)] = round(float(np.mean(accs)) if accs else 0.0, 4)

    return results


def overall_metrics_summary(
    predictions_list: list[list[dict[str, Any]]],
    ground_truth_list: list[dict[str, int]],
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute all custom metrics in one call.

    Args:
        predictions_list: Per-image detection dicts.
        ground_truth_list: Per-image ground truth count dicts.
        conf_threshold: Confidence filter for counting.

    Returns:
        Dict with counting accuracy, MACE, per-category accuracy,
        and accuracy-at-thresholds.
    """
    filtered_preds = [
        {d["category"]: 1 for d in dets if d.get("confidence", 0) >= conf_threshold}
        for dets in predictions_list
    ]
    # Build per-image count dicts properly
    pred_counts_list = [
        dict(Counter(d["category"] for d in dets if d.get("confidence", 0) >= conf_threshold))
        for dets in predictions_list
    ]

    ca_values = [counting_accuracy(p, g) for p, g in zip(pred_counts_list, ground_truth_list)]
    mace_values = [mean_absolute_count_error(p, g) for p, g in zip(pred_counts_list, ground_truth_list)]

    return {
        "mean_counting_accuracy": round(float(np.mean(ca_values)), 4),
        "mean_absolute_count_error": round(float(np.mean(mace_values)), 4),
        "per_category_accuracy": per_category_accuracy(pred_counts_list, ground_truth_list),
        "accuracy_at_thresholds": detection_accuracy_at_thresholds(
            predictions_list, ground_truth_list
        ),
    }
