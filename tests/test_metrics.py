"""Tests for analysis/metrics.py — verified with known inputs/outputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.metrics import (
    counting_accuracy,
    detection_accuracy_at_thresholds,
    mean_absolute_count_error,
    overall_metrics_summary,
    per_category_accuracy,
    shelf_fill_rate,
)


# ── counting_accuracy ────────────────────────────────────────────────────────

def test_counting_accuracy_perfect():
    pred = {"magnum_classic": 3, "empty_slot": 2}
    gt = {"magnum_classic": 3, "empty_slot": 2}
    assert counting_accuracy(pred, gt) == 1.0


def test_counting_accuracy_all_wrong():
    pred = {"magnum_classic": 1}
    gt = {"magnum_classic": 5}
    assert counting_accuracy(pred, gt) == 0.0


def test_counting_accuracy_partial():
    pred = {"magnum_classic": 3, "cornetto_vanilla": 1}
    gt = {"magnum_classic": 3, "cornetto_vanilla": 2}
    assert counting_accuracy(pred, gt) == 0.5


def test_counting_accuracy_empty_dicts():
    assert counting_accuracy({}, {}) == 1.0


def test_counting_accuracy_extra_predicted_class():
    pred = {"magnum_classic": 3, "ghost_class": 1}
    gt = {"magnum_classic": 3}
    # ghost_class predicted=1, gt=0 → wrong; magnum_classic correct
    assert counting_accuracy(pred, gt) == 0.5


# ── mean_absolute_count_error ─────────────────────────────────────────────────

def test_mace_perfect():
    pred = {"a": 3, "b": 2}
    gt = {"a": 3, "b": 2}
    assert mean_absolute_count_error(pred, gt) == 0.0


def test_mace_known_value():
    pred = {"a": 5, "b": 1}
    gt = {"a": 3, "b": 3}
    # |5-3| + |1-3| = 2 + 2 = 4 / 2 classes = 2.0
    assert mean_absolute_count_error(pred, gt) == 2.0


def test_mace_empty():
    assert mean_absolute_count_error({}, {}) == 0.0


# ── shelf_fill_rate ───────────────────────────────────────────────────────────

def test_fill_rate_all_products():
    dets = [{"category": "magnum_classic"} for _ in range(5)]
    assert shelf_fill_rate(dets, total_slots=5) == 1.0


def test_fill_rate_half_empty():
    dets = [
        {"category": "magnum_classic"},
        {"category": "magnum_classic"},
        {"category": "empty_slot"},
        {"category": "empty_slot"},
    ]
    assert shelf_fill_rate(dets, total_slots=4) == 0.5


def test_fill_rate_zero_slots():
    assert shelf_fill_rate([], total_slots=0) == 0.0


# ── per_category_accuracy ─────────────────────────────────────────────────────

def test_per_category_accuracy_perfect():
    preds = [{"a": 2, "b": 1}]
    gts = [{"a": 2, "b": 1}]
    result = per_category_accuracy(preds, gts)
    assert result["a"] == 1.0
    assert result["b"] == 1.0


def test_per_category_accuracy_mixed():
    preds = [{"a": 2}, {"a": 1}]
    gts = [{"a": 2}, {"a": 3}]
    result = per_category_accuracy(preds, gts)
    # First image: a correct; second: a wrong → 0.5
    assert result["a"] == 0.5


# ── detection_accuracy_at_thresholds ─────────────────────────────────────────

def test_accuracy_at_thresholds_returns_all_keys():
    dets = [[{"category": "a", "confidence": 0.9}]]
    gts = [{"a": 1}]
    result = detection_accuracy_at_thresholds(dets, gts, thresholds=[0.3, 0.5, 0.9])
    assert set(result.keys()) == {"0.3", "0.5", "0.9"}


def test_accuracy_at_high_threshold_loses_detections():
    dets = [[{"category": "a", "confidence": 0.4}]]
    gts = [{"a": 1}]
    low = detection_accuracy_at_thresholds(dets, gts, thresholds=[0.3])["0.3"]
    high = detection_accuracy_at_thresholds(dets, gts, thresholds=[0.9])["0.9"]
    # At 0.3 threshold the detection passes → count=1=gt, acc=1.0
    # At 0.9 threshold the detection is filtered → count=0≠gt=1, acc=0.0
    assert low == 1.0
    assert high == 0.0
