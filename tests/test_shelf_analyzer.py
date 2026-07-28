"""Tests for shelf_analyzer.py."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.shelf_analyzer import _cluster_rows, analyze_shelf


def _make_prediction_result(detections, total_products=6):
    """predictor.predict_shelf() artık shelf_fill_rate/total_empty döndürmüyor
    (lattice concern — bkz. analyze_shelf). Fixture buna uygun."""
    summary = {}
    for d in detections:
        cat = d["category"]
        if cat not in summary:
            summary[cat] = {"count": 0, "avg_confidence": 0.9}
        summary[cat]["count"] += 1
    return {
        "image_path": "test.jpg",
        "detections": detections,
        "summary": summary,
        "total_products": total_products,
        "overall_confidence": 0.85,
        "demo_mode": True,
    }


def _slot_report(is_empty_flags):
    """Minimal, hazır slot_report — grid estimation'ı (gerçek görüntü gerektirir)
    bypass eder, yalnız analyze_shelf'in doluluk/restock mantığını izole test eder."""
    return {
        "grid_status": "ok",
        "slots": [
            {"is_empty": flag, "column_no": i + 1, "product_name": None if flag else "magnum_classic",
             "confidence": 0.0 if flag else 0.9, "x": i * 100.0}
            for i, flag in enumerate(is_empty_flags)
        ],
    }


def _fake_det(category, x1, y1, x2, y2, conf=0.9):
    return {"category": category, "confidence": conf, "bbox": [x1, y1, x2, y2]}


def test_cluster_rows_single_row():
    dets = [
        _fake_det("magnum_classic", 10, 100, 80, 160),
        _fake_det("cornetto_vanilla", 90, 105, 160, 165),
        _fake_det("empty_slot", 170, 95, 240, 155),
    ]
    rows = _cluster_rows(dets, eps=50)
    assert len(rows) == 1


def test_cluster_rows_two_rows():
    dets = [
        _fake_det("magnum_classic", 10, 50, 80, 110),   # row 0
        _fake_det("cornetto_vanilla", 90, 55, 160, 115),
        _fake_det("empty_slot", 10, 200, 80, 260),       # row 1
        _fake_det("popsicle_fruit", 90, 205, 160, 265),
    ]
    rows = _cluster_rows(dets, eps=30)
    assert len(rows) == 2


def test_cluster_rows_empty():
    assert _cluster_rows([]) == {}


def test_analyze_shelf_restock_flag():
    # Ham empty_slot dedeksiyonu artık YOK — doluluk slot_report'tan (lattice)
    # gelir. 4 slot, hepsi is_empty=True → fill_rate=0.0, restock tetiklenir.
    pred = _make_prediction_result([], total_products=0)
    slot_report = _slot_report([True, True, True, True])
    import tempfile, os
    cfg = {
        "shelf_analysis": {
            "restock_threshold": 0.5,
            "row_cluster_eps": 50,
            "low_confidence_threshold": 0.4,
            "empty_slot_restock_count": 3,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    try:
        report = analyze_shelf(pred, config_path=tmp_cfg, slot_report=slot_report)
        assert report["restock_needed"] is True
        assert report["total_empty"] == 4
        assert report["shelf_fill_rate"] == 0.0
    finally:
        os.unlink(tmp_cfg)


def test_analyze_shelf_no_restock():
    dets = [_fake_det("magnum_classic", i * 60, 50, i * 60 + 50, 110) for i in range(5)]
    pred = _make_prediction_result(dets, total_products=5)
    slot_report = _slot_report([False] * 5)
    import tempfile, os
    cfg = {
        "shelf_analysis": {
            "restock_threshold": 0.5,
            "row_cluster_eps": 50,
            "low_confidence_threshold": 0.4,
            "empty_slot_restock_count": 3,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    try:
        report = analyze_shelf(pred, config_path=tmp_cfg, slot_report=slot_report)
        assert report["restock_needed"] is False
        assert report["shelf_fill_rate"] == 1.0
    finally:
        os.unlink(tmp_cfg)


def test_analyze_shelf_lattice_unavailable_no_alarm():
    """Izgara kurulamazsa (belirsiz) kanıtsız restock alarmı ÜRETİLMEMELİ."""
    pred = _make_prediction_result([], total_products=0)
    slot_report = {"grid_status": "belirsiz", "slots": []}
    import tempfile, os
    cfg = {
        "shelf_analysis": {
            "restock_threshold": 0.5,
            "row_cluster_eps": 50,
            "low_confidence_threshold": 0.4,
            "empty_slot_restock_count": 3,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    try:
        report = analyze_shelf(pred, config_path=tmp_cfg, slot_report=slot_report)
        assert report["restock_needed"] is False
        assert report["lattice_available"] is False
    finally:
        os.unlink(tmp_cfg)


def test_analyze_shelf_anomaly_detection():
    dets = [_fake_det("magnum_classic", 0, 0, 50, 50, conf=0.2)]  # below threshold
    pred = _make_prediction_result(dets, total_products=1)
    slot_report = _slot_report([False])
    import tempfile, os
    cfg = {
        "shelf_analysis": {
            "restock_threshold": 0.5,
            "row_cluster_eps": 50,
            "low_confidence_threshold": 0.4,
            "empty_slot_restock_count": 3,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name
    try:
        report = analyze_shelf(pred, config_path=tmp_cfg, slot_report=slot_report)
        assert len(report["anomalies"]) == 1
        assert report["anomalies"][0]["type"] == "low_confidence"
    finally:
        os.unlink(tmp_cfg)
