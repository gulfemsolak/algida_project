"""Tests for predictor.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import predict_shelf


@pytest.fixture()
def blank_image() -> np.ndarray:
    """Return a small blank BGR image."""
    import numpy as np
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_predict_shelf_demo_mode_returns_dict(blank_image):
    result = predict_shelf(blank_image, model_path=None, conf_threshold=0.5)
    assert isinstance(result, dict)


def test_predict_shelf_required_keys(blank_image):
    """shelf_fill_rate/total_empty artık burada YOK — lattice concern, bkz.
    src.analysis.shelf_analyzer.analyze_shelf."""
    result = predict_shelf(blank_image)
    required_keys = {
        "image_path", "detections", "summary",
        "total_products", "overall_confidence", "annotated_image",
    }
    assert required_keys.issubset(result.keys())
    assert "shelf_fill_rate" not in result
    assert "total_empty" not in result


def test_predict_shelf_confidence_range(blank_image):
    result = predict_shelf(blank_image)
    for det in result["detections"]:
        assert 0.0 <= det["confidence"] <= 1.0


def test_predict_shelf_detection_structure(blank_image):
    result = predict_shelf(blank_image)
    for det in result["detections"]:
        assert "category" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4


def test_predict_shelf_annotated_image_shape(blank_image):
    result = predict_shelf(blank_image)
    ann = result["annotated_image"]
    assert ann is not None
    assert ann.shape == blank_image.shape


def test_predict_shelf_summary_matches_detections(blank_image):
    result = predict_shelf(blank_image, conf_threshold=0.0)
    total_from_summary = sum(v["count"] for v in result["summary"].values())
    assert total_from_summary == len(result["detections"])


def test_predict_shelf_missing_image_raises():
    with pytest.raises(FileNotFoundError):
        predict_shelf("/nonexistent/path/image.jpg")


def test_predict_shelf_missing_model_raises_not_demo(blank_image):
    """A requested-but-missing model must raise, never silently run demo mode.

    Blocker B: an operator supplying weights should never be shown fabricated
    detections when those weights can't be found."""
    with pytest.raises(FileNotFoundError):
        predict_shelf(blank_image, model_path="/nonexistent/best.pt", conf_threshold=0.5)


def test_predict_shelf_none_model_runs_demo(blank_image):
    """model_path=None is the ONLY path to demo detections."""
    result = predict_shelf(blank_image, model_path=None, conf_threshold=0.5)
    assert result["demo_mode"] is True


def test_run_detection_never_returns_empty_slot():
    """empty_slot recall is ~0.12-0.21 (unreliable) — filtered at inference via
    model.predict(classes=...), never surfaced to detections/summary. Doluluk
    artık slot_assigner'ın lattice'inden geliyor (bkz. shelf_analyzer.py)."""
    import cv2
    from src.models.predictor import load_model, run_detection

    model = load_model("models/best.pt")
    img = cv2.imread("data/real_dataset/train/images/108_jpeg.rf.cYfFENe7a2jnjOaCwlK6.jpeg")
    assert img is not None
    dets, class_names = run_detection(model, img, conf_threshold=0.1)
    assert "empty_slot" in class_names  # model hâlâ 35 class biliyor
    assert not any(d["category"] == "empty_slot" for d in dets)
