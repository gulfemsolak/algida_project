"""Tests for orientation.py — Katman 1 oryantasyon normalizasyonu."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.orientation import (
    estimate_orientation,
    rotate_image,
    _product_centers,
)


def _grid_dets():
    """4 dikey slot × 5 istiflenmiş ürün — dik (normalize) raf."""
    dets = []
    for cx in (100, 200, 300, 400):
        for cy in (50, 90, 130, 170, 210):
            dets.append({"category": "magnum_klasik", "bbox": [cx - 20, cy - 15, cx + 20, cy + 15]})
    return dets


def _centers(dets):
    return _product_centers(dets, "empty_slot")


def _rotate_pts(pts, a_deg):
    """cv2 nokta dönüşümü (merkez etrafında): x'=cos·x+sin·y, y'=-sin·x+cos·y."""
    c = pts.mean(0)
    p = pts - c
    r = np.radians(a_deg)
    cos, sin = np.cos(r), np.sin(r)
    x = cos * p[:, 0] + sin * p[:, 1]
    y = -sin * p[:, 0] + cos * p[:, 1]
    return np.c_[x, y] + c


def _dets_from_pts(pts):
    return [{"category": "magnum_klasik", "bbox": [x - 20, y - 15, x + 20, y + 15]} for x, y in pts]


def test_upright_needs_no_rotation():
    angle, conf = estimate_orientation(_grid_dets())
    assert abs(angle) < 2.0
    assert conf > 0.5


def test_recovers_known_tilts():
    """Bilinen açıyla eğ, tahmini uygula, tekrar tahmin et → ~0 (dikleşmeli)."""
    base = _centers(_grid_dets())
    for tilt in (15, 30, -25, 70, 90):
        tilted = _dets_from_pts(_rotate_pts(base, tilt))
        angle, conf = estimate_orientation(tilted)
        assert conf > 0.5
        corrected = _dets_from_pts(_rotate_pts(_centers(tilted), angle))
        angle2, _ = estimate_orientation(corrected)
        assert abs(angle2) < 3.0, f"tilt={tilt}: düzeltme sonrası {angle2}"


def test_returned_angle_in_range():
    base = _centers(_grid_dets())
    for tilt in (-89, -45, 0, 45, 89, 120):
        angle, _ = estimate_orientation(_dets_from_pts(_rotate_pts(base, tilt)))
        assert -90.0 < angle <= 90.0


def test_too_few_products_is_unreliable():
    angle, conf = estimate_orientation(_grid_dets()[:2])
    assert angle == 0.0
    assert conf == 0.0


def test_empty_slot_excluded_from_axis():
    dets = _grid_dets() + [{"category": "empty_slot", "bbox": [9000, 9000, 9040, 9030]}]
    # Uzaktaki empty_slot ekseni bozmamalı — açı hâlâ ~0.
    angle, conf = estimate_orientation(dets)
    assert abs(angle) < 2.0
    assert len(_centers(dets)) == 20


def test_rotate_image_expands_canvas():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = rotate_image(img, 90.0)
    # 90° döndürünce tuval en/boy takas edilerek büyür, içerik kırpılmaz.
    assert out.shape[0] >= 200
    assert out.shape[1] >= 100
