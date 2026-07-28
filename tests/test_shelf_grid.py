"""Tests for shelf_grid.py — homografi tabanlı ızgara (tepsi 4 köşesi → rektifiye uzay)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.analysis.shelf_grid import (
    estimate_grid,
    detect_tray_corners,
    slot_index_for,
    shelf_number_from_tray_bar,
)


# ── slot_index_for (H tabanlı, rektifiye uzay) ──────────────────────────────────
def _grid(column_count=7, pitch=100.0, roi_left=50.0, rect_height=1000.0):
    """Test için sabit homografili ızgara — H basit x-öteleme (dönüşsüz, ölçeksiz).

    ``roi_left=50, pitch=100`` ile eski (origin=100, pitch=100) testleriyle aynı
    x_rect değerlerini üretir: x_rect = x_orig - roi_left.
    """
    h_mat = [[1.0, 0.0, -roi_left], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    h_inv = [[1.0, 0.0, roi_left], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    boundaries = [i * pitch for i in range(column_count + 1)]
    centers = [pitch / 2.0 + i * pitch for i in range(column_count)]
    return {
        "grid_status": "ok", "column_count": column_count, "pitch": pitch,
        "rect_width": pitch * column_count, "rect_height": rect_height,
        "centers": centers, "boundaries": boundaries,
        "corners": None, "H": h_mat, "H_inv": h_inv,
        "grid_source": "kontur", "shelf_number_from_bar": None, "confident": True,
    }


def test_slot_index_pitch_based():
    g = _grid()
    assert slot_index_for(100, 100, g) == 0
    assert slot_index_for(200, 100, g) == 1
    assert slot_index_for(640, 100, g) == 5
    assert slot_index_for(-500, 100, g) == 0     # kırpma, düşmez
    assert slot_index_for(99999, 100, g) == 6


# ── Belirsiz: çapa yok → atama yapma ────────────────────────────────────────────
def test_blank_image_is_uncertain():
    img = np.full((400, 700, 3), 20, np.uint8)  # düz koyu, kontur yok (kadraj = tek bileşen)
    grid = estimate_grid(img, column_count=7)
    assert grid["grid_status"] == "belirsiz"
    assert grid["H"] is None


def test_no_image_is_uncertain():
    grid = estimate_grid(None, column_count=7)
    assert grid["grid_status"] == "belirsiz"


# ── Manuel köşe override → kesin ızgara ──────────────────────────────────────────
def test_corners_override_builds_grid():
    corners = [(100, 0), (800, 0), (800, 500), (100, 500)]
    grid = estimate_grid(None, column_count=7, corners_override=corners)
    assert grid["grid_status"] == "ok"
    assert grid["grid_source"] == "manuel"
    assert abs(grid["pitch"] - 100) < 1e-6       # (800-100)/7


def test_corners_override_order_independent():
    # Köşeler karışık sırada verilse bile _order_corners TL/TR/BR/BL'ye oturtur.
    corners = [(800, 500), (100, 0), (100, 500), (800, 0)]
    grid = estimate_grid(None, column_count=7, corners_override=corners)
    assert grid["grid_status"] == "ok"
    assert abs(grid["pitch"] - 100) < 1e-6


# ── Tepsi konturu (koyu maske → 4 köşe) ─────────────────────────────────────────
def test_detect_tray_corners_dark_rectangle():
    img = np.full((400, 800, 3), 200, np.uint8)
    img[120:340, 150:650] = (30, 30, 30)
    corners = detect_tray_corners(img)
    assert corners is not None
    xs = corners[:, 0]
    assert abs(xs.min() - 150) < 40 and abs(xs.max() - 650) < 40


def test_estimate_grid_from_contour():
    img = np.full((400, 800, 3), 200, np.uint8)
    img[120:340, 150:650] = (30, 30, 30)
    grid = estimate_grid(img, column_count=7)
    assert grid["grid_status"] == "ok"
    assert grid["grid_source"] == "kontur"
    assert grid["column_count"] == 7
    assert abs(grid["pitch"] - 500 / 7) < 15


def test_tray_touching_frame_edge_is_uncertain():
    # Tepsi kadraj kenarına yapışıksa (kesilmiş görünüm) köşeler reddedilir.
    img = np.full((400, 800, 3), 200, np.uint8)
    img[0:340, 0:650] = (30, 30, 30)  # üst-sol kenara yapışık
    grid = estimate_grid(img, column_count=7)
    assert grid["grid_status"] == "belirsiz"


def test_tray_too_small_is_uncertain():
    img = np.full((400, 800, 3), 200, np.uint8)
    img[180:220, 380:420] = (30, 30, 30)  # çok küçük koyu leke
    grid = estimate_grid(img, column_count=7)
    assert grid["grid_status"] == "belirsiz"


# ── Raf numarası: tepsi barından OCR (pytesseract yoksa None — çökme yok) ──────
def test_shelf_number_from_tray_bar_no_crash():
    img = np.full((400, 800, 3), 200, np.uint8)
    img[120:340, 150:650] = (30, 30, 30)
    corners = detect_tray_corners(img)
    assert corners is not None
    result = shelf_number_from_tray_bar(img, corners)
    assert result is None or isinstance(result, int)
