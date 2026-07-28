"""Tests for skew_corrector.py — EXIF-only kanonik döndürme (içerik tabanlı tahmin YOK).

İçerik tabanlı ``estimate_coarse_orientation`` kaldırıldı: tepsi bloğunun kadrajdaki
en-boy oranı, fotoğrafın gerçek yönelimini güvenilir yansıtmıyordu (somut vaka:
WhatsApp Image 2026-07-17 at 12.03.36.jpeg — gerçekten yatay çekilmiş düz bir fotoğraf,
tepsi bloğu kadrajda 0.83 en-boy verdiği için yanlışlıkla 90° döndürülüyordu). Yönelim
artık YALNIZ EXIF Orientation etiketinden gelir; yoksa döndürme yapılmaz.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.skew_corrector import exif_coarse_angle, rotate_coarse, deskew


def test_exif_coarse_angle_mapping():
    assert exif_coarse_angle(1) == 0
    assert exif_coarse_angle(3) == 180
    assert exif_coarse_angle(6) == 90
    assert exif_coarse_angle(8) == 270


def test_exif_coarse_angle_none_is_zero():
    assert exif_coarse_angle(None) == 0


def test_exif_coarse_angle_mirror_ignored():
    # 2/4/5/7 ayna (flip) gerektirir, döndürme değildir — içerikten tahmin YOK, 0.
    for tag in (2, 4, 5, 7):
        assert exif_coarse_angle(tag) == 0


def test_rotate_coarse_lossless_shapes():
    img = np.zeros((300, 500, 3), np.uint8)
    assert rotate_coarse(img, 0).shape == (300, 500, 3)
    assert rotate_coarse(img, 90).shape == (500, 300, 3)
    assert rotate_coarse(img, 180).shape == (300, 500, 3)
    assert rotate_coarse(img, 270).shape == (500, 300, 3)


def test_deskew_no_exif_is_untouched():
    img = np.zeros((300, 500, 3), np.uint8)
    out = deskew(img, exif_orientation=None)
    assert out["coarse"] == 0
    assert out["angle"] == 0.0
    assert out["image"].shape == (300, 500, 3)


def test_deskew_landscape_photo_never_rotated_by_content():
    # REGRESYON: gerçekten yatay çekilmiş düz bir fotoğraf, içerik ne olursa olsun
    # EXIF yoksa döndürülmemeli (önceki hatanın tam kendisi).
    img = np.full((1148, 2040, 3), 60, np.uint8)
    img[900:1100, 500:1500] = (30, 30, 30)  # tepsi benzeri koyu leke, kadrajda dar/uzun
    out = deskew(img, exif_orientation=None)
    assert out["coarse"] == 0
    dh, dw = out["image"].shape[:2]
    assert dw > dh  # yatay kalmalı


def test_deskew_exif_rotates_90():
    img = np.zeros((300, 500, 3), np.uint8)
    out = deskew(img, exif_orientation=6)
    assert out["coarse"] == 90
    assert out["angle"] == 90.0
    dh, dw = out["image"].shape[:2]
    assert (dh, dw) == (500, 300)


def test_deskew_exif_rotates_180():
    img = np.zeros((300, 500, 3), np.uint8)
    out = deskew(img, exif_orientation=3)
    assert out["coarse"] == 180
    assert out["image"].shape == (300, 500, 3)
