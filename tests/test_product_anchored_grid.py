"""Tests for product_anchored_grid.py — faz-hizalı LATİS modeli.

Lane kimlikten DEĞİL, eşit-pitch fiziksel kanaldan kurulur: tek açı θ + pitch (ürün
eni & kimlik-geçişi) + faz-hizalı latis. Aynı ürün iki kanala yayılırsa iki slot;
sınırlar ürünlerin arasına düşer (merkezden geçmez); ürünsüz kanal = boş slot.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.product_anchored_grid import (
    build_chains,
    dominant_axis,
    estimate_pitch,
    build_lattice,
    _phase_offset,
    chain_slot_index_for,
    estimate_grid_lane_based,
)


def _det(category, cx, cy=100, w=30, h=50):
    return {"category": category, "confidence": 0.9, "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


def _stack(category, cx, n=4, cy0=40, dy=50, shear=0.0, w=30, h=50):
    """Aynı üründen dikey lane; shear!=0 → her adımda cx de shear*dy kadar kayar."""
    dets = []
    for i in range(n):
        cy = cy0 + i * dy
        dets.append(_det(category, cx + shear * (cy - cy0), cy=cy, w=w, h=h))
    return dets


def _hstack(category, cy, n=4, cx0=40, dx=50, w=30, h=50):
    """90°-dönük fotoğraf simülasyonu: aynı üründen YATAY lane."""
    return [_det(category, cx0 + i * dx, cy=cy, w=w, h=h) for i in range(n)]


# ── build_chains (yalnız θ kestirimi için — sınır KURMAZ) ────────────────────────
def test_build_chains_separates_distinct_columns():
    dets = _stack("magnum_klasik", 100, n=4) + _stack("cornetto_cilek", 300, n=4) + _stack("magnum_beyaz", 500, n=4)
    chains = build_chains(dets)
    assert len(chains) == 3
    assert {c["category"] for c in chains} == {"magnum_klasik", "cornetto_cilek", "magnum_beyaz"}


def test_build_chains_singleton_valid():
    dets = [_det("magnum_klasik", 100)]
    chains = build_chains(dets)
    assert len(chains) == 1
    assert len(chains[0]["members"]) == 1


# ── dominant_axis ────────────────────────────────────────────────────────────────
def test_dominant_axis_vertical_for_vertical_lanes():
    dets = _stack("magnum_klasik", 100, n=5) + _stack("cornetto_cilek", 300, n=5) + _stack("magnum_beyaz", 500, n=5)
    chains = build_chains(dets)
    v, u, angle_deg = dominant_axis(chains)
    assert abs(abs(angle_deg) - 90) < 5   # v ~ dikey


def test_dominant_axis_horizontal_for_rotated_photo():
    dets = _hstack("magnum_klasik", 100, n=5) + _hstack("cornetto_cilek", 300, n=5) + _hstack("magnum_beyaz", 500, n=5)
    chains = build_chains(dets)
    v, u, angle_deg = dominant_axis(chains)
    assert abs(angle_deg) < 5   # v ~ yatay (0°) — yön-agnostik kanıtı


def test_dominant_axis_no_bias_without_data():
    v, u, angle_deg = dominant_axis([])
    assert angle_deg == 90.0  # son çare varsayılan: dikey


# ── estimate_pitch — Kaynak A (ürün eni) + Kaynak B (kimlik geçişi), B tercihli ──
def test_estimate_pitch_from_product_width():
    # Beş farklı ürün, eni=100, aralık=100 (paketli) → pitch ~100.
    dets = [_det(cat, 100 + i * 100, w=100)
            for i, cat in enumerate(["a", "b", "c", "d", "e"])]
    pitch, pitch_A, pitch_B, unrel = estimate_pitch(dets, u=(1, 0), origin=(0, 0))
    assert abs(pitch_A - 100) < 1
    assert abs(pitch - 100) < 15


def test_estimate_pitch_source_b_validates_transitions():
    # Kimlik geçişleri tam pitch aralığında → pitch_B pitch_A ile tutarlı.
    dets = [_det(cat, 100 + i * 100, w=95)
            for i, cat in enumerate(["a", "b", "a", "b", "a", "b"])]
    pitch, pitch_A, pitch_B, unrel = estimate_pitch(dets, u=(1, 0), origin=(0, 0))
    assert pitch_B is not None and not unrel
    assert abs(pitch_B - 100) < 10


def test_estimate_pitch_inconsistent_uses_stable_a():
    # Tutarsız (A≫B) durumda GÜVENLİ taban pitch_A seçilir — min(A,B) temiz tepsileri
    # aşırı-böldüğü için (ölçülmüş regresyon) kullanılmaz. B yalnız telemetride kalır.
    dets = [_det(cat, 100 + i * 100, w=180)
            for i, cat in enumerate(["a", "b", "a", "b", "a", "b", "a"])]
    pitch, pitch_A, pitch_B, unrel = estimate_pitch(dets, u=(1, 0), origin=(0, 0))
    assert pitch_A > 150 and pitch_B is not None and pitch_B < 130
    assert abs(pitch - pitch_A) < 1e-6   # tutarsız → kararlı pitch_A


def test_estimate_pitch_b_unreliable_single_product_falls_back_to_a():
    # Tek sınıf → 0 kimlik-geçişi → pitch_B güvenilmez → pitch_A + bayrak.
    dets = [_det("a", 100 + i * 100, w=100) for i in range(4)]
    pitch, pitch_A, pitch_B, unrel = estimate_pitch(dets, u=(1, 0), origin=(0, 0))
    assert unrel and abs(pitch - pitch_A) < 1e-6


# ── _phase_offset / build_lattice ────────────────────────────────────────────────
def test_phase_offset_centers_on_products():
    ss = [100.0, 200.0, 300.0, 400.0]  # pitch=100, hepsi tam katta
    phi = _phase_offset(ss, 100.0)
    # φ, ürünlerin bulunduğu faza (0 mod 100) oturmalı
    assert min(abs(phi - 0), abs(phi - 100)) < 5


def test_build_lattice_splits_same_product_block():
    # 10'luk tek-ürün bandı, u boyunca pitch aralıklarla → ~10 kanal (kimlikten bağımsız).
    ss = [i * 100.0 for i in range(10)]
    centers = build_lattice(ss, 100.0, left_edge_s=None, right_edge_s=None)
    assert len(centers) == 10


def test_build_lattice_interior_empty_channel():
    # İki ürün grubu arası 300'lük boşluk, pitch=100 → aradaki 2 kanal ürünsüz (boş).
    ss = [0.0, 100.0, 400.0, 500.0]
    centers = build_lattice(ss, 100.0, left_edge_s=None, right_edge_s=None)
    assert len(centers) == 6  # 0,100,200,300,400,500


def test_build_lattice_extends_to_edge_no_partial():
    # Sol kenar -280'de: kanal TAM sığdığı sürece uzar, kısmi kanal üretilmez.
    ss = [0.0, 100.0]
    centers = build_lattice(ss, 100.0, left_edge_s=-280.0, right_edge_s=None)
    # -200 kanalı [-250,-150] kenar içinde; -300 kanalı [-350,-250] kenarı (-280) geçer → durur.
    assert min(centers) == -200.0


# ── uçtan uca ────────────────────────────────────────────────────────────────────
def test_lane_grid_same_product_two_channels_split():
    # Aynı ürün fiziksel olarak geniş banda yayılmış → tek dev slot ÇIKMAMALI.
    wide = [_det("magnum_beyaz", i * 100, cy=40 + (i % 4) * 50, w=95) for i in range(10)]
    grid = estimate_grid_lane_based(None, 7, wide)
    assert grid["grid_status"] == "ok"
    assert grid["grid_source"] == "zincir"
    # 10 ürün birden fazla kanala dağılmalı — tek dev kanal DEĞİL. Sayım TEK KAYNAK:
    # chain_slot_index_for (assign_slots ile aynı test).
    counts = [0] * grid["column_count"]
    for d in wide:
        counts[chain_slot_index_for((d["bbox"][0] + d["bbox"][2]) / 2,
                                    (d["bbox"][1] + d["bbox"][3]) / 2, grid)] += 1
    assert max(counts) < 10
    assert sum(counts) == 10


def test_lane_grid_neighboring_same_product_not_merged():
    dets = _stack("magnum_beyaz", 100, n=4, w=95) + _stack("magnum_beyaz", 400, n=4, w=95)
    grid = estimate_grid_lane_based(None, 7, dets)
    assert grid["grid_status"] == "ok"
    idx_a = chain_slot_index_for(100, 100, grid)
    idx_b = chain_slot_index_for(400, 100, grid)
    assert idx_a != idx_b


def test_lane_grid_boundaries_never_cross_detection_center():
    # DOĞRULAMA (spec assert): hiçbir sınır bir tespit kutusunun merkezinden geçmez.
    dets = (_stack("magnum_klasik", 100, n=5, w=95)
            + _stack("cornetto_cilek", 200, n=5, w=95)
            + _stack("magnum_beyaz", 300, n=5, w=95)
            + _stack("twister_kavun", 400, n=5, w=95))
    grid = estimate_grid_lane_based(None, 7, dets)
    assert grid["grid_status"] == "ok"
    assert grid["lane_meta"]["boundary_center_violations"] == 0


def test_lane_grid_every_detection_stays_in_own_channel():
    dets = (_stack("magnum_klasik", 100, n=5, w=95)
            + _stack("cornetto_cilek", 200, n=5, w=95)
            + _stack("magnum_beyaz", 300, n=5, w=95)
            + _stack("twister_kavun", 400, n=5, w=95)
            + _stack("nogger_karamel", 500, n=5, w=95)
            + _stack("frigola_bar", 600, n=5, w=95)
            + _stack("viennetta", 700, n=5, w=95))
    grid = estimate_grid_lane_based(None, 7, dets)
    assert grid["grid_status"] == "ok"
    for cat in ["magnum_klasik", "cornetto_cilek", "magnum_beyaz", "twister_kavun",
                "nogger_karamel", "frigola_bar", "viennetta"]:
        members = [d for d in dets if d["category"] == cat]
        idxs = {chain_slot_index_for((d["bbox"][0] + d["bbox"][2]) / 2,
                                     (d["bbox"][1] + d["bbox"][3]) / 2, grid) for d in members}
        assert len(idxs) == 1, f"{cat} birden fazla kanala dağıldı: {idxs}"


def test_lane_grid_interior_empty_regression():
    dets = (_stack("magnum_klasik", 100, n=4, w=95) + _stack("magnum_beyaz", 200, n=4, w=95)
            + _stack("cornetto_cilek", 300, n=4, w=95) + _stack("magnum_karamel", 500, n=4, w=95))
    grid = estimate_grid_lane_based(None, 7, dets)
    assert grid["grid_status"] == "ok"
    assert grid["lane_meta"]["n_interior_empty"] == 1
    idx_a = chain_slot_index_for(300, 100, grid)
    idx_b = chain_slot_index_for(500, 100, grid)
    assert idx_b - idx_a == 2


def test_lane_grid_rotated_photo_still_separates_lanes():
    dets = (_hstack("magnum_klasik", 100, n=5) + _hstack("cornetto_cilek", 300, n=5)
            + _hstack("magnum_beyaz", 500, n=5))
    grid = estimate_grid_lane_based(None, 7, dets)
    assert grid["grid_status"] == "ok"
    assert abs(grid["lane_meta"]["angle_deg"]) < 5
    idx_a = chain_slot_index_for(100, 100, grid)
    idx_b = chain_slot_index_for(100, 300, grid)
    idx_c = chain_slot_index_for(100, 500, grid)
    assert len({idx_a, idx_b, idx_c}) == 3


def test_lane_grid_insufficient_products_is_uncertain():
    grid = estimate_grid_lane_based(None, 7, [_det("magnum_klasik", 100)])
    assert grid["grid_status"] == "belirsiz"


def test_lane_grid_manual_corners_bypasses_lattice_pipeline():
    dets = _stack("magnum_klasik", 100, n=4)
    corners = [(0, 0), (700, 0), (700, 500), (0, 500)]
    grid = estimate_grid_lane_based(None, 7, dets, corners_override=corners)
    assert grid["grid_status"] == "ok"
    assert grid["grid_source"] == "manuel"
