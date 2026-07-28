"""Tests for slot_assigner.py — fiziksel ızgaraya çapalı pitch-tabanlı slot atama.

Izgara artık tespit bulutundan DEĞİL, ``shelf_grid`` ile rafın fiziksel geometrisinden
türetilir. Bu testler atama değişmezlerini (hiçbir tespit düşmez), boş-raf regresyonunu
ve doğrulama senaryolarını (sola/sağa yaslı, yoğun kolon, eğim) kilitler.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.slot_assigner import (
    make_slot_id,
    assign_slots,
    build_slot_report,
)
from src.analysis.shelf_grid import estimate_grid, slot_index_for


def _det(category, cx, cy=100, w=30, h=50, conf=0.9):
    return {"category": category, "confidence": conf, "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


def _stack(category, cx, n=4, cy0=40, dy=50):
    """Aynı sütunda dikey istiflenmiş n ürün (x-merkezleri aynı)."""
    return [_det(category, cx, cy=cy0 + i * dy) for i in range(n)]


def _grid(origin=100.0, pitch=100.0, count=7):
    """Test için sabit homografili ızgara (görüntüden bağımsız, grid_status=ok).

    H, orijinal x'i rektifiye uzaya öteler: x_rect = x_orig - (origin - pitch/2).
    Böylece eski (origin, pitch) çiftiyle aynı kolon indekslerini üretir.
    """
    roi_left = origin - pitch / 2.0
    h_mat = [[1.0, 0.0, -roi_left], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    h_inv = [[1.0, 0.0, roi_left], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "grid_status": "ok", "column_count": count, "pitch": pitch,
        "rect_width": pitch * count, "rect_height": 1000.0,
        "centers": [pitch / 2.0 + i * pitch for i in range(count)],
        "boundaries": [i * pitch for i in range(count + 1)],
        "corners": None, "H": h_mat, "H_inv": h_inv,
        "grid_source": "kontur", "shelf_number_from_bar": None,
        "confident": True,
    }


# ── Numaralandırma ──────────────────────────────────────────────────────────────
def test_make_slot_id():
    assert make_slot_id(1, 2) == 12
    assert make_slot_id(3, 1) == 31
    assert make_slot_id(7, 7) == 77


# ── Pitch tabanlı indeksleme ────────────────────────────────────────────────────
def test_slot_index_pitch_based():
    g = _grid(origin=100, pitch=100, count=7)
    assert slot_index_for(100, 100, g) == 0
    assert slot_index_for(200, 100, g) == 1
    assert slot_index_for(640, 100, g) == 5   # (640-100)/100 = 5.4 → 5
    # Kırpma: ızgara dışına taşan tespit en yakın kolona atanır, düşmez.
    assert slot_index_for(-500, 100, g) == 0
    assert slot_index_for(99999, 100, g) == 6


# ── Değişmez: hiçbir tespit düşmez ──────────────────────────────────────────────
def test_conservation_invariant():
    dets = (_stack("magnum_klasik", 100) + _stack("cornetto_cilek", 300)
            + _stack("magnum_beyaz", 700) + [_det("magnum_karamel", 5000)])
    out = assign_slots(dets, _grid(), shelf_number=1)
    assert sum(s["count"] for s in out["slots"]) == len([d for d in dets if d["category"] != "empty_slot"])


def test_conservation_with_empty_slots_excluded():
    dets = _stack("magnum_klasik", 100, n=3) + [_det("empty_slot", 700), _det("empty_slot", 750)]
    out = assign_slots(dets, _grid(), shelf_number=1)
    # empty_slot ürün sayımına girmez; sadece 3 ürün.
    assert sum(s["count"] for s in out["slots"]) == 3


# ── Boş raf REGRESYONU (Hata C — kritik) ────────────────────────────────────────
def test_empty_shelf_regression_no_phantom_columns():
    # 7 kolonluk fiziksel ızgara tüm tepsiyi kaplar; ürünler yalnız sağdaki 2 kolonda.
    # Izgara tespitlerden türetilmediği için sol kolonlar BOŞ kalmalı (sahte kolon yok).
    g = _grid(origin=100, pitch=100, count=7)   # merkezler 100..700
    dets = _stack("magnum_klasik", 600, n=4) + _stack("magnum_beyaz", 700, n=3)
    out = assign_slots(dets, g, shelf_number=1)
    empties = [s["is_empty"] for s in out["slots"]]
    assert empties == [True, True, True, True, True, False, False]
    assert sum(s["count"] for s in out["slots"]) == 7
    # Boş kolonlara ürün yazılmadı.
    assert out["slots"][0]["count"] == 0
    assert out["slots"][5]["count"] == 4 and out["slots"][6]["count"] == 3


# ── Sola / sağa yaslı dolu raf (Hata A/B) ───────────────────────────────────────
def test_left_aligned_first_slots_not_spuriously_empty():
    g = _grid(origin=100, pitch=100, count=7)
    # İlk 3 kolon dolu (sola yaslı) → slot 1-3 dolu, gerisi boş.
    dets = _stack("magnum_klasik", 100) + _stack("magnum_beyaz", 200) + _stack("magnum_cookie", 300)
    out = assign_slots(dets, g, shelf_number=1)
    assert out["slots"][0]["count"] == 4
    assert not out["slots"][0]["is_empty"]
    assert sum(s["count"] for s in out["slots"]) == 12


def test_right_aligned_last_slots_filled():
    g = _grid(origin=100, pitch=100, count=7)
    dets = _stack("magnum_klasik", 500) + _stack("magnum_beyaz", 600) + _stack("magnum_cookie", 700)
    out = assign_slots(dets, g, shelf_number=1)
    assert out["slots"][6]["count"] == 4
    assert not out["slots"][6]["is_empty"]


# ── Yoğun tek-kolon dağılımı ────────────────────────────────────────────────────
def test_dense_shelf_distributes_across_columns():
    # 7 kolon × 5 ürün, tüm genişliğe yayılı → her kolon makul (~5), tek kolonda 35 DEĞİL.
    g = _grid(origin=100, pitch=100, count=7)
    dets = []
    for i in range(7):
        dets += _stack("magnum_klasik", 100 + i * 100, n=5)
    out = assign_slots(dets, g, shelf_number=1)
    counts = [s["count"] for s in out["slots"]]
    assert counts == [5, 5, 5, 5, 5, 5, 5]
    assert all(3 <= c <= 12 for c in counts)


# ── Çoğunluk oyu + toplam adet ──────────────────────────────────────────────────
def test_majority_vote_and_total_count():
    g = _grid(origin=100, pitch=100, count=1)
    col = [_det("magnum_klasik", 100, cy=40), _det("magnum_klasik", 100, cy=100),
           _det("magnum_klasik", 100, cy=160), _det("cornetto_cilek", 100, cy=220)]
    out = assign_slots(col, g, shelf_number=1)
    slot = out["slots"][0]
    assert slot["product_name"] == "magnum_klasik"
    assert slot["count"] == 4                  # TOPLAM (çoğunluk 3 değil)
    assert slot["is_empty"] is False


def test_empty_slot_does_not_override_product():
    g = _grid(origin=100, pitch=100, count=1)
    dets = _stack("magnum_klasik", 100, n=3) + [_det("empty_slot", 100, cy=250)]
    out = assign_slots(dets, g, shelf_number=1)
    slot = out["slots"][0]
    assert slot["product_name"] == "magnum_klasik"
    assert slot["count"] == 3
    assert slot["is_empty"] is False


def test_slot_ids_and_shelf_number():
    g = _grid(origin=100, pitch=100, count=2)
    dets = _stack("magnum_klasik", 100) + _stack("magnum_klasik", 200)
    out = assign_slots(dets, g, shelf_number=3)
    ids = [s["slot_id"] for s in out["slots"]]
    assert ids == [31, 32]


# ── Hizalama şüphesi + kapasite ─────────────────────────────────────────────────
def test_alignment_suspect_over_capacity():
    # Bir slotta kapasiteyi (~12) aşan yığın (14) → şüphe işareti (ama ≤18, belirsiz değil).
    g = _grid(origin=100, pitch=100, count=7)
    dets = ([_det("magnum_klasik", 300, cy=40 + i * 20) for i in range(14)]
            + _stack("magnum_beyaz", 100, n=3) + _stack("magnum_cookie", 700, n=3))
    out = assign_slots(dets, g, shelf_number=1)
    assert out["grid_status"] == "ok"
    suspect = out["slots"][2]
    assert suspect["count"] == 14
    assert suspect["alignment_suspect"] is True
    assert 13 in out["alignment_suspects"]


def test_over_hard_capacity_downgrades_to_uncertain():
    # 20 > 18 (1.5× kapasite) → ızgara belirsize düşer, atama yapılmaz.
    g = _grid(origin=100, pitch=100, count=7)
    dets = [_det("magnum_klasik", 300, cy=40 + i * 15) for i in range(20)]
    out = assign_slots(dets, g, shelf_number=1)
    assert out["grid_status"] == "belirsiz"
    assert out["slots"] == []


def test_mixed_slot_flags_suspect():
    # Tek slotta iki sınıf, ikincil pay %25'ten fazla → karışık → şüphe.
    g = _grid(origin=100, pitch=100, count=7)
    dets = _stack("magnum_klasik", 300, n=6) + _stack("cornetto_cilek", 300, n=4)
    out = assign_slots(dets, g, shelf_number=1)
    slot = out["slots"][2]
    assert slot["product_name"] == "magnum_klasik"
    assert slot["mixed"] is True
    assert slot["alignment_suspect"] is True


def test_breakdown_lists_all_classes():
    g = _grid(origin=100, pitch=100, count=7)
    dets = _stack("magnum_klasik", 300, n=5) + _stack("cornetto_cilek", 300, n=2)
    out = assign_slots(dets, g, shelf_number=1)
    bd = {b["category"]: b["count"] for b in out["slots"][2]["breakdown"]}
    assert bd == {"magnum_klasik": 5, "cornetto_cilek": 2}


def test_no_false_alignment_suspect_when_balanced():
    g = _grid(origin=100, pitch=100, count=7)
    dets = []
    for i in range(7):
        dets += _stack("magnum_klasik", 100 + i * 100, n=4)
    out = assign_slots(dets, g, shelf_number=1)
    assert out["alignment_suspects"] == []


# ── build_slot_report ───────────────────────────────────────────────────────────
def test_build_slot_report_without_image_falls_back_to_lane_clustering():
    # Görüntü yok → homografi çapası yapılamaz, ama yeterli tespit varsa lane-kümeleme
    # ızgarası devreye girer (uydurma DEĞİL — tespit bulutundan sağlam türetilir).
    dets = _stack("magnum_klasik", 100) + _stack("magnum_klasik", 400)
    rep = build_slot_report(dets, shelf_number=2, column_count=7)
    assert rep["grid_status"] == "ok"
    assert rep["grid"]["grid_source"] == "zincir"
    assert sum(s["count"] for s in rep["slots"]) == 8


def test_build_slot_report_without_image_and_without_products_is_uncertain():
    # Ne görüntü ne yeterli tespit → hiçbir çapa yok → belirsiz (uydurma fallback YOK).
    dets = [_det("magnum_klasik", 100)]
    rep = build_slot_report(dets, shelf_number=2, column_count=7)
    assert rep["grid_status"] == "belirsiz"
    assert rep["slots"] == []


def test_build_slot_report_with_grid_assigns():
    dets = _stack("magnum_klasik", 100) + _stack("magnum_beyaz", 400)
    rep = build_slot_report(dets, shelf_number=2, column_count=7, grid=_grid())
    assert rep["grid_status"] == "ok"
    assert len(rep["slots"]) == 7
    assert sum(s["count"] for s in rep["slots"]) == 8


def test_build_slot_report_corners_override():
    dets = _stack("magnum_klasik", 150) + _stack("magnum_beyaz", 750)
    corners = [(100, 0), (800, 0), (800, 500), (100, 500)]
    rep = build_slot_report(dets, shelf_number=1, column_count=7,
                            image=None, corners_override=corners)
    assert rep["grid_status"] == "ok"
    assert sum(s["count"] for s in rep["slots"]) == 8
