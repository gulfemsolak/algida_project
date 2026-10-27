"""Kanal ızgarası kurulduktan sonra sayımı şişiren iki tür hatalı kutuyu ayıklar.

Ölçüldü (08.10.2026, 122 fotoğraf, köşe modeliyle kurulan ızgaralar):

1. ALT / ÜST RAF ÜRÜNÜ: tepsinin önündeki boşluktan alttaki rafın, kadrajın üstünden
   üstteki rafın ürünleri görünüyor (ORSAN raf 5, slot 53-55: 6 Cornetto + 1 Viennetta /
   Twister → 7). Alttakiler de aynı
   sütuna düştükleri için bu rafın kanalına sayılıyor (ORSAN raf 1, slot 15: 8 Magnum
   beyaz + 1 Viennetta + 1 Twister → 10). Ayırt edici iki koşul BİRLİKTE: kutu kanalın
   çoğunluk ürününden farklı VE merkezi tepsinin ön kenarının ötesinde (ya da arka
   kenarının gerisinde). Tek başına konum
   yetmez (açılı çekimde gerçek ön paketler tepsinin %50 önüne taşıyor), tek başına
   sınıf da yetmez (karışık kanallar var). İkisi birlikte 122 fotoda yalnız bu 2 kutuyu
   yakaladı.

2. PARÇA KUTU: model zaten kutusu olan bir paketin bir bölümüne (alt logo, üst kenar)
   ikinci, küçük bir kutu çiziyor (ORSAN raf 2, slot 23: 8 paket → 9). Koşullar BİRLİKTE:
   kanalın ortanca kutu boyunun %60'ından kısa VE alanının ≥%80'i aynı ürünün başka
   kutularıyla örtülü. Sıkışık spiralde gerçek paketlerin de çoğu ≥%70 örtülü olduğu için
   boy koşulu şart. 122 fotoda 12 kutu yakaladı; hepsi gözle kontrol edildi, hepsi parça.

Izgara ``belirsiz`` ise hiçbir şey ayıklanmaz.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.analysis.shelf_grid import transform_point
from src.analysis.slot_assigner import _col_index
from src.utils.logger import get_logger

log = get_logger(__name__)

LATERAL_MARGIN = 0.12          # tepsi genişliğinin bu kadar dışı (gerçek taşma ölçümde ≤ %3)
BACK_MARGIN = 0.10             # tepsi derinliğinin bu kadar gerisi (gerçek paket ölçümde ≥ -0.069)
FOREIGN_MIN_MAJORITY = 3        # çoğunluk ürünü en az bu kadar kutuyla kanalda olmalı
NEST_INSIDE = 0.85             # iç kutunun bu kadarı tek bir dış kutunun içinde
NEST_OUTER_COVERED = 0.50      # dış kutunun iç kutu dışı kısmının bu kadarı başka kutularca kaplı → dış kutu birleşik
FRAGMENT_MAX_HEIGHT_FRAC = 0.60
FRAGMENT_MIN_COVER = 0.80


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(a, b) -> float:
    """a'nın alanının b içinde kalan oranı."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / _area(a) if _area(a) > 0 else 0.0


def _cover_frac(box: list[float], others: list[list[float]], haric: list[float] | None = None) -> float:
    """box alanının others tarafından kaplanan oranı; haric verilirse o bölge hesaba katılmaz."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    mask = np.zeros((h, w), dtype=bool)
    for o in others:
        a1, b1 = max(int(o[0]), x1), max(int(o[1]), y1)
        a2, b2 = min(int(o[2]), x2), min(int(o[3]), y2)
        if a2 > a1 and b2 > b1:
            mask[b1 - y1:b2 - y1, a1 - x1:a2 - x1] = True
    if haric is None:
        return float(mask.mean())
    gecerli = np.ones((h, w), dtype=bool)
    a1, b1 = max(int(haric[0]), x1), max(int(haric[1]), y1)
    a2, b2 = min(int(haric[2]), x2), min(int(haric[3]), y2)
    if a2 > a1 and b2 > b1:
        gecerli[b1 - y1:b2 - y1, a1 - x1:a2 - x1] = False
    return float(mask[gecerli].mean()) if gecerli.any() else 1.0


def clean_detections(
    detections: list[dict[str, Any]],
    grid: dict[str, Any],
    *,
    empty_class_name: str = "empty_slot",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hatalı kutuları ayıkla. Dönüş: ``(temiz_tespitler, {"alt_raf": n, "parca": n})``."""
    info = {"alt_raf": 0, "parca": 0, "tepsi_disi": 0, "ic_ice": 0}
    if grid.get("grid_status") != "ok" or not detections:
        return detections, info

    h_mat = grid.get("H")
    rect_h = grid.get("rect_height")
    prods = [i for i, d in enumerate(detections) if d.get("category") != empty_class_name]
    rect_w = grid.get("rect_width")
    channel: dict[int, int] = {}
    depth: dict[int, float | None] = {}
    drop: set[int] = set()
    for i in prods:
        x1, y1, x2, y2 = detections[i]["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        channel[i] = _col_index(cx, cy, grid)
        depth[i] = None
        if h_mat is not None and rect_h and rect_w:
            u, v = transform_point(cx, cy, h_mat)
            depth[i] = v / rect_h
            # 0) tepsinin yanlarından tamamen dışarıda: duvar, kapı, yan bölme (ORSAN raf 4:
            # kadraj köşesindeki duvar kutusu 41. slota sayıldı). Kanal ataması kenar kanala
            # kırptığı için bunlar sessizce en dış kanala ekleniyordu.
            # Arka kenarın gerisi: üst rafın kadraj tepesinde görünen ürünü. Kanal boşsa
            # "çoğunluktan farklı" kuralı yakalayamaz (ORSAN raf 5, slot 56). Ölçüldü (09.10.2026,
            # 2 model × 160 foto): gerçek arka sıra paketleri en fazla -0.069, üst raf -0.128.
            if (u / rect_w < -LATERAL_MARGIN or u / rect_w > 1 + LATERAL_MARGIN
                    or depth[i] < -BACK_MARGIN):
                drop.add(i)
                info["tepsi_disi"] += 1
    for ch in set(channel.values()):
        idx = [i for i in prods if channel[i] == ch and i not in drop]
        if not idx:
            continue
        majority, n_major = Counter(detections[i]["category"] for i in idx).most_common(1)[0]

        # 1) alt raf ürünü
        if n_major >= FOREIGN_MIN_MAJORITY:
            for i in idx:
                if (detections[i]["category"] != majority and depth[i] is not None
                        and (depth[i] > 1.0 or depth[i] < 0.0)):
                    drop.add(i)
                    info["alt_raf"] += 1

        # 2) parça kutu
        heights = [detections[i]["bbox"][3] - detections[i]["bbox"][1] for i in idx]
        med_h = float(np.median(heights)) if heights else 0.0
        for i, h in zip(idx, heights):
            if i in drop or med_h <= 0 or h >= FRAGMENT_MAX_HEIGHT_FRAC * med_h:
                continue
            same = [detections[j]["bbox"] for j in idx
                    if j != i and j not in drop and detections[j]["category"] == detections[i]["category"]]
            if same and _cover_frac(detections[i]["bbox"], same) >= FRAGMENT_MIN_COVER:
                drop.add(i)
                info["parca"] += 1

    # 3) birleşik dış kutu: aynı ürünün bir kutusu ötekinin neredeyse tamamen içinde ve dış
    # kutunun geri kalanı da başka kutularca ≥%50 kaplı → dış kutu, zaten kendi kutuları olan
    # paketleri birleştirmiş; atmak hiçbir paketi kaybettirmez (ORSAN raf 4, slot 45: 8 → 9).
    # "İç kutu küçükse iç kutuyu at" kuralı BİLEREK yok: etiketli veride gerçek, yarısı görünen
    # bir paketin büyük çizilmiş komşu kutunun içinde kaldığı görüldü (dış kalan %13 kaplı) ve
    # gerçek parçalardan (%10-36) ayrışmıyor. Parçalar için yukarıdaki 2. kural var.
    kalan = [i for i in prods if i not in drop]
    for i in kalan:
        if i in drop:
            continue
        a = detections[i]["bbox"]
        for j in kalan:
            if j == i or j in drop or detections[j]["category"] != detections[i]["category"]:
                continue
            b = detections[j]["bbox"]
            if _area(b) <= _area(a) or _inside(a, b) < NEST_INSIDE:
                continue
            digerleri = [detections[k]["bbox"] for k in kalan
                         if k not in (i, j) and k not in drop and detections[k]["category"] == detections[i]["category"]]
            if _cover_frac(b, digerleri, haric=a) >= NEST_OUTER_COVERED:
                drop.add(j)
                info["ic_ice"] += 1
            break

    if drop:
        log.info("tespit ayıklama: tepsi dışı %d, komşu raf %d, parça %d, iç içe %d",
                 info["tepsi_disi"], info["alt_raf"], info["parca"], info["ic_ice"])
    return [d for i, d in enumerate(detections) if i not in drop], info
