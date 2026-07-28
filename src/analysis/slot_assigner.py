"""Slot (sütun) katmanı — tespitleri fiziksel ızgaraya göre kolonlara atar ve numaralandırır.

Bir fotoğraf = bir raf. Izgara ARTIK tespit bulutundan türetilmez; ``shelf_grid`` rafın
fiziksel geometrisinden (tepsi 4 köşesi → homografi → belirsiz) rektifiye bir kolon
ızgarası üretir. Tepsi yukarıdan açılı çekildiği için lane'ler perspektifle yelpaze
gibi açılır (paralel değildir); atama bu yüzden REKTİFİYE uzayda yapılır — orada
lane'ler gerçekten eşit aralıklıdır. Bu üç hatayı birden çözer:

  * Hata A/B — tespitler eşit-genişlikli bin yerine gerçek kolon merkezine (rektifiye
    uzayda pitch tabanlı indeks) atanır; ROI tepsiye çapalı olduğundan ilk/son slotlar
    boş kalmaz.
  * Hata C — ROI genişliği tespitlerden gelmediği için büyük ölçüde boş rafta boş kolonlar
    boş kalır; sağdaki 1–2 kolon 7 sahte kolona bölünmez.

Izgara güvenilir değilse (``grid_status="belirsiz"``) ATAMA YAPILMAZ — tespitler yine
gösterilir ama slot dağılımı hesaplanmaz. Kendinden emin yanlış > açıkça belirsiz.

Atama (homografi ile rektifiye uzayda, kümeleme YOK):
  tespit merkezi H ile rektifiye uzaya taşınır, ``slot_index = floor(x_rect / pitch)``,
  [0, count-1]'e kırpılır. Her tespit MUTLAKA bir slot alır (en yakın kolon); hiçbir
  tespit düşmez — ``sum(slot_counts) == len(products)`` assert ile garanti edilir.

Katmanlar:
  * Baskın ürün: slot içi çoğunluk oyu. ``empty_slot`` baskın sınıfı BELİRLEMEZ; yalnız o
    slotta başka ürün yoksa "boş" kararını destekler. İkincil sınıf payı %25'i geçerse
    (karışık slot) hizalama şüphesi işaretlenir.
  * Adet: slottaki TOPLAM ürün tespiti.
  * Numaralandırma: slot_id = raf_no * 10 + kolon_index (1-tabanlı, soldan sağa).
  * Kapasite: bir slot sayımı fiziksel kapasiteyi (~12) aşarsa hizalama şüphesi; 1.5 katını
    (~18) aşarsa ızgara ``belirsiz``'e düşürülür (24 ürün raporlamaktansa belirsizlik).

``analyze_slots`` config + KANONİK görüntüden ızgara üretir (dashboard girişi).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import yaml

from src.analysis.product_anchored_grid import (
    _bbox_size, _dist, chain_slot_index_for, estimate_grid_lane_based,
)
from src.analysis.shelf_grid import slot_index_for
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Ayarlanabilir varsayılanlar (config.yaml > slot_analysis geçersiz kılar) ────
DEFAULT_COLUMN_COUNT = 7          # freezer'ın fiziksel sütun sayısı
DEFAULT_SHELF_NUMBER = 1          # operatör/QR/ray yoksa raf numarası
EMPTY_CLASS_NAME = "empty_slot"

# Tutarlılık: slot sayımının boş-olmayan medyana oranı bu eşiği aşar VE komşusu boşsa
# → hizalama şüphesi (tespitler yanlış kolona düşmüş olabilir).
ALIGNMENT_SUSPECT_RATIO = 1.8
# Fiziksel kolon kapasitesi (~ürün). Aşımı hizalama hatasıdır, veri değil.
SLOT_CAPACITY = 12
SLOT_CAPACITY_HARD = 18          # 1.5× — bu üstünde ızgara belirsiz sayılır
# Karışık slot: ikincil sınıf payı bu oranı geçerse hizalama şüphesi.
SECONDARY_CLASS_FRAC = 0.25
# Sayım-tabanlı rapor (low_detection): merkezler arası mesafe <= bu * medyan-ürün-boyu ise
# aynı kanal (üst üste), aksi halde ayrı kanal. build_chains'teki LINK_DIST_FACTOR ile aynı
# değer/mantık — yalnız SINIF KISITI YOK (kanal aynılığı kimlikten bağımsız).
COUNT_REPORT_LINK_DIST_FACTOR = 1.8


def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _count_based_report(products: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Geometri (pitch/faz/eksen) kurmadan SAYIM tabanlı doluluk raporu.

    ``low_detection`` durumunda (N < ``COMB_MIN_PRODUCTS``, latis mimari olarak
    güvenilmez — bkz. ``product_anchored_grid`` modül docstring'i) kullanılır.

    ÖNCE DENENDİ, ÖLÇÜLEREK REDDEDİLDİ: ``dominant_axis``/PCA ile bir eksene izdüşürüp
    ardışık boşluğa bakmak — sentetik bir testte (4 ürün, birbirinden uzak, tek bir
    doğru üzerinde) YANLIŞ ÇIKTI: PCA az noktada (chain yoksa) noktaların en çok
    YAYILDIĞI yönü "kanal boyu" (v) sayıp dik ekseni (u, kanal-ayrımı) buna göre seçer —
    ürünler yatayda yayılmışsa bu YANLIŞLIKLA v=yatay, u=dikey seçip hepsini aynı s'ye
    çökertir (tam 2-nokta PCA dejenereliğinin genellemesi, bkz. N=2 tartışması aşağıda).

    Bunun yerine EKSENSİZ, saf 2B yakınlık kümelemesi: ``build_chains``'teki AYNI
    mesafe eşiği (``LINK_DIST_FACTOR``·medyan-ürün-boyu) — ama SINIF KISITI YOK (kanal
    aynılığı kimlikten bağımsız, komşu kanaldaki farklı ürün de "ayrı küme" sayılmalı).
    İki tespit merkezleri bu mesafenin içindeyse aynı kanalda (üst üste/yakın), aksi
    halde ayrı kanal.

    N=2 ÖZEL DURUM: bu yöntemde dejenere DEĞİL — mesafe mutlak, yöne bakmaz. Gerçek
    referans fotoğrafta (2 ürün aynı kanalda üst üste, merkezler arası ~148px, medyan
    boy ~170px → 148 < 1.8·170) doğru şekilde "1 kanal" verir; birbirinden UZAK 2 ürün
    (farklı kanal) de doğru şekilde "2 kanal" verir (bkz. testler).

    ``k``'dan fazla küme bulunursa (kümeleme K'yı aşarsa) ``k``'ya kırpılır ve
    ``kume_asimi=True`` işaretlenir (doluluk %100'ü asla geçmez).
    """
    if not products or k <= 0:
        return {"dolu_kanal": 0, "bos_kanal": max(k, 0), "doluluk": 0.0, "kume_asimi": False}

    n = len(products)
    sizes = [_bbox_size(d) for d in products]
    median_size = float(np.median(sizes)) or 1.0
    max_dist = COUNT_REPORT_LINK_DIST_FACTOR * median_size

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _dist(products[i], products[j]) <= max_dist:
                union(i, j)

    n_kume = len({find(i) for i in range(n)})

    kume_asimi = n_kume > k
    dolu_kanal = min(n_kume, k)
    return {
        "dolu_kanal": dolu_kanal,
        "bos_kanal": k - dolu_kanal,
        "doluluk": round(dolu_kanal / k, 4),
        "kume_asimi": kume_asimi,
    }


def _cx(det: dict[str, Any]) -> float:
    x1, _, x2, _ = det["bbox"]
    return (x1 + x2) / 2.0


def _cy(det: dict[str, Any]) -> float:
    _, y1, _, y2 = det["bbox"]
    return (y1 + y2) / 2.0


def _col_index(x: float, y: float, grid: dict[str, Any]) -> int:
    """Kolon indeksi: ``zincir`` ızgarası kendi sınır-doğrusu testini (yelpazeye izin
    verir), diğerleri (kontur/manuel homografi) pitch-tabanlı ``slot_index_for``'u kullanır."""
    if grid.get("grid_source") == "zincir" and grid.get("boundary_lines"):
        return chain_slot_index_for(x, y, grid)
    return slot_index_for(x, y, grid)


def make_slot_id(shelf_number: int, column_index_1based: int) -> int:
    """Slot kimliği = raf_no * 10 + sütun_index (1-tabanlı, soldan sağa).

    Raf 1, sütun 2 → 12 · Raf 3, sütun 1 → 31.
    """
    return shelf_number * 10 + column_index_1based


def _slot_record(
    index0: int,
    products: list[dict[str, Any]],
    shelf_number: int,
    empty_class_name: str,
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Bir kolon indeksinden slot kaydı üret (çoğunluk oyu + toplam adet + çizim sınırları).

    ``grid["numbering_reversed"]`` (GÖREV 1, Aday B — bkz. ``product_anchored_grid.py``
    ``_lane_grid`` notu): yalnızca ETİKETLEME'yi (``column_no``/``slot_id``) çevirir;
    ``index0`` hâlâ GEOMETRİK bin indeksidir (``centers``/``boundaries`` sırasıyla
    birebir), atama/çizim konumları asla etkilenmez — tek değişen, hangi fiziksel
    kanala hangi numaranın yazıldığıdır."""
    if grid.get("numbering_reversed"):
        index_1based = grid["column_count"] - index0
    else:
        index_1based = index0 + 1
    slot_id = make_slot_id(shelf_number, index_1based)
    center = grid["centers"][index0]
    x_left = grid["boundaries"][index0]
    x_right = grid["boundaries"][index0 + 1]

    base = {
        "slot_id": slot_id, "shelf_no": shelf_number, "column_no": index_1based,
        "x": center, "x_left": x_left, "x_right": x_right,
        "alignment_suspect": False,
    }

    if products:
        counts = Counter(d["category"] for d in products)
        product_name = counts.most_common(1)[0][0]
        dom = [d for d in products if d["category"] == product_name]
        confidence = round(sum(d["confidence"] for d in dom) / len(dom), 4)
        # Slottaki TÜM sınıflar (baskın öne, ikincil düşük vurgulu gösterilir).
        breakdown = [{"category": c, "count": n} for c, n in counts.most_common()]
        secondary = len(products) - len(dom)
        base.update({
            "count": len(products),          # TOPLAM ürün tespiti (çoğunluk değil, hepsi)
            "product_name": product_name, "confidence": confidence, "is_empty": False,
            "breakdown": breakdown,
            # Karışık slot: ikincil sınıf payı %25'i geçerse hizalama şüphesi sinyali.
            "mixed": secondary / len(products) > SECONDARY_CLASS_FRAC,
        })
    else:
        base.update({
            "count": 0, "product_name": None, "confidence": 0.0, "is_empty": True,
            "breakdown": [], "mixed": False,
        })
    return base


def assign_slots(
    detections: list[dict[str, Any]],
    grid: dict[str, Any],
    shelf_number: int = DEFAULT_SHELF_NUMBER,
    empty_class_name: str = EMPTY_CLASS_NAME,
) -> dict[str, Any]:
    """Tespitleri fiziksel ızgaraya (pitch tabanlı en yakın indeks) ata.

    Args:
        detections: dikleştirilmiş görüntünün tespitleri.
        grid: ``shelf_grid.estimate_grid`` çıktısı (origin/pitch/shear/column_count/...).
        shelf_number: QR / ray / operatörden gelen raf numarası.
        empty_class_name: boş sınıf adı.

    Returns:
        ``{shelf_number, column_count, grid, slots, alignment_suspects}``. Her slot:
        {slot_id, shelf_no, column_no, count, product_name, confidence, is_empty,
         x, x_left, x_right, alignment_suspect}.

    Garanti: ``sum(s["count"] for s in slots) == len(products)`` (hiçbir tespit düşmez).
    Izgara ``belirsiz`` ise atama yapılmaz (slots=[], grid_status korunur).
    """
    column_count = grid["column_count"]

    if grid.get("grid_status") != "ok":
        return {
            "shelf_number": shelf_number, "column_count": column_count,
            "grid": grid, "grid_status": grid.get("grid_status", "belirsiz"),
            "slots": [], "alignment_suspects": [],
        }

    products = [d for d in detections if d.get("category") != empty_class_name]
    empties = [d for d in detections if d.get("category") == empty_class_name]

    prod_bins: list[list[dict[str, Any]]] = [[] for _ in range(column_count)]
    empty_bins: list[int] = [0] * column_count

    for d in products:
        idx = _col_index(_cx(d), _cy(d), grid)
        prod_bins[idx].append(d)
    for d in empties:
        idx = _col_index(_cx(d), _cy(d), grid)
        empty_bins[idx] += 1

    slots = [
        _slot_record(i, prod_bins[i], shelf_number, empty_class_name, grid)
        for i in range(column_count)
    ]

    # empty_slot yalnız "boş" kararını DESTEKLER — ürün varken ezmez.
    for i, s in enumerate(slots):
        s["empty_slot_hits"] = empty_bins[i]

    # Hiçbir tespit düşmesin (en yakın kolona atama + kırpma) — sağlam garanti.
    assigned = sum(s["count"] for s in slots)
    assert assigned == len(products), (
        f"tespit kaybı: {assigned} atandı, {len(products)} bekleniyordu"
    )

    # Kapasite sertliği: bir slot 1.5× kapasiteyi (~18) aşıyorsa hizalama hatasıdır,
    # veri değil → 24 ürün raporlamaktansa ızgarayı belirsize düşür. Operatör ROI'yi elle
    # onayladıysa (manuel) düşürme — operatörün kararına güvenilir.
    if grid.get("grid_source") != "manuel" and any(s["count"] > SLOT_CAPACITY_HARD for s in slots):
        over = max(s["count"] for s in slots)
        log.warning("slot kapasitesi aşıldı (%d > %d) → ızgara belirsiz", over, SLOT_CAPACITY_HARD)
        belirsiz = {**grid, "grid_status": "belirsiz",
                    "reason": f"slot sayımı fiziksel kapasiteyi aştı ({over})"}
        return {
            "shelf_number": shelf_number, "column_count": column_count,
            "grid": belirsiz, "grid_status": "belirsiz",
            "slots": [], "alignment_suspects": [],
        }

    _flag_alignment_suspects(slots)

    return {
        "shelf_number": shelf_number,
        "column_count": column_count,
        "grid": grid,
        "grid_status": "ok",
        "slots": slots,
        "alignment_suspects": [s["slot_id"] for s in slots if s["alignment_suspect"]],
    }


def _flag_alignment_suspects(slots: list[dict[str, Any]]) -> None:
    """Hizalama şüphesi işaretle. Üç sinyal:

    1. Sayım boş-olmayan medyanın 1.8 katından fazla VE komşusu 0 (yanlış kolona yığılma).
    2. Sayım fiziksel kapasiteyi (~12) aşıyor (birden fazla lane yutulmuş).
    3. Karışık slot: ikincil sınıf payı %25'i geçiyor (``mixed``).
    """
    counts = [s["count"] for s in slots if not s["is_empty"]]
    median = 0
    if len(counts) >= 2:
        sorted_counts = sorted(counts)
        median = sorted_counts[len(sorted_counts) // 2]
    threshold = ALIGNMENT_SUSPECT_RATIO * median if median > 0 else float("inf")

    for i, s in enumerate(slots):
        if s["is_empty"]:
            continue
        if s["count"] > SLOT_CAPACITY:
            s["alignment_suspect"] = True
            continue
        if s.get("mixed"):
            s["alignment_suspect"] = True
            continue
        if s["count"] > threshold:
            left_empty = i > 0 and slots[i - 1]["count"] == 0
            right_empty = i < len(slots) - 1 and slots[i + 1]["count"] == 0
            if left_empty or right_empty:
                s["alignment_suspect"] = True


def build_slot_report(
    detections: list[dict[str, Any]],
    shelf_number: int = DEFAULT_SHELF_NUMBER,
    column_count: int = DEFAULT_COLUMN_COUNT,
    image: Any | None = None,
    grid: dict[str, Any] | None = None,
    corners_override: Any | None = None,
    empty_class_name: str = EMPTY_CLASS_NAME,
    expected_column_count: int | None = None,
) -> dict[str, Any]:
    """Tespitlerden slot bazlı rapor üret (ızgara tepsi köşelerine/homografiye çapalanır).

    Args:
        detections: tespitler (orijinal, döndürülmemiş görüntü koordinatlarında).
        shelf_number: bar-OCR / operatörden gelen raf numarası.
        column_count: rafın fiziksel sütun sayısı (config ``column_count``).
        image: BGR görüntü — tepsi köşe çapası için. None → grid belirsiz.
        grid: hazır ızgara (verilirse yeniden hesaplanmaz — tek kaynak paylaşımı için).
        corners_override: operatörün elle işaretlediği 4 köşe.
        empty_class_name: boş sınıf adı.
        expected_column_count: operatörün girdiği toplam kanal sayısı K. ``grid["low_detection"]``
            ise (N < ``COMB_MIN_PRODUCTS``, latis kurulamaz) ve bu verilmişse, geometrisiz
            SAYIM tabanlı ``count_report`` üretir (bkz. ``_count_based_report``).

    Returns:
        ``{shelf_number, column_count, grid, grid_source, grid_status, confident,
           slots, alignment_suspects, count_report}``. ``grid_status="belirsiz"`` ise
        ``slots=[]`` ve overlay yalnız tespitleri gösterir; ``count_report`` yalnız
        ``low_detection`` + ``expected_column_count`` verilmişse dolu olur, aksi halde None.
    """
    if grid is None:
        grid = estimate_grid_lane_based(
            image, column_count, detections,
            corners_override=corners_override, empty_class_name=empty_class_name,
            expected_column_count=expected_column_count,
        )

    assigned = assign_slots(detections, grid, shelf_number, empty_class_name)
    used_grid = assigned["grid"]  # kapasite aşımında belirsize düşmüş olabilir

    count_report = None
    if used_grid.get("low_detection") and expected_column_count:
        products = [d for d in detections if d.get("category") != empty_class_name]
        count_report = _count_based_report(products, expected_column_count)

    return {
        "shelf_number": shelf_number,
        "column_count": used_grid["column_count"],
        "grid": used_grid,
        "grid_source": used_grid.get("grid_source"),
        "grid_status": assigned["grid_status"],
        "confident": used_grid.get("confident", False),
        "slots": assigned["slots"],
        "alignment_suspects": assigned["alignment_suspects"],
        "count_report": count_report,
    }


def analyze_slots(
    prediction_result: dict[str, Any],
    image: Any | None = None,
    shelf_number: int | None = None,
    column_count: int | None = None,
    corners_override: Any | None = None,
    config_path: str = "config/config.yaml",
) -> dict[str, Any]:
    """Config + görüntüden ``predict_shelf`` çıktısı için slot raporu üretir.

    Args:
        prediction_result: ``predictor.predict_shelf()`` çıktısı.
        image: tespitlerin yapıldığı BGR görüntü — tepsi köşe çapası için.
        shelf_number: raf numarası; None ise config ``default_shelf_number``.
        column_count: sütun sayısı; None ise config ``column_count``.
        corners_override: operatörün elle işaretlediği 4 köşe (belirsiz durumda).
        config_path: config.yaml yolu.
    """
    cfg = _load_config(config_path)
    slot_cfg = cfg.get("slot_analysis", {})

    if shelf_number is None:
        shelf_number = int(slot_cfg.get("default_shelf_number", DEFAULT_SHELF_NUMBER))
    if column_count is None:
        column_count = int(slot_cfg.get("column_count", DEFAULT_COLUMN_COUNT))

    return build_slot_report(
        prediction_result.get("detections", []),
        shelf_number=shelf_number,
        column_count=column_count,
        image=image,
        corners_override=corners_override,
        empty_class_name=EMPTY_CLASS_NAME,
    )
