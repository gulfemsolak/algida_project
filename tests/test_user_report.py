"""Tests for dashboard/export/user_report.py — end-user JSON rapor şeması."""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.export.user_report import (
    batch_export_filename,
    batch_urun_dagilimi_rows,
    build_batch_report,
    build_error_report,
    build_user_report,
    resolve_model_version,
    single_export_filename,
)

_METADATA = {
    "dosya": "WhatsApp Image 2026-07-09 at 20.52.04.jpeg",
    "tarih": "2026-07-27T20:42:37+03:00",
    "raf_numarasi": 1,
    "qr_okundu": True,
    "model_versiyonu": "best_20260727",
}

# Backend'in gerçek slot_report/shelf_report şekliyle (bkz. slot_assigner.py /
# shelf_analyzer.py) uyumlu, minimal fixture'lar.
_GRID_OK = {
    "grid_status": "ok",
    "grid_source": "zincir",
    "low_detection": False,
    "H_inv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "boundary_lines": [((0, 0), (1, 1))],
    "pitch": 123.4,
    "lane_meta": {"comb_score": 0.9},
}

_SLOTS_OK = [
    {
        "slot_id": "11", "shelf_no": 1, "column_no": 1, "is_empty": False,
        "product_name": "cornetto_fistik", "count": 4, "confidence": 0.91,
        "breakdown": [{"category": "cornetto_fistik", "count": 4}],
        "x": 10, "x_left": 0, "x_right": 20, "alignment_suspect": False,
    },
    {
        "slot_id": "12", "shelf_no": 1, "column_no": 2, "is_empty": False,
        "product_name": "magnum_klasik", "count": 2, "confidence": 0.85,
        "breakdown": [
            {"category": "magnum_klasik", "count": 2},
            {"category": "magnum_badem", "count": 1},
        ],
        "x": 30, "x_left": 20, "x_right": 40, "alignment_suspect": True,
    },
    {
        "slot_id": "13", "shelf_no": 1, "column_no": 3, "is_empty": True,
        "product_name": None, "count": 0, "confidence": 0.0, "breakdown": [],
        "x": 50, "x_left": 40, "x_right": 60, "alignment_suspect": False,
    },
]

_SLOT_REPORT_OK = {
    "shelf_number": 1, "column_count": 3, "grid": _GRID_OK, "grid_source": "zincir",
    "grid_status": "ok", "confident": True, "slots": _SLOTS_OK,
    "alignment_suspects": [], "count_report": None,
}

_SHELF_REPORT_OK = {
    "total_rows": 1, "total_slots": 3, "total_products": 7, "total_empty": 1,
    "shelf_fill_rate": 0.6667, "lattice_available": True, "restock_needed": False,
    "rows_needing_restock": [], "rows": {}, "shelf_map": {}, "slot_report": _SLOT_REPORT_OK,
    "anomalies": [], "overall_confidence": 0.88,
}

_GRID_LOW_DET = {"grid_status": "belirsiz", "low_detection": True}
_SLOT_REPORT_LOW_DET = {
    "shelf_number": 1, "column_count": 7, "grid": _GRID_LOW_DET, "grid_source": None,
    "grid_status": "belirsiz", "confident": False, "slots": [],
    "alignment_suspects": [], "count_report": None,
}
_SHELF_REPORT_LOW_DET = {
    "total_rows": 0, "total_slots": 2, "total_products": 2, "total_empty": 0,
    "shelf_fill_rate": 1.0, "lattice_available": False, "restock_needed": False,
    "rows_needing_restock": [], "rows": {}, "shelf_map": {}, "slot_report": _SLOT_REPORT_LOW_DET,
    "anomalies": [], "overall_confidence": 0.55,
}
_SUMMARY_LOW_DET = {
    "cornetto_fistik": {"count": 2, "avg_confidence": 0.55},
    "empty_slot": {"count": 3, "avg_confidence": 0.2},
}

_INTERNAL_KEYS = ["H_inv", "boundary_lines", "lane_meta", "pitch", "axis_u", "comb_score", "theta"]


def _snake_ascii(s: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_]+", s))


# ── analiz_tamamlandi ─────────────────────────────────────────────────────────
def test_build_user_report_completed_has_full_schema():
    report = build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, _METADATA)

    assert report["analiz"] == _METADATA
    assert report["ozet"]["durum"] == "analiz_tamamlandi"
    assert report["ozet"]["doluluk_orani_yuzde"] == round(0.6667 * 100, 2)
    assert report["ozet"]["toplam_kanal"] == 3
    assert report["ozet"]["dolu_kanal"] == 2
    assert report["ozet"]["bos_kanal"] == 1
    assert report["ozet"]["yenilenmeli"] is False

    assert len(report["kanallar"]) == 3
    dolu = [k for k in report["kanallar"] if k["durum"] == "dolu"]
    bos = [k for k in report["kanallar"] if k["durum"] == "urun_tespit_edilemedi"]
    assert len(dolu) == 2 and len(bos) == 1
    assert bos[0]["kanal_kodu"] == "13"
    assert bos[0]["urunler"] == []

    mixed_slot = next(k for k in report["kanallar"] if k["kanal_kodu"] == "12")
    assert mixed_slot["urunler"] == [
        {"ad": "magnum_klasik", "adet": 2},
        {"ad": "magnum_badem", "adet": 1},
    ]

    urun_dagilimi = {u["ad"]: u for u in report["urun_dagilimi"]}
    assert urun_dagilimi["cornetto_fistik"] == {"ad": "cornetto_fistik", "toplam_adet": 4, "kanal_sayisi": 1}
    assert urun_dagilimi["magnum_klasik"] == {"ad": "magnum_klasik", "toplam_adet": 2, "kanal_sayisi": 1}
    assert urun_dagilimi["magnum_badem"] == {"ad": "magnum_badem", "toplam_adet": 1, "kanal_sayisi": 1}


# ── manuel_giris_gerekli ──────────────────────────────────────────────────────
def test_build_user_report_manual_entry_omits_channel_fields():
    report = build_user_report(
        _SLOT_REPORT_LOW_DET, _SHELF_REPORT_LOW_DET, _METADATA,
        detection_summary=_SUMMARY_LOW_DET,
    )

    assert report["ozet"]["durum"] == "manuel_giris_gerekli"
    assert "aciklama" in report["ozet"]

    # Belirsiz bilgi kesin sayıymış gibi verilmesin: bu field'lar hiç OLMASIN.
    assert "kanallar" not in report
    for forbidden in ("doluluk_orani_yuzde", "toplam_kanal", "dolu_kanal", "bos_kanal", "yenilenmeli"):
        assert forbidden not in report["ozet"]

    # urun_dagilimi ham tespit sayımından geliyor, empty_slot filtrelenmiş,
    # kanal_sayisi YOK (kanal bilgisi zaten yok).
    assert report["urun_dagilimi"] == [{"ad": "cornetto_fistik", "toplam_adet": 2}]
    for item in report["urun_dagilimi"]:
        assert "kanal_sayisi" not in item


def test_build_user_report_manual_entry_without_summary_is_empty_list():
    report = build_user_report(_SLOT_REPORT_LOW_DET, _SHELF_REPORT_LOW_DET, _METADATA)
    assert report["urun_dagilimi"] == []


# ── analiz_basarisiz ──────────────────────────────────────────────────────────
def test_build_error_report():
    report = build_error_report(_METADATA, "Model yüklenemedi: models/best.pt")
    assert report["ozet"]["durum"] == "analiz_basarisiz"
    assert report["ozet"]["hata_mesaji"] == "Model yüklenemedi: models/best.pt"
    assert report["analiz"] == _METADATA


# ── Field adı kuralları ───────────────────────────────────────────────────────
def test_field_names_are_ascii_snake_case():
    report = build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, _METADATA)

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk_keys(item)

    for key in _walk_keys(report):
        assert _snake_ascii(key), f"field adı ASCII/snake_case değil: {key!r}"


def test_timestamp_is_iso8601():
    # metadata çağıran sayfa tarafından üretiliyor — burada üretim şeklini test ediyoruz.
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    assert datetime.fromisoformat(ts) is not None
    assert re.search(r"[+-]\d{2}:\d{2}$", ts)


# ── Internal/debug alanlar sızmamalı ──────────────────────────────────────────
def test_internal_debug_fields_not_leaked():
    report = build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, _METADATA)
    dumped = json.dumps(report, ensure_ascii=False)
    for key in _INTERNAL_KEYS:
        assert key not in dumped, f"internal alan sızdı: {key}"


def test_internal_debug_fields_not_leaked_batch():
    batch = build_batch_report(
        metadata={"tarih": "2026-07-27T20:42:37+03:00", "gorsel_sayisi": 1, "model_versiyonu": "best_20260727"},
        toplam_ozet={"islenen_gorsel": 1},
        gorseller=[build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, _METADATA)],
    )
    dumped = json.dumps(batch, ensure_ascii=False)
    for key in _INTERNAL_KEYS:
        assert key not in dumped


# ── Model versiyonu / dosya adı yardımcıları ──────────────────────────────────
def test_resolve_model_version_no_path_returns_demo():
    assert resolve_model_version(None) == "demo"


def test_resolve_model_version_uses_stem_and_mtime(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    version = resolve_model_version(weights)
    assert version.startswith("best_")
    assert re.fullmatch(r"best_\d{8}", version)


def test_single_export_filename_is_ascii_and_clean():
    now = datetime(2026, 7, 27, 20, 42)
    name = single_export_filename("WhatsApp Image 2026-07-09 at 20.52.04.jpeg", now=now)
    assert name.startswith("raf_analizi_")
    assert name.endswith("_20260727_2042.json")
    assert name.encode("ascii")  # ASCII-only, hata atmaz


def test_batch_export_filename():
    now = datetime(2026, 7, 27, 20, 42)
    assert batch_export_filename(5, now=now) == "toplu_analiz_5gorsel_20260727_2042.json"


# ── build_batch_report saf sarma — pass-through fidelity ──────────────────────
def test_build_batch_report_wraps_without_recomputing():
    toplam_ozet = {
        "islenen_gorsel": 5, "toplam_urun": 185, "ortalama_doluluk_yuzde": 82.4,
        "manuel_giris_bekleyen": 1, "yenilenmeli_sayisi": 2,
        "ortalama_hesabina_dahil_gorsel": 4,
        "aciklama": "1 görsel manuel giriş beklediği için ortalama hesabına dahil değil.",
    }
    gorseller = [build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, _METADATA)]
    batch = build_batch_report(
        metadata={"tarih": "t", "gorsel_sayisi": 5, "model_versiyonu": "best_20260727"},
        toplam_ozet=toplam_ozet,
        gorseller=gorseller,
    )
    # Aggregate'ler DEĞİŞTİRİLMEDEN geçmiş olmalı — UI'daki hesapla JSON'daki
    # sayının birebir aynı kalması bu fonksiyonun hiçbir şeyi yeniden
    # hesaplamamasına bağlı.
    assert batch["toplam_ozet"] == toplam_ozet
    assert batch["gorseller"] == gorseller
    assert batch["analiz_grubu"]["gorsel_sayisi"] == 5


# ── CSV kaynağı — ürün bazında, JSON'la aynı kaynaktan ────────────────────────
def test_batch_urun_dagilimi_rows_flattens_with_filename():
    metadata_a = {**_METADATA, "dosya": "raf_a.jpg"}
    metadata_b = {**_METADATA, "dosya": "raf_b.jpg"}
    gorseller = [
        build_user_report(_SLOT_REPORT_OK, _SHELF_REPORT_OK, metadata_a),
        build_user_report(
            _SLOT_REPORT_LOW_DET, _SHELF_REPORT_LOW_DET, metadata_b,
            detection_summary=_SUMMARY_LOW_DET,
        ),
    ]
    rows = batch_urun_dagilimi_rows(gorseller)

    a_rows = [r for r in rows if r["dosya"] == "raf_a.jpg"]
    b_rows = [r for r in rows if r["dosya"] == "raf_b.jpg"]
    assert len(a_rows) == 3  # cornetto_fistik, magnum_klasik, magnum_badem
    assert all("kanal_sayisi" in r for r in a_rows)

    assert len(b_rows) == 1  # manuel_giris_gerekli — kanal_sayisi yok
    assert b_rows[0]["ad"] == "cornetto_fistik"
    assert "kanal_sayisi" not in b_rows[0]


def test_batch_urun_dagilimi_rows_skips_error_photos():
    error_report = build_error_report(_METADATA, "Görsel çözümlenemedi.")
    rows = batch_urun_dagilimi_rows([error_report])
    assert rows == []
