"""End-user (operatör/saha) okunabilir JSON rapor şeması.

Ham ``slot_report``/``shelf_report`` dict'lerini backend'ten hiç değiştirmeden
alır, yalnızca Türkçe/ASCII alan adlarıyla ve internal/debug alanları
(homografi matrisi, pitch/θ, lane_meta, bbox koordinatları, model class
index'leri) süzerek yeniden biçimlendirir. Analiz mantığına DOKUNMAZ — saf
sunum/serialization katmanı. Ham geliştirici export'u (``?debug=1`` arkasında)
bu modülden bağımsız, değişmeden yaşamaya devam eder.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

_TR_ASCII_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ç": "c", "Ç": "c",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ğ": "g", "Ğ": "g",
})


# ── Dosya adı yardımcıları ──────────────────────────────────────────────────
def _slugify(name: str) -> str:
    """Dosya adını ASCII, alt-çizgili, uzantısız bir gövdeye indirger."""
    stem = Path(name).stem.translate(_TR_ASCII_MAP)
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return stem or "gorsel"


def single_export_filename(source_filename: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"raf_analizi_{_slugify(source_filename)}_{now.strftime('%Y%m%d_%H%M')}.json"


def batch_export_filename(image_count: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"toplu_analiz_{image_count}gorsel_{now.strftime('%Y%m%d_%H%M')}.json"


# ── Model versiyonu ──────────────────────────────────────────────────────────
def resolve_model_version(model_path: str | Path | None) -> str:
    """Ağırlık dosyasının stem + değiştirme-tarihi (ör. ``best_20260727``).

    Kod tabanında semantik bir model versiyon string'i yok (bkz. keşif raporu)
    — tek iz bırakan bilgi ağırlık dosyasının mtime'ı: biri ``best.pt``'yi
    değiştirdiğinde tarih değişir, JSON'da geriye dönük bir işaret kalır.
    """
    if not model_path:
        return "demo"
    p = Path(model_path)
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d")
        return f"{p.stem}_{mtime}"
    except OSError:
        return p.stem


# ── Tekli rapor ──────────────────────────────────────────────────────────────
def _kanal_item(slot: dict[str, Any]) -> dict[str, Any]:
    if slot.get("is_empty"):
        return {"kanal_kodu": slot["slot_id"], "durum": "urun_tespit_edilemedi", "urunler": []}
    breakdown = slot.get("breakdown") or [{"category": slot["product_name"], "count": slot["count"]}]
    return {
        "kanal_kodu": slot["slot_id"],
        "durum": "dolu",
        "urunler": [{"ad": b["category"], "adet": b["count"]} for b in breakdown],
    }


def _urun_dagilimi_from_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kanal (slot) bazlı dağılım — toplam adet + görüldüğü kanal sayısı."""
    toplam: dict[str, int] = {}
    kanal_sayisi: dict[str, int] = {}
    for s in slots:
        if s.get("is_empty"):
            continue
        breakdown = s.get("breakdown") or [{"category": s["product_name"], "count": s["count"]}]
        for b in breakdown:
            ad = b["category"]
            toplam[ad] = toplam.get(ad, 0) + b["count"]
            kanal_sayisi[ad] = kanal_sayisi.get(ad, 0) + 1
    return [
        {"ad": ad, "toplam_adet": toplam[ad], "kanal_sayisi": kanal_sayisi[ad]}
        for ad in sorted(toplam, key=lambda a: -toplam[a])
    ]


def _urun_dagilimi_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Izgara kurulamadığında (manuel_giris_gerekli) ham tespit sayımı — kanal
    bilgisi yok, bu yüzden ``kanal_sayisi`` alanı hiç eklenmez (belirsiz bilgiyi
    kesin bir sayıymış gibi göstermemek için)."""
    items = [
        {"ad": ad, "toplam_adet": data.get("count", 0)}
        for ad, data in summary.items() if ad != "empty_slot"
    ]
    items.sort(key=lambda d: -d["toplam_adet"])
    return items


def build_user_report(
    slot_report: dict[str, Any] | None,
    shelf_report: dict[str, Any] | None,
    metadata: dict[str, Any],
    detection_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bir fotoğrafın end-user JSON raporunu üret.

    Args:
        slot_report: ``slot_assigner.build_slot_report`` çıktısı (veya
            ``shelf_report["slot_report"]``).
        shelf_report: ``shelf_analyzer.analyze_shelf`` çıktısı.
        metadata: ``{dosya, tarih, raf_numarasi, qr_okundu, model_versiyonu}`` —
            çağıran sayfa QR/model bağlamını bildiği için burada hazır geçirilir;
            bu modül QR/model detaylarını bilmez, sadece diziyor.
        detection_summary: ``predict_shelf`` çıktısındaki ``summary`` — YALNIZ
            ``lattice_available=False`` durumunda ham ürün dağılımı için
            kullanılır (slot yoksa kanal-bazlı sayım da yok).

    Returns:
        ``{"analiz": ..., "ozet": ..., "kanallar"?, "urun_dagilimi"}``.
        ``lattice_available=False`` ise ``kanallar`` ve doluluk sayıları HİÇ
        eklenmez (0 yazmak yerine anahtarın kendisi yok) — kanıtsız bir "boş"
        iddiası üretilmesin diye.
    """
    report: dict[str, Any] = {"analiz": dict(metadata)}

    lattice_available = bool(shelf_report and shelf_report.get("lattice_available"))

    if lattice_available:
        total_slots = shelf_report["total_slots"]
        total_empty = shelf_report["total_empty"]
        report["ozet"] = {
            "durum": "analiz_tamamlandi",
            "doluluk_orani_yuzde": round(shelf_report["shelf_fill_rate"] * 100, 2),
            "toplam_kanal": total_slots,
            "dolu_kanal": total_slots - total_empty,
            "bos_kanal": total_empty,
            "yenilenmeli": bool(shelf_report["restock_needed"]),
        }
        slots = (slot_report or {}).get("slots", [])
        report["kanallar"] = [_kanal_item(s) for s in slots]
        report["urun_dagilimi"] = _urun_dagilimi_from_slots(slots)
    else:
        report["ozet"] = {
            "durum": "manuel_giris_gerekli",
            "aciklama": "Otomatik ızgara kurulamadı, manuel giriş gerekli.",
        }
        report["urun_dagilimi"] = _urun_dagilimi_from_summary(detection_summary or {})

    return report


def build_error_report(metadata: dict[str, Any], hata_mesaji: str) -> dict[str, Any]:
    """Fotoğraf okunamadı / model yüklenemedi gibi tam başarısızlık durumu."""
    return {
        "analiz": dict(metadata),
        "ozet": {"durum": "analiz_basarisiz", "hata_mesaji": hata_mesaji},
    }


# ── Toplu (batch) rapor ───────────────────────────────────────────────────────
def batch_urun_dagilimi_rows(gorseller: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Toplu rapordaki her fotoğrafın ürün dağılımını ``dosya`` etiketiyle tek
    düz satır listesine indirger (CSV export için) — JSON'daki AYNI
    ``gorseller`` listesinden beslenir, ayrı bir hesap yolu açılmaz."""
    rows: list[dict[str, Any]] = []
    for gorsel in gorseller:
        dosya = gorsel.get("analiz", {}).get("dosya", "")
        for item in gorsel.get("urun_dagilimi", []):
            rows.append({"dosya": dosya, **item})
    return rows


def build_batch_report(
    metadata: dict[str, Any],
    toplam_ozet: dict[str, Any],
    gorseller: list[dict[str, Any]],
) -> dict[str, Any]:
    """Toplu analiz sarma yapısı.

    ``toplam_ozet`` kasıtlı olarak burada HESAPLANMAZ — çağıran sayfa (Toplu
    Analiz) zaten UI kartları için aynı aggregate'leri hesaplamış durumda;
    onları olduğu gibi geçirir ki JSON'daki sayılar ekranda görünenle
    birebir aynı kalsın (iki ayrı hesap yolu = tutarsızlık riski).
    """
    return {
        "analiz_grubu": dict(metadata),
        "toplam_ozet": dict(toplam_ozet),
        "gorseller": list(gorseller),
    }
