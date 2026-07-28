"""Reusable HTML/Streamlit UI blocks built on top of the IceVision theme."""
from __future__ import annotations

import html as _html
from typing import Any

import streamlit as st

from dashboard.theme import (
    AMBER,
    ICE,
    MINT,
    ALGIDA_RED,
    QR_WARN,
    format_class_name,
    infer_brand,
    pill,
    product_color,
)


# ── Slot detay tablosu (Raf Detayı — Raf Analizi sayfası) ─────────────────────
def slot_detail_table(slot_report: dict[str, Any]) -> None:
    """Slotları soldan sağa, hedef formatta listeler: "1. raf 2. slot → 4 Magnum Klasik".

    Boş slotlar "ürün tespit edilemedi" ile işaretlenir ("BOŞ" DEĞİL — bazı ürün
    sınıflarında recall düşük olduğundan gerçekten boş slot ile modelin göremediği
    dolu slot ayırt edilemez; ifade bu belirsizliği yansıtır). Adet ürün adının
    parçasıdır, ayrı bir rozet değil. ``product-row`` / ``product-dot`` /
    ``conf-badge`` stilleri yeniden kullanılır — yeni CSS eklenmez.
    """
    slots = slot_report.get("slots", [])
    if not slots:
        st.caption("Slot verisi üretilemedi.")
        if slot_report.get("grid", {}).get("low_detection"):
            st.caption("Manuel giriş için soldaki formu kullanın: toplam kanal sayısı ya da tepsi köşeleri.")
        return

    rows_html = []
    for s in slots:
        if s["is_empty"]:
            dot = '<span class="product-dot" style="background:#2E3A55;"></span>'
            text = f'{s["shelf_no"]}. raf {s["column_no"]}. slotta ürün tespit edilemedi'
            value = f'<span class="product-name" style="color:#8195B5;">{text}</span>'
            right = ""
        else:
            color = product_color(s["product_name"])
            dot = f'<span class="product-dot" style="background:{color};"></span>'
            # Baskın sınıf öne çıkar; ikincil sınıflar aynı satırda düşük vurgulu listelenir
            # (karışık slotta bilgi gizlenmesin — tablo toplamı = tespit sayısı).
            breakdown = s.get("breakdown") or [{"category": s["product_name"], "count": s["count"]}]
            dom = breakdown[0]
            text = (f'{s["shelf_no"]}. raf {s["column_no"]}. slotta '
                    f'{dom["count"]} {format_class_name(dom["category"])}')
            if len(breakdown) > 1:
                extra = ", ".join(
                    f'{b["count"]} {format_class_name(b["category"])}' for b in breakdown[1:]
                )
                text += (f' <span style="color:#8195B5;font-size:0.72rem;font-weight:400;">'
                         f'+ {extra}</span>')
            # Hizalama şüphesi: tespitler yanlış kolona düşmüş olabilir — düşük vurgulu,
            # mevcut renk diline uygun (ikon/emoji yok).
            suspect = (
                f' <span style="color:{AMBER};font-size:0.72rem;font-weight:400;">'
                f'· hizalama şüpheli</span>'
                if s.get("alignment_suspect") else ""
            )
            value = f'<span class="product-name">{text}{suspect}</span>'
            right = f'<span class="conf-badge">{round(s["confidence"] * 100)}%</span>'

        rows_html.append(
            f'<div class="product-row"><div class="product-row-top">'
            f'<div class="product-row-left">'
            f'<span class="conf-badge">{s["slot_id"]}</span>'
            f'{dot}{value}'
            f"</div>"
            f'<div class="product-row-right">{right}</div>'
            f"</div></div>"
        )

    st.markdown("".join(rows_html), unsafe_allow_html=True)


# ── Product breakdown row (Shelf Analysis results panel) ───────────────────────
def product_breakdown_row(category: str, count: int, avg_confidence: float) -> None:
    color = product_color(category)
    conf_pct = round(avg_confidence * 100)
    count_color = AMBER if category == "empty_slot" else "#EDF3FB"
    st.markdown(
        f"""
        <div class="product-row">
            <div class="product-row-top">
                <div class="product-row-left">
                    <span class="product-dot" style="background:{color};"></span>
                    <span class="product-name">{format_class_name(category)}</span>
                </div>
                <div class="product-row-right">
                    <span class="product-count" style="color:{count_color};">{count}</span>
                    <span class="conf-badge">{conf_pct}%</span>
                </div>
            </div>
            <div class="mini-bar-track">
                <div class="mini-bar-fill" style="width:{conf_pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── QR status badge (Shelf Analysis results, Recent Analyses rows) ─────────────
def qr_status_badge(qr_info: dict[str, Any] | None) -> str:
    """HTML for a QR verification status badge. Render with
    ``st.markdown(qr_status_badge(...), unsafe_allow_html=True)``."""
    qr_info = qr_info or {}

    if qr_info.get("qr_found"):
        values = ", ".join(qr_info.get("qr_values", []))
        verified = qr_info.get("verified")
        if verified is True:
            color = MINT
            text = f"QR: {values}"
        elif verified is False:
            color = ALGIDA_RED
            text = qr_info.get("message") or f"QR uyumsuz: {values}"
        else:
            color = ICE
            text = f"QR: {values}"
    else:
        color = QR_WARN
        text = "QR tespit edilemedi"

    text = _html.escape(text)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;'
        f"border-radius:6px;font-size:0.75rem;font-weight:500;"
        f"font-family:'SF Mono','Fira Code',monospace;color:{color};"
        f'background:{color}15;border:1px solid {color}30;">'
        f'<span style="color:{color};">&#9679;</span>{text}</span>'
    )


# ── Supported products catalog (Dashboard home page) ────────────────────────────
def product_catalog(categories: list[str]) -> None:
    """Brand-grouped chip catalog — one bordered panel, chips wrap per brand row."""
    if not categories:
        st.caption("Tanımlı ürün sınıfı yok.")
        return

    groups: dict[str, list[str]] = {}
    for cat in categories:
        groups.setdefault(infer_brand(cat), []).append(cat)

    brand_order = sorted(b for b in groups if b != "Other")
    if "Other" in groups:
        brand_order.append("Other")

    with st.container(border=True):
        for brand in brand_order:
            cats = sorted(groups[brand])
            chips = []
            for cat in cats:
                is_empty = cat == "empty_slot"
                chip_cls = "product-chip product-chip-empty" if is_empty else "product-chip"
                dot_color = AMBER if is_empty else product_color(cat)
                chips.append(
                    f'<span class="{chip_cls}">'
                    f'<span class="product-chip-dot" style="background:{dot_color};"></span>'
                    f'{_html.escape(format_class_name(cat))}'
                    f"</span>"
                )
            st.markdown(
                f'<div class="product-brand-row">'
                f'<span class="product-brand-label">{_html.escape(brand)}</span>'
                f'<span class="product-brand-count">{len(cats)}</span>'
                f"</div>"
                f'<div class="product-chip-row">{"".join(chips)}</div>',
                unsafe_allow_html=True,
            )


# ── Priority pill for restock table ─────────────────────────────────────────────
def priority_pill(priority: str) -> str:
    mapping = {
        "KRİTİK": ALGIDA_RED,
        "AZALDI": AMBER,
        "YETERLİ": MINT,
    }
    return pill(priority, mapping.get(priority, AMBER))


# ── Shelf map grid (Restock Planner) ────────────────────────────────────────────
def shelf_map_grid(shelf_map: dict[str, list[dict[str, Any]]]) -> None:
    """Render each detected row as row-label + a strip of small rectangular slot cells."""
    if not shelf_map:
        st.markdown('<span style="color:#8195B5;font-size:0.85rem;">Raf sırası tespit edilemedi.</span>', unsafe_allow_html=True)
        return

    for i, row_id in enumerate(sorted(shelf_map.keys(), key=lambda r: int(r.split("_")[-1]))):
        slots = shelf_map[row_id]
        cells = []
        for slot in slots:
            cat = slot["category"]
            if cat == "empty_slot":
                cells.append('<div class="shelf-cell shelf-cell-empty"></div>')
            else:
                color = product_color(cat)
                cells.append(f'<div class="shelf-cell" style="background:{color};" title="{format_class_name(cat)}"></div>')

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <div class="shelf-row-label">Sıra {i + 1}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(cells)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def shelf_map_legend(categories: list[str]) -> None:
    items = "".join(
        f'<span class="legend-item">'
        f'<span class="product-dot" style="background:{product_color(c)};"></span>'
        f'{format_class_name(c)}</span>'
        for c in categories
    )
    st.markdown(f'<div style="margin-top:10px;">{items}</div>', unsafe_allow_html=True)
