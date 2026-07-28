"""Paginated, filterable Recent Analyses list for the Dashboard — SQLite-backed
(see dashboard.db). Dense, single-line rows in the Linear/Vercel list style."""
from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import math
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.components.widgets import qr_status_badge
from dashboard.db import clear_all, delete_analysis, get_analyses, get_total_count
from dashboard.theme import fill_rate_color

PER_PAGE = 10


def _relative_time(ts: str | _dt.datetime) -> str:
    if isinstance(ts, str):
        try:
            ts = _dt.datetime.fromisoformat(ts)
        except ValueError:
            return ts
    secs = (_dt.datetime.now() - ts).total_seconds()
    if secs < 60:
        return "az önce"
    if secs < 3600:
        return f"{int(secs // 60)} dk önce"
    if ts.date() == _dt.date.today():
        return ts.strftime("%H:%M")
    return ts.strftime("%d.%m, %H:%M")


def _status_color(fill_rate: float) -> str:
    return fill_rate_color(fill_rate)


def _thumb_b64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return None


def _row_qr_info(entry: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a qr_reader-shaped dict from the flat SQLite columns
    (qr_verified is stored as NULL/0/1 — map back to None/False/True)."""
    raw_verified = entry.get("qr_verified")
    verified = None if raw_verified is None else bool(raw_verified)
    return {
        "qr_found": bool(entry.get("qr_found")),
        "qr_values": entry.get("qr_values") or [],
        "verified": verified,
        "message": entry.get("qr_message") or "",
    }


def _analysis_row(entry: dict[str, Any]) -> None:
    fill_rate = entry.get("fill_rate") or 0.0
    color = _status_color(fill_rate)
    filename = _html.escape(entry.get("filename") or "raf fotoğrafı")
    time_label = _relative_time(entry["timestamp"])

    thumb_b64 = _thumb_b64(entry.get("thumbnail_path"))
    thumb_html = (
        f'<img class="analysis-row-thumb" src="data:image/jpeg;base64,{thumb_b64}" alt="" />'
        if thumb_b64 else '<div class="analysis-row-thumb analysis-row-thumb-empty"></div>'
    )
    qr_badge_html = qr_status_badge(_row_qr_info(entry))

    row_col, del_col = st.columns([40, 1])
    with row_col:
        st.markdown(
            f"""
            <div class="analysis-row">
                <div class="analysis-row-stripe" style="background:{color};"></div>
                <div class="analysis-row-body">
                    {thumb_html}
                    <div class="analysis-row-meta">
                        <span class="analysis-row-filename">{filename}</span>
                        <span class="analysis-row-time">{time_label}</span>
                    </div>
                    {qr_badge_html}
                    <div class="analysis-row-metrics">
                        <div class="analysis-row-metric">
                            <span class="analysis-row-metric-value" style="color:{color};">{fill_rate:.0%}</span>
                            <span class="analysis-row-metric-label">Doluluk</span>
                        </div>
                        <div class="analysis-row-metric">
                            <span class="analysis-row-metric-value">{entry.get('total_products', 0)}</span>
                            <span class="analysis-row-metric-label">Ürün</span>
                        </div>
                        <div class="analysis-row-metric">
                            <span class="analysis-row-metric-value">{entry.get('empty_slots', 0)}</span>
                            <span class="analysis-row-metric-label">Boş</span>
                        </div>
                        <div class="analysis-row-metric">
                            <span class="analysis-row-metric-value">{(entry.get('avg_confidence') or 0):.0%}</span>
                            <span class="analysis-row-metric-label">Güven</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with del_col:
        with st.popover("✕"):
            st.markdown(f"**{filename}** silinsin mi?")
            if st.button("Sil", key=f"confirm_del_{entry['id']}", type="primary"):
                delete_analysis(entry["id"])
                st.rerun()


def _reset_filters() -> None:
    st.session_state["analyses_search"] = ""
    st.session_state["analyses_date_from"] = None
    st.session_state["analyses_date_to"] = None
    st.session_state["analyses_page"] = 1


def recent_analyses_section() -> None:
    st.session_state.setdefault("analyses_page", 1)

    # ── Filters row ──────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
    with f1:
        search = st.text_input(
            "Dosya adı ara", placeholder="Dosya adı ara…",
            label_visibility="collapsed", key="analyses_search",
        )
    with f2:
        date_from = st.date_input(
            "Başlangıç", value=None, label_visibility="collapsed", key="analyses_date_from",
        )
    with f3:
        date_to = st.date_input(
            "Bitiş", value=None, label_visibility="collapsed", key="analyses_date_to",
        )
    with f4:
        st.button("Sıfırla", use_container_width=True, on_click=_reset_filters)

    filters_key = (search, date_from, date_to)
    if st.session_state.get("_analyses_filters_key") != filters_key:
        st.session_state["_analyses_filters_key"] = filters_key
        st.session_state["analyses_page"] = 1

    total = get_total_count(search_query=search or None, date_from=date_from, date_to=date_to)

    if total == 0:
        st.markdown(
            '<div style="text-align:center;color:#46536E;font-size:0.85rem;padding:28px 0;">'
            "Henüz analiz yok — sonuçları burada görmek için bir raf analizi çalıştırın."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    total_pages = max(1, math.ceil(total / PER_PAGE))
    page = min(max(1, st.session_state["analyses_page"]), total_pages)
    st.session_state["analyses_page"] = page

    entries = get_analyses(
        page=page, per_page=PER_PAGE,
        search_query=search or None, date_from=date_from, date_to=date_to,
    )

    start = (page - 1) * PER_PAGE + 1
    end = min(page * PER_PAGE, total)
    st.markdown(
        f'<div class="analyses-summary-line">{total} analizden {start}–{end} arası gösteriliyor</div>',
        unsafe_allow_html=True,
    )

    for entry in entries:
        _analysis_row(entry)

    # ── Pagination controls ────────────────────────────────────────────────
    p1, p2, p3 = st.columns([1, 2, 1])
    with p1:
        if st.button("← Önceki", disabled=page <= 1, use_container_width=True):
            st.session_state["analyses_page"] = page - 1
            st.rerun()
    with p2:
        st.markdown(
            f'<div class="analyses-page-label">Sayfa {page} / {total_pages}</div>',
            unsafe_allow_html=True,
        )
    with p3:
        if st.button("Sonraki →", disabled=page >= total_pages, use_container_width=True):
            st.session_state["analyses_page"] = page + 1
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    with st.popover("Tüm geçmişi temizle"):
        st.markdown("**Tüm** analiz geçmişi silinsin mi? Bu işlem geri alınamaz.")
        if st.button("Tümünü sil", type="primary", key="confirm_clear_all"):
            clear_all()
            st.session_state["analyses_page"] = 1
            st.rerun()
