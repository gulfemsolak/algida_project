"""Sayfa 3 — Dolum Planlama: stok durumu ve dolum önerileri."""
import io
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from dashboard.components.sidebar import render_sidebar
from dashboard.components.widgets import priority_pill, shelf_map_grid, shelf_map_legend
from dashboard.db import init_db, save_analysis, save_thumbnail
from dashboard.theme import apply_theme, format_class_name, page_header
from src.analysis.shelf_analyzer import analyze_shelf
from src.models.predictor import predict_shelf


def _load_config() -> dict:
    p = Path("config/config.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compute_restock_table(shelf_report: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a per-category restock priority list.

    Each category's capacity is estimated from its "home" row — the shelf row
    where it has the most detections — using that row's total slot count
    (products + empty) as a stand-in for the lane's full capacity.
    """
    rows = shelf_report.get("rows", {})

    home_row_count: dict[str, int] = {}
    home_row_capacity: dict[str, int] = {}
    for stats in rows.values():
        for cat, cnt in stats.get("category_counts", {}).items():
            if cat == "empty_slot":
                continue
            if cnt > home_row_count.get(cat, 0):
                home_row_count[cat] = cnt
                home_row_capacity[cat] = stats["total_slots"]

    entries = []
    for cat, data in summary.items():
        if cat == "empty_slot":
            continue
        count = data["count"]
        capacity = max(home_row_capacity.get(cat, count), count, 1)
        fill_pct = count / capacity

        if fill_pct < 0.25 or count <= 1:
            priority = "KRİTİK"
        elif fill_pct <= 0.50:
            priority = "AZALDI"
        else:
            priority = "YETERLİ"

        entries.append({
            "category": cat,
            "count": count,
            "capacity": capacity,
            "fill_pct": fill_pct,
            "priority": priority,
            "units_needed": max(capacity - count, 0),
        })

    order = {"KRİTİK": 0, "AZALDI": 1, "YETERLİ": 2}
    entries.sort(key=lambda e: (order[e["priority"]], e["fill_pct"]))
    return entries


def run():
    cfg = _load_config()
    st.set_page_config(page_title="Dolum Planlama — IceVision", layout="wide", initial_sidebar_state="expanded")
    init_db()
    apply_theme()
    render_sidebar(cfg)

    page_header("Dolum Planlama")

    uploaded = st.file_uploader(
        "Raf fotoğrafı yükleyin veya aşağıda son analiz edilen görseli kullanın",
        type=["jpg", "jpeg", "png"],
    )

    conf_threshold = st.session_state.get("conf_threshold", 0.5)
    model_path = st.session_state.get("model_path")
    model = st.session_state.get("model")

    result = None
    shelf_report = None
    display_bgr = None

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_bgr is None:
            st.error("Görsel okunamadı. Geçerli bir JPG veya PNG dosyası yükleyin.")
            return
        with st.spinner("Tespit çalışıyor..."):
            try:
                result = predict_shelf(
                    image_path=original_bgr,
                    model_path=model_path,
                    model=model,
                    conf_threshold=conf_threshold,
                    config_path="config/config.yaml",
                )
            except Exception as exc:
                st.error(f"Tespit başarısız: {exc}")
                return
        shelf_report = analyze_shelf(result, config_path="config/config.yaml", image=original_bgr)
        save_analysis({
            "filename": uploaded.name,
            "total_products": result.get("total_products", 0),
            "empty_slots": shelf_report.get("total_empty", 0),
            "fill_rate": shelf_report.get("shelf_fill_rate", 0.0),
            "avg_confidence": result.get("overall_confidence", 0.0),
            "shelf_rows": shelf_report.get("total_rows", 0),
            "class_counts": {cat: d.get("count", 0) for cat, d in result.get("summary", {}).items()},
            "thumbnail_path": save_thumbnail(original_bgr),
        })
        display_bgr = original_bgr
        st.session_state["last_result"] = result
        st.session_state["last_shelf_report"] = shelf_report
        st.session_state["last_image_bgr"] = original_bgr
    elif st.session_state.get("last_result") is not None:
        result = st.session_state["last_result"]
        shelf_report = st.session_state["last_shelf_report"]
        display_bgr = st.session_state["last_image_bgr"]
        st.caption("En son analiz edilen raf görseli gösteriliyor.")
    else:
        st.caption("Dolum planı için bir raf fotoğrafı yükleyin veya önce Raf Analizi sayfasında analiz yapın.")
        return

    # ── Shelf status overview ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Raf Durumu</div>', unsafe_allow_html=True)
    col_img, col_map = st.columns([1, 1.3])
    with col_img:
        annotated = result.get("annotated_image")
        st.image(_bgr_to_rgb(annotated if annotated is not None else display_bgr), use_column_width=True)
    with col_map:
        shelf_map_grid(shelf_report.get("shelf_map", {}))
        present_categories = [c for c in result.get("summary", {}).keys()]
        shelf_map_legend(present_categories)

    st.markdown('<hr class="glow-divider"/>', unsafe_allow_html=True)

    # ── Restock priority list ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Dolum Öncelik Listesi</div>', unsafe_allow_html=True)
    summary = result.get("summary", {})
    table = _compute_restock_table(shelf_report, summary)

    if not table:
        st.caption("Ürün tespit edilemedi — planlanacak dolum yok.")
        return

    for entry in table:
        c1, c2, c3, c4, c5 = st.columns([1, 2, 1.3, 1.6, 1.6])
        c1.markdown(priority_pill(entry["priority"]), unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#EDF3FB;'>{format_class_name(entry['category'])}</span>", unsafe_allow_html=True)
        c3.markdown(
            f"<span style='font-family:\"SF Mono\",\"Fira Code\",monospace;color:#EDF3FB;'>"
            f"{entry['count']} / {entry['capacity']}</span>",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"<div><span style='color:#8195B5;font-size:0.85rem;'>{entry['fill_pct']:.0%}</span>"
            f"<div class='mini-bar-track' style='width:80px;'>"
            f"<div class='mini-bar-fill' style='width:{entry['fill_pct'] * 100:.0f}%;background:#5CB8FF;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        action = {
            "KRİTİK": "Hemen doldurun",
            "AZALDI": "Yakında doldurun",
            "YETERLİ": "İşlem gerekmiyor",
        }[entry["priority"]]
        c5.markdown(f"<span style='color:#8195B5;font-size:0.85rem;'>{action}</span>", unsafe_allow_html=True)

    st.markdown('<hr class="glow-divider"/>', unsafe_allow_html=True)

    # ── Restock summary ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dolum Özeti</div>', unsafe_allow_html=True)
    total_units = sum(e["units_needed"] for e in table)

    col_summary, col_breakdown = st.columns([1, 1.5])
    with col_summary:
        with st.container(border=True):
            st.markdown('<div class="section-header" style="margin-top:0;">Gereken Adet</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="units-needed-value">{total_units}</div>', unsafe_allow_html=True)

    with col_breakdown:
        needing = [e for e in table if e["units_needed"] > 0]
        if needing:
            for e in needing:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                    f"color:#8195B5;font-size:0.85rem;'>"
                    f"<span>{format_class_name(e['category'])}</span>"
                    f"<span style='font-family:\"SF Mono\",\"Fira Code\",monospace;color:#EDF3FB;'>+{e['units_needed']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Tüm ürünler yeterli stokta.")

    csv_df = pd.DataFrame([
        {
            "product": format_class_name(e["category"]),
            "current_count": e["count"],
            "capacity": e["capacity"],
            "units_needed": e["units_needed"],
        }
        for e in table
    ])
    csv_buf = io.StringIO()
    csv_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Dolum listesini indir",
        data=csv_buf.getvalue(),
        file_name="restock_list.csv",
        mime="text/csv",
    )


run()
