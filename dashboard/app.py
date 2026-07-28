"""Streamlit dashboard entry point — Genel Bakış (home/overview) page."""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import yaml

from dashboard.components.analyses_list import recent_analyses_section
from dashboard.components.sidebar import render_sidebar
from dashboard.components.widgets import product_catalog
from dashboard.db import get_aggregate_stats, init_db
from dashboard.theme import apply_theme, page_header, section_header, status_line

_CFG_PATH = Path("config/config.yaml")


def _load_config() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def main():
    cfg = _load_config()
    dash_cfg = cfg.get("dashboard", {})

    st.set_page_config(
        page_title="IceVision",
        layout=dash_cfg.get("layout", "wide"),
        initial_sidebar_state="expanded",
    )

    init_db()
    apply_theme()
    render_sidebar(cfg)

    page_header("Genel Bakış")

    model_path = st.session_state.get("model_path")
    if model_path:
        status_line(f"HAZIR · MODEL: {Path(model_path).name.upper()}")
    else:
        status_line("MODEL YÜKLENMEDİ", ok=False)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    stats = get_aggregate_stats()
    if stats["count"] == 0:
        st.caption("Henüz analiz yok — metrikleri görmek için bir raf analizi çalıştırın.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Analiz", stats["count"])
        m2.metric("Ort. Doluluk", f"{stats['avg_fill_rate']:.0%}")
        m3.metric("Tespit Edilen Ürün", stats["products_detected"])
        m4.metric("Uyarı", stats["alerts"])

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Recent analyses ─────────────────────────────────────────────────────
    section_header("Son Analizler")
    recent_analyses_section()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Quick actions ────────────────────────────────────────────────────────
    section_header("Hızlı İşlemler")
    qa1, qa2 = st.columns(2)
    with qa1:
        with st.container(border=True):
            st.markdown('<div class="quick-action-title">Raf analizi</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="quick-action-desc">Tek bir raf fotoğrafı yükleyin, anında ürün tespiti ve doluluk skoru alın.</div>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/01_shelf_analysis.py", label="Aç →")
    with qa2:
        with st.container(border=True):
            st.markdown('<div class="quick-action-title">Toplu analiz</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="quick-action-desc">Bir rotanın tamamını tek seferde analiz edin — birden fazla raf fotoğrafı yükleyin.</div>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/02_batch_analysis.py", label="Aç →")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Supported products ───────────────────────────────────────────────────
    # empty_slot bir "ürün" değil — katalogdan gizlenir (config.yaml'daki 35'lik
    # class listesi/index'ler değişmiyor, yalnız bu görünürlükten çıkarılıyor).
    classes: list[str] = [c for c in cfg.get("classes", []) if c != "empty_slot"]
    section_header(f"Desteklenen Ürünler · {len(classes)}" if classes else "Desteklenen Ürünler")
    product_catalog(classes)


if __name__ == "__main__":
    main()
