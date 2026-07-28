"""Sidebar: wordmark, text-only navigation, shelf verification, model status."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.models.predictor import load_model


@st.cache_resource(show_spinner="Model yükleniyor...")
def _cached_model(weights_path: str):
    """Process-wide cached YOLO load — keyed by weights path.

    Without this, ``predict_shelf(model_path=...)`` reloads the 52 MB
    checkpoint from disk on every Streamlit rerun (every widget interaction).
    """
    return load_model(weights_path)


def _resolve_model(cfg: dict[str, Any]) -> Path | None:
    """Return the configured weights path if it exists, else None.

    The model is named explicitly in ``config.inference.weights_path`` — no
    size-based auto-discovery. A missing file yields None so the sidebar can
    flag it; the pipeline (predictor.predict_shelf) then raises rather than
    running demo detections behind the operator's back.
    """
    weights_path = cfg.get("inference", {}).get("weights_path")
    if not weights_path:
        return None
    path = Path(weights_path)
    return path if path.exists() else None


def render_sidebar(cfg: dict[str, Any]) -> None:
    """Render the persistent sidebar and store selections in ``st.session_state``."""
    inference_cfg = cfg.get("inference", {})

    # Detection threshold comes from config — not a user-facing knob.
    st.session_state["conf_threshold"] = float(inference_cfg.get("conf_threshold", 0.5))

    model_path = _resolve_model(cfg)
    st.session_state["model_path"] = str(model_path) if model_path else None
    if model_path is not None:
        try:
            st.session_state["model"] = _cached_model(str(model_path))
        except RuntimeError:
            # Yükleme hatası burada yutulmaz — predict_shelf model=None,
            # model_path=str geçildiğinde aynı hatayı tekrar deneyip fırlatır,
            # sayfa kendi try/except'inde kullanıcıya gösterir.
            st.session_state["model"] = None
    else:
        st.session_state["model"] = None

    with st.sidebar:
        st.markdown(
            '<div class="brand-mark">'
            '<svg class="brand-heart" viewBox="0 0 24 24" width="15" height="15">'
            '<path fill="#FF3355" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5'
            ' 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3'
            ' 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>'
            "ICEVISION</div>"
            '<div class="brand-sub">Algida Otomat Analitiği</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="glow-divider" style="margin: 0 0 14px 0;"/>', unsafe_allow_html=True)

        st.page_link("app.py", label="Genel Bakış")
        st.page_link("pages/01_shelf_analysis.py", label="Raf Analizi")
        st.page_link("pages/02_batch_analysis.py", label="Toplu Analiz")

        # ── Raf seçimi (QR doğrulama) ────────────────────────────────────
        st.markdown('<div class="sidebar-section-label">Raf Doğrulama</div>', unsafe_allow_html=True)
        shelf_options = ["Otomatik (QR'dan oku)"] + [f"Raf {i}" for i in range(1, 11)]
        selected_shelf = st.selectbox(
            "Raf Seçimi",
            options=shelf_options,
            label_visibility="collapsed",
            help="QR kodu okunan raf kimliğiyle karşılaştırılacak beklenen raf.",
        )
        st.session_state["expected_shelf"] = (
            None if selected_shelf == shelf_options[0] else selected_shelf.replace("Raf ", "")
        )

        # ── Status ────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-label">Sistem</div>', unsafe_allow_html=True)
        if model_path is not None:
            size_mb = round(model_path.stat().st_size / 1_048_576, 1)
            st.caption(f"Model {model_path.name} · {size_mb} MB")
        else:
            configured = cfg.get("inference", {}).get("weights_path", "—")
            st.caption(f"Model bulunamadı: {configured}")

        st.markdown('<div class="sidebar-footer">IceVision v1.0 · Algida</div>', unsafe_allow_html=True)
