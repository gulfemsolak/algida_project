"""Sayfa 2 — Toplu Analiz: birden fazla raf fotoğrafını işleyin, CSV indirin."""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import ExifTags, Image

from dashboard.components.charts import fill_rate_histogram
from dashboard.components.sidebar import render_sidebar
from dashboard.components.widgets import slot_detail_table
from dashboard.db import init_db, save_analysis, save_thumbnail
from dashboard.export.user_report import (
    batch_export_filename,
    batch_urun_dagilimi_rows,
    build_batch_report,
    build_error_report,
    build_user_report,
    resolve_model_version,
)
from dashboard.theme import ALGIDA_RED, AMBER, ICE, apply_theme, page_header, pill
from src.analysis.shelf_analyzer import analyze_shelf
from src.analysis.skew_corrector import deskew
from src.models.predictor import predict_shelf
from src.utils.visualization import draw_slots

_EXIF_ORIENTATION_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
_TEXT_MUTED = "#8195B5"
_TEXT_DEFAULT = "#EDF3FB"


def _load_config() -> dict:
    p = Path("config/config.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _exif_orientation(file_bytes: bytes) -> int | None:
    """bkz. 01_shelf_analysis.py — cv2.imdecode EXIF'i atar, PIL ile ayrıca okunur."""
    try:
        exif = Image.open(io.BytesIO(file_bytes)).getexif()
        return exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    except Exception:
        return None


def _metric_card(label: str, value, color: str = _TEXT_DEFAULT) -> str:
    """Tekli sayfadaki st.metric kart görünümünü taklit eden, değer rengi
    değiştirilebilen HTML kart (st.metric'in stMetricValue rengi global CSS'te
    !important ile sabitlendiği için koşullu renk için burada özel çizilir)."""
    return (
        '<div style="background: linear-gradient(180deg, rgba(255,255,255,0.025), '
        'rgba(255,255,255,0) 38%), #101828; border: 1px solid #1B2536; '
        'border-radius: 10px; padding: 1.25rem;">'
        f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{_TEXT_MUTED};font-weight:400;">{label}</div>'
        f'<div style="font-family:\'Sora\',system-ui,sans-serif;font-size:2rem;'
        f'font-weight:300;color:{color};margin-top:4px;">{value}</div>'
        '</div>'
    )


def _process_image(file_bytes: bytes, model_path, model, conf_threshold: float, cfg: dict) -> dict | None:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    slot_cfg = cfg.get("slot_analysis", {})
    if bool(slot_cfg.get("auto_rotate", True)):
        exif_orientation = _exif_orientation(file_bytes)
        work_img = deskew(img, exif_orientation=exif_orientation)["image"]
    else:
        work_img = img

    try:
        result = predict_shelf(work_img, model_path, conf_threshold, "config/config.yaml", model=model)
        # slot_report burada verilmiyor — analyze_shelf onu config'teki varsayılan
        # default_shelf_number/column_count ile kendi içinde kurar (bkz. keşif raporu:
        # toplu analizde QR/manuel raf seçimi yok, hassas atama tekli sayfanın işi).
        shelf = analyze_shelf(result, "config/config.yaml", image=work_img)
        return {"prediction": result, "shelf": shelf, "image": work_img}
    except Exception as exc:
        return {"error": str(exc)}


def run():
    cfg = _load_config()
    st.set_page_config(page_title="Toplu Analiz — IceVision", layout="wide", initial_sidebar_state="expanded")
    init_db()
    apply_theme()
    render_sidebar(cfg)

    page_header("Toplu Analiz")

    # Tekli sayfadaki manuel-giriş yönlendirme linki için — ana içerik alanındaki
    # page_link'leri ICE renginde göster (sidebar'ın kendi linkleri ayrı bir
    # selector'la zaten stillendiriliyor, bu kural onlara dokunmaz).
    st.markdown(
        f'<style>[data-testid="stMain"] [data-testid="stPageLink-NavLink"] p '
        f"{{ color: {ICE} !important; }}</style>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Raf fotoğraflarını sürükleyin veya tıklayıp yükleyin",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        m1, m2, m3 = st.columns(3)
        m1.metric("İşlenen Görsel", "—")
        m2.metric("Toplam Ürün", "—")
        m3.metric("Ort. Doluluk", "—")
        c1, c2 = st.columns(4)[:2]
        with c1:
            st.markdown(_metric_card("Yenilenmeli", "—"), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("Manuel Giriş", "—"), unsafe_allow_html=True)
        return

    conf_threshold = st.session_state.get("conf_threshold", 0.5)
    model_path = st.session_state.get("model_path")
    model = st.session_state.get("model")
    debug_mode = st.query_params.get("debug") == "1"

    if st.button(f"{len(uploaded_files)} görseli işle", type="primary"):
        results_rows = []
        detail_records = []
        progress = st.progress(0, text="İşleniyor...")
        status_placeholder = st.empty()

        for i, uploaded in enumerate(uploaded_files):
            status_placeholder.text(f"{uploaded.name} işleniyor ({i + 1}/{len(uploaded_files)})...")
            data = _process_image(uploaded.read(), model_path, model, conf_threshold, cfg)

            if data is None or "error" in data:
                results_rows.append({
                    "filename": uploaded.name,
                    "error": data["error"] if data else "Decode failed",
                    "fill_rate": 0,
                    "total_products": 0,
                    "total_empty": 0,
                    "restock_needed": False,
                    "overall_confidence": 0,
                    "anomalies": 0,
                    "lattice_available": False,
                })
                detail_records.append({
                    "filename": uploaded.name,
                    "error": True,
                    "error_message": data["error"] if data else "Görsel çözümlenemedi (bozuk dosya).",
                })
            else:
                pred = data["prediction"]
                shelf = data["shelf"]
                save_analysis({
                    "filename": uploaded.name,
                    "total_products": pred.get("total_products", 0),
                    "empty_slots": shelf.get("total_empty", 0),
                    "fill_rate": shelf.get("shelf_fill_rate", 0.0),
                    "avg_confidence": pred.get("overall_confidence", 0.0),
                    "shelf_rows": shelf.get("total_rows", 0),
                    "class_counts": {cat: d.get("count", 0) for cat, d in pred.get("summary", {}).items()},
                    "thumbnail_path": save_thumbnail(data["image"]),
                })
                lattice_available = bool(shelf.get("lattice_available"))
                results_rows.append({
                    "filename": uploaded.name,
                    "fill_rate": shelf["shelf_fill_rate"],
                    "total_products": pred["total_products"],
                    "total_empty": shelf["total_empty"],
                    "restock_needed": shelf["restock_needed"],
                    "overall_confidence": pred["overall_confidence"],
                    "anomalies": len(shelf.get("anomalies", [])),
                    "lattice_available": lattice_available,
                    "error": None,
                })
                detail_records.append({
                    "filename": uploaded.name,
                    "error": False,
                    "image": data["image"],
                    "annotated": pred.get("annotated_image"),
                    "summary": pred.get("summary", {}),
                    "shelf": shelf,
                    "lattice_available": lattice_available,
                })

            progress.progress((i + 1) / len(uploaded_files))

        progress.empty()
        status_placeholder.empty()

        st.session_state["batch_df"] = pd.DataFrame(results_rows)
        st.session_state["batch_details"] = detail_records

    df: pd.DataFrame | None = st.session_state.get("batch_df")
    if df is None or df.empty:
        return

    valid = df[df["error"].isna()]
    valid_lattice = valid[valid["lattice_available"]] if not valid.empty else valid
    manual_count = int((~valid["lattice_available"]).sum()) if not valid.empty else 0
    restock_count = int(valid["restock_needed"].sum()) if not valid.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("İşlenen Görsel", len(df))
    m2.metric("Toplam Ürün", int(valid["total_products"].sum()) if not valid.empty else 0)
    m3.metric("Ort. Doluluk", f"{valid_lattice['fill_rate'].mean():.1%}" if not valid_lattice.empty else "—")
    with m4:
        st.markdown(
            _metric_card("Yenilenmeli", restock_count, ALGIDA_RED if restock_count > 0 else _TEXT_DEFAULT),
            unsafe_allow_html=True,
        )

    if manual_count > 0:
        st.caption(f"{manual_count} fotoğraf manuel giriş beklediği için ortalamaya dahil değil.")

    # Beşinci kartı ayrı bir satırda tutuyoruz — 5 eşit kolon dar ekranlarda
    # sıkışır, bu yüzden ilk satırın ilk kolonuyla aynı genişlikte, kendi satırında.
    manual_row = st.columns(4)
    with manual_row[0]:
        st.markdown(
            _metric_card("Manuel Giriş", manual_count, AMBER if manual_count > 0 else _TEXT_DEFAULT),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    if not valid.empty:
        st.plotly_chart(fill_rate_histogram(valid), use_container_width=True)

    st.markdown('<hr class="glow-divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Görsel Bazında Sonuçlar</div>', unsafe_allow_html=True)
    details = st.session_state.get("batch_details", [])
    for rec in details:
        if rec.get("error"):
            with st.expander(f"{rec['filename']} — işlenemedi"):
                st.caption("Bu görsel işlenemedi.")
            continue

        shelf = rec["shelf"]
        slot_report = shelf.get("slot_report", {}) or {}

        if not rec["lattice_available"]:
            st.markdown(pill("manuel giriş gerekli", ALGIDA_RED), unsafe_allow_html=True)
            with st.expander(rec["filename"]):
                st.markdown(
                    f'<div class="detection-meta" style="color:{_TEXT_MUTED};">'
                    "Bu fotoğraf az tespit içerdiği için otomatik ızgara kurulamadı. "
                    "Detaylı analiz için Tekli Analiz sayfasına gidin.</div>",
                    unsafe_allow_html=True,
                )
                if debug_mode:
                    grid = slot_report.get("grid", {}) or {}
                    reason = "az tespit" if grid.get("low_detection") else "tepsi konturu bulunamadı"
                    st.markdown(
                        f'<div class="detection-meta" style="color:#46536E;">sebep: {reason}</div>',
                        unsafe_allow_html=True,
                    )
                st.page_link("pages/01_shelf_analysis.py", label="Tekli Analiz sayfasına git →")
            continue

        pct = round(shelf["shelf_fill_rate"] * 100)
        if shelf["restock_needed"]:
            st.markdown(pill("yenilenmeli", ALGIDA_RED), unsafe_allow_html=True)
        with st.expander(f"{rec['filename']} — %{pct} dolu"):
            col_img, col_slot = st.columns([1, 1])
            with col_img:
                base_img = rec["annotated"] if rec["annotated"] is not None else rec["image"]
                overlay = draw_slots(base_img, slot_report)
                st.image(_bgr_to_rgb(overlay), use_column_width=True)
            with col_slot:
                slot_detail_table(slot_report)

    st.markdown('<hr class="glow-divider"/>', unsafe_allow_html=True)

    default_shelf = int(cfg.get("slot_analysis", {}).get("default_shelf_number", 1))
    model_versiyonu = resolve_model_version(model_path)

    def _photo_report(rec: dict) -> dict:
        metadata = {
            "dosya": rec["filename"],
            "tarih": datetime.now().astimezone().isoformat(timespec="seconds"),
            "raf_numarasi": default_shelf,
            # Toplu analizde QR/manuel raf doğrulaması yok (bkz. önceki tur kararı:
            # hassas raf ataması tekli sayfanın işi) — bu yüzden hep False.
            "qr_okundu": False,
            "model_versiyonu": model_versiyonu,
        }
        if rec.get("error"):
            return build_error_report(metadata, rec.get("error_message", "İşlenemedi."))
        return build_user_report(
            rec["shelf"].get("slot_report"), rec["shelf"], metadata,
            detection_summary=rec.get("summary"),
        )

    gorseller = [_photo_report(rec) for rec in details]
    toplam_ozet = {
        "islenen_gorsel": len(df),
        "toplam_urun": int(valid["total_products"].sum()) if not valid.empty else 0,
        "manuel_giris_bekleyen": manual_count,
        "yenilenmeli_sayisi": restock_count,
        "ortalama_hesabina_dahil_gorsel": int(len(valid_lattice)),
    }
    if not valid_lattice.empty:
        toplam_ozet["ortalama_doluluk_yuzde"] = round(valid_lattice["fill_rate"].mean() * 100, 2)
    if manual_count > 0:
        toplam_ozet["aciklama"] = (
            f"{manual_count} görsel manuel giriş beklediği için ortalama hesabına dahil değil."
        )
    batch_report = build_batch_report(
        metadata={
            "tarih": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gorsel_sayisi": len(df),
            "model_versiyonu": model_versiyonu,
        },
        toplam_ozet=toplam_ozet,
        gorseller=gorseller,
    )

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="Rapor indir (JSON)",
            data=json.dumps(batch_report, indent=2, ensure_ascii=False),
            file_name=batch_export_filename(len(df)),
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        # JSON'daki "gorseller[].urun_dagilimi" ile AYNI kaynak (gorseller listesi) —
        # ayrı bir hesap yolu açmaz, CSV JSON'la her zaman birebir tutarlı kalır.
        csv_rows = batch_urun_dagilimi_rows(gorseller)
        csv_buf = io.StringIO()
        pd.DataFrame(csv_rows).to_csv(csv_buf, index=False)
        st.download_button(
            label="Ürün dağılımı indir (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"urun_dagilimi_{len(df)}gorsel.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if debug_mode:
        dev_export_data = [
            {"filename": rec["filename"], "error": True, "message": rec.get("error_message")}
            if rec.get("error")
            else {"filename": rec["filename"], "shelf_analysis": rec["shelf"]}
            for rec in details
        ]
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Geliştirici verisi (JSON)",
            data=json.dumps(dev_export_data, indent=2),
            file_name="batch_analysis_raw.json",
            mime="application/json",
        )


run()
