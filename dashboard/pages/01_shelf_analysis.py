"""Sayfa 1 — Raf Analizi: raf fotoğrafı yükleyin, anında sonuç alın."""
import base64
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import ExifTags, Image

from dashboard.components.charts import category_count_bar
from dashboard.components.sidebar import render_sidebar
from dashboard.components.widgets import product_breakdown_row, qr_status_badge, slot_detail_table
from dashboard.db import init_db, save_analysis, save_manual_extension, save_thumbnail
from dashboard.export.user_report import build_user_report, resolve_model_version, single_export_filename
from dashboard.theme import alert_bar, apply_theme, fill_rate_ring, format_class_name, page_header, section_header
from src.models.predictor import predict_shelf
from src.analysis.shelf_analyzer import analyze_shelf
from src.analysis.slot_assigner import build_slot_report, EMPTY_CLASS_NAME
from src.analysis.product_anchored_grid import (
    NUMBERING_AXIS_AMBIGUITY_EPS, SCREEN_BOTH, SCREEN_DOWN, SCREEN_LEFT, SCREEN_RIGHT,
    SCREEN_UP, estimate_grid_lane_based, resolve_screen_extend,
)
from src.analysis.skew_corrector import deskew
from src.utils.visualization import draw_slots
from qr_reader import verify_shelf_qr

_EXIF_ORIENTATION_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")


def _load_config() -> dict:
    p = Path("config/config.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _exif_orientation(file_bytes: bytes) -> int | None:
    """Yüklenen dosya baytlarından EXIF Orientation etiketini (1..8) çıkar.

    cv2.imdecode EXIF'i göz ardı eder/atar — bu yüzden döndürme kararı için
    ayrı ayrı, PIL ile, ham baytlardan okunmalı.
    """
    try:
        exif = Image.open(io.BytesIO(file_bytes)).getexif()
        return exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    except Exception:
        return None


def _shelf_number_from_qr(qr_result: dict | None) -> int | None:
    """QR değerlerinden raf numarasını çıkar (ilk tam sayı). QR yoksa None.

    Slot kimlikleri buradan üretilir: QR raf 3 → slotlar 31, 32, ... (shelf_col şeması).
    """
    if not qr_result or not qr_result.get("qr_found"):
        return None
    for value in qr_result.get("qr_values", []):
        m = re.search(r"\d+", str(value))
        if m:
            n = int(m.group())
            if 1 <= n <= 99:
                return n
    return None


def run():
    cfg = _load_config()
    st.set_page_config(page_title="Raf Analizi — IceVision", layout="wide", initial_sidebar_state="expanded")
    init_db()
    apply_theme()
    render_sidebar(cfg)

    page_header("Raf Analizi")

    # Geliştirici görünümü — s-uzayı/lane_meta gibi iç detaylar operatöre gösterilmez,
    # yalnız ?debug=1 ile.
    debug_mode = st.query_params.get("debug") == "1"

    uploaded = st.file_uploader(
        "Raf fotoğrafını sürükleyin veya tıklayıp yükleyin",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Ürün", "—")
        m2.metric("Boş Yuva", "—")
        m3.metric("Ort. Güven", "—")
        m4.metric("Raf Sırası", "—")
        return

    raw_bytes = uploaded.read()
    # Yüklenen görsele KAPSAMLI session_state anahtarı: operatörün bir fotoda verdiği
    # elle düzeltme (köşe/kanal sayısı) yeni bir fotoğrafa SESSİZCE taşınmasın.
    img_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
    exif_orientation = _exif_orientation(raw_bytes)
    file_bytes = np.frombuffer(raw_bytes, np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if original_bgr is None:
        st.error("Görsel okunamadı. Geçerli bir JPG veya PNG dosyası yükleyin.")
        return

    conf_threshold = st.session_state.get("conf_threshold", 0.5)
    model_path = st.session_state.get("model_path")
    model = st.session_state.get("model")

    slot_cfg = cfg.get("slot_analysis", {})
    column_count = int(slot_cfg.get("column_count", 7))

    # ── Deskew: YALNIZ EXIF Orientation'a göre kanonik (90° katı) döndürme. İçerikten
    #    tahmin YOK — kalan perspektif/eğim shelf_grid homografisinin işi.
    with st.spinner("Tespit çalışıyor..."):
        t0 = time.perf_counter()
        dk = deskew(original_bgr, exif_orientation=exif_orientation) if bool(slot_cfg.get("auto_rotate", True)) else {
            "image": original_bgr, "angle": 0.0, "coarse": 0, "orient_source": "kapalı",
        }
        work_bgr = dk["image"]
        rotation_angle = dk["angle"]
        try:
            result = predict_shelf(
                image_path=work_bgr,
                model_path=model_path,
                model=model,
                conf_threshold=conf_threshold,
                config_path="config/config.yaml",
            )
        except Exception as exc:
            st.error(f"Tespit başarısız: {exc}")
            return
        elapsed = time.perf_counter() - t0

    # QR tespiti — YOLO akışından bağımsız katman, YOLO sonucu ne olursa olsun çalışır.
    expected_shelf = st.session_state.get("expected_shelf")
    qr_result = verify_shelf_qr(original_bgr, expected_shelf=expected_shelf)

    # ── Izgara çapası: tepsi 4 köşesinden (kontur → belirsiz), tespit bulutundan DEĞİL.
    #    Tek kaynak: bu grid hem atamaya hem overlay çizimine hem de doluluk hesabına
    #    (analyze_shelf'e slot_report olarak) verilir. Operatör elle köşe işaretlediyse
    #    (belirsiz sonrası) onu kullan. Anahtarlar img_hash'e KAPSAMLI — bir fotoda verilen
    #    elle düzeltme başka bir fotoğrafa sessizce taşınmaz.
    manual_corners_key = f"manual_corners::{img_hash}"
    expected_k_key = f"expected_column_count::{img_hash}"
    extend_dir_key = f"manual_extend_direction::{img_hash}"
    extend_count_key = f"manual_extend_count::{img_hash}"
    manual_corners = st.session_state.get(manual_corners_key)
    expected_column_count = st.session_state.get(expected_k_key)
    manual_extend_direction = st.session_state.get(extend_dir_key)
    manual_extend_count = st.session_state.get(extend_count_key, 0)
    grid = estimate_grid_lane_based(
        work_bgr, column_count, result["detections"], corners_override=manual_corners,
        expected_column_count=expected_column_count,
        manual_extend_direction=manual_extend_direction, manual_extend_count=manual_extend_count,
    )

    # ── Raf numarası: QR → tepsi barı OCR → operatör (sessiz varsayılana DÜŞME) ──
    default_shelf = int(slot_cfg.get("default_shelf_number", 1))
    qr_shelf = _shelf_number_from_qr(qr_result)
    bar_shelf = grid.get("shelf_number_from_bar")
    shelf_verified = True
    if qr_shelf is not None:
        shelf_number = qr_shelf
        st.markdown(
            f'<div class="detection-meta">Raf numarası QR\'dan okundu: '
            f'<b>{shelf_number}</b> &middot; slotlar {shelf_number}1…{shelf_number}{column_count}</div>',
            unsafe_allow_html=True,
        )
    elif bar_shelf is not None:
        shelf_number = bar_shelf
        st.markdown(
            f'<div class="detection-meta">Raf numarası tepsi barından okundu: '
            f'<b>{shelf_number}</b> &middot; <span style="color:#FFB547;">doğrulanmadı</span></div>',
            unsafe_allow_html=True,
        )
        shelf_verified = False
    else:
        sel_col, _ = st.columns([1, 3])
        with sel_col:
            shelf_number = st.selectbox(
                "Raf Numarası",
                options=list(range(1, 11)),
                index=default_shelf - 1 if 1 <= default_shelf <= 10 else 0,
                help="QR ve etiket rayı okunamadı — raf numarasını doğrulayın (Raf 3 → 31, 32, ...).",
            )
        st.markdown(
            '<div class="detection-meta"><span style="color:#8195B5;">'
            'QR ve tepsi barı okunamadı — seçilen raf numarası doğrulanmadı.</span></div>',
            unsafe_allow_html=True,
        )
        shelf_verified = False

    slot_report = build_slot_report(
        result["detections"],
        shelf_number=shelf_number,
        column_count=column_count,
        grid=grid,
        empty_class_name=EMPTY_CLASS_NAME,
        expected_column_count=expected_column_count,
    )

    # slot_report zaten kuruldu — analyze_shelf'e veriyoruz ki lattice'i (tepsi köşe
    # çapası) İKİNCİ KEZ hesaplamasın (tek kaynak, çifte iş yok).
    shelf_report = analyze_shelf(result, config_path="config/config.yaml", slot_report=slot_report)

    # Persist this analysis for the Dashboard KPIs/history, and cache it for Restock Planner reuse.
    save_analysis({
        "filename": uploaded.name,
        "total_products": result.get("total_products", 0),
        "empty_slots": shelf_report.get("total_empty", 0),
        "fill_rate": shelf_report.get("shelf_fill_rate", 0.0),
        "avg_confidence": result.get("overall_confidence", 0.0),
        "shelf_rows": shelf_report.get("total_rows", 0),
        "class_counts": {cat: d.get("count", 0) for cat, d in result.get("summary", {}).items()},
        "thumbnail_path": save_thumbnail(original_bgr),
        "qr_info": qr_result,
    })
    st.session_state["last_result"] = result
    st.session_state["last_shelf_report"] = shelf_report
    st.session_state["last_image_bgr"] = work_bgr  # tespitlerle aynı (döndürülmüş) görsel
    st.session_state["last_qr_result"] = qr_result

    # ── Hero metrics ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Ürün", result["total_products"])
    m2.metric("Boş Yuva", shelf_report["total_empty"])
    m3.metric("Ort. Güven", f"{result['overall_confidence']:.1%}")
    m4.metric("Raf Sırası", shelf_report["total_rows"])

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_main, col_side = st.columns([1.2, 1])

    # ── Left panel: detection view ──────────────────────────────────────────
    with col_main:
        annotated = result.get("annotated_image")
        base_img = annotated if annotated is not None else work_bgr
        slot_overlay = draw_slots(base_img, slot_report)
        tab_slots, tab_detected, tab_original = st.tabs(["SLOTLAR", "TESPİT", "ORİJİNAL"])
        with tab_slots:
            st.image(_bgr_to_rgb(slot_overlay), use_column_width=True)
        with tab_detected:
            st.image(_bgr_to_rgb(base_img), use_column_width=True)
        with tab_original:
            st.image(_bgr_to_rgb(original_bgr), use_column_width=True)

        # Deskew notu: yalnız EXIF kaynaklı 90° katı döndürme (varsa).
        rot_note = f' &middot; EXIF ile döndürüldü {rotation_angle:+.0f}°' if rotation_angle else ""
        # Izgara belirsizse kullanıcı sonucun hesaplanamadığını görsün (düşük vurgu).
        grid_note = (
            ' &middot; <span style="color:#8195B5;">ızgara belirsiz</span>'
            if slot_report.get("grid_status") == "belirsiz" else ""
        )
        st.markdown(
            f'<div class="detection-meta">{len(result["detections"])} tespit '
            f'&middot; {elapsed:.1f}s &middot; conf &ge; {conf_threshold:.2f}{rot_note}{grid_note}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(qr_status_badge(qr_result), unsafe_allow_html=True)

        # ── Köşe-picker: tepsinin 4 köşesini elle işaretlemek için paylaşılan UI.
        #    İki bağlamda kullanılır: (a) kontur bulunamadı, (b) az tespit/low_detection.
        #    İkisi de aynı yoldan geçer — estimate_grid_lane_based(corners_override=...)
        #    verilince _lane_grid'e HİÇ uğramaz, doğrudan homografi kurulur (bkz.
        #    product_anchored_grid.py estimate_grid_lane_based) — yani low_detection'ın
        #    erken-dönüşünü zaten bypass eder, backend'de ekstra bağlantı gerekmedi.
        def _corner_picker(caption: str, key_suffix: str) -> None:
            st.markdown(
                f'<div class="detection-meta" style="color:#8195B5;margin-top:8px;">{caption}</div>',
                unsafe_allow_html=True,
            )
            img_h, img_w = work_bgr.shape[:2]
            default_corners = [
                (int(img_w * 0.1), int(img_h * 0.15)),   # sol üst
                (int(img_w * 0.9), int(img_h * 0.15)),   # sağ üst
                (int(img_w * 0.9), int(img_h * 0.85)),   # sağ alt
                (int(img_w * 0.1), int(img_h * 0.85)),   # sol alt
            ]
            saved = manual_corners or default_corners
            labels = ["Sol üst", "Sağ üst", "Sağ alt", "Sol alt"]
            corner_cols = st.columns(4)
            new_corners = []
            for i, (label, (dx, dy)) in enumerate(zip(labels, saved)):
                with corner_cols[i]:
                    st.markdown(f"**{label}**")
                    x = st.number_input("x", min_value=0, max_value=img_w, value=int(dx),
                                         key=f"corner_x_{i}_{key_suffix}::{img_hash}")
                    y = st.number_input("y", min_value=0, max_value=img_h, value=int(dy),
                                         key=f"corner_y_{i}_{key_suffix}::{img_hash}")
                    new_corners.append((x, y))
            c_apply, c_clear = st.columns(2)
            if c_apply.button("Bu köşelerle hesapla", use_container_width=True,
                               key=f"corner_apply_{key_suffix}::{img_hash}"):
                st.session_state[manual_corners_key] = new_corners
                st.rerun()
            if c_clear.button("Temizle", use_container_width=True,
                               key=f"corner_clear_{key_suffix}::{img_hash}"):
                st.session_state.pop(manual_corners_key, None)
                st.rerun()

        # ── Az tespit (near-empty tepsi): latis mimari olarak güvenilmez (bkz.
        #    product_anchored_grid.py "low_detection" notu). İki manuel yol: (1) toplam
        #    kanal sayısı (K) — geometrisiz SAYIM tabanlı doluluk raporu, (2) tepsi
        #    köşeleri — TAM lattice (kontur-bulunamadı senaryosuyla birebir aynı backend
        #    yolu). "Eksik kanal nerede?" burada YOK: o widget yalnız BAŞARILI latis'i
        #    (zincir) ince-ayar yapar; low_detection'da henüz latis kurulmadığından
        #    backend'de karşılığı yoktu (sessiz no-op'tu, bkz. keşif raporu).
        if grid.get("low_detection"):
            n_prod = len([d for d in result["detections"] if d.get("category") != EMPTY_CLASS_NAME])
            st.markdown(
                '<div class="detection-meta" style="color:#8195B5;margin-top:8px;">'
                '<b style="color:#EDF3FB;font-weight:500;">Kanal yapısı çıkarılamadı — manuel giriş</b><br/>'
                f'Ürün az olduğu için otomatik ızgara kurulamadı ({n_prod} tespit). Toplam kanal '
                'sayısını girin ya da tepsi köşelerini işaretleyerek daha doğru sonuç alın.</div>',
                unsafe_allow_html=True,
            )
            current_k = expected_column_count if expected_column_count is not None else grid.get("column_count") or 7
            current_k = int(min(max(current_k, 4), 12))
            with st.expander("Toplam kanal sayısı", expanded=False):
                new_k = st.number_input(
                    "Toplam kanal sayısı", min_value=4, max_value=12,
                    value=current_k, step=1, key=f"expected_k_input_ld::{img_hash}",
                )
                k_apply, k_clear = st.columns(2)
                if k_apply.button("Uygula", use_container_width=True, key=f"k_apply_ld::{img_hash}"):
                    st.session_state[expected_k_key] = int(new_k)
                    st.rerun()
                if k_clear.button("Temizle", use_container_width=True, key=f"k_clear_ld::{img_hash}"):
                    st.session_state.pop(expected_k_key, None)
                    st.rerun()
            if expected_column_count is not None:
                st.markdown(
                    f'<div class="detection-meta">Kanal sayısı elle ayarlandı: '
                    f'<b>{expected_column_count}</b></div>',
                    unsafe_allow_html=True,
                )

            count_report = slot_report.get("count_report")
            if count_report is not None:
                pct = round(count_report["doluluk"] * 100)
                overflow_note = (
                    ' <span style="color:#FFB547;">— tespit edilen küme sayısı girilen '
                    'kanal sayısını aşıyor, kırpıldı</span>' if count_report.get("kume_asimi") else ""
                )
                st.markdown(
                    f'<div class="detection-meta">Dolu kanal: <b>{count_report["dolu_kanal"]}/'
                    f'{expected_column_count}</b> &middot; Boş kanal: '
                    f'<b>{count_report["bos_kanal"]}</b> &middot; Doluluk: <b>%{pct}</b>'
                    f'{overflow_note}</div>',
                    unsafe_allow_html=True,
                )

            show_corners_key = f"show_corner_picker::{img_hash}"
            if st.button("Tepsi köşelerini işaretle", key=f"toggle_corners::{img_hash}"):
                st.session_state[show_corners_key] = not st.session_state.get(show_corners_key, False)
            if st.session_state.get(show_corners_key):
                _corner_picker(
                    "Tepsinin 4 köşesini işaretleyin — kanal sayısından bağımsız olarak "
                    "en doğru sonucu verir.",
                    key_suffix="ld",
                )

        # ── Belirsiz ızgara (tepsi konturu bulunamadı): sahte slot üretme; operatöre
        #    tepsinin 4 köşesini işaretlet. low_detection İSE bu blok hiç görünmez —
        #    o durum kendi köşe-picker'ını yukarıda (low_detection bloğunda) sunuyor.
        if slot_report.get("grid_status") == "belirsiz" and not grid.get("low_detection"):
            _corner_picker(
                "Raf ızgarası çıkarılamadı (tepsi konturu bulunamadı) — tespitler gösteriliyor "
                "ancak slot dağılımı hesaplanamadı. Tepsinin 4 köşesini aşağıdan işaretleyip "
                "yeniden hesaplayın.",
                key_suffix="contour",
            )

        # ── Kanal sayısı doğrulama — Tür-2 (en dış ürünün ötesindeki boş kanal) için
        #    operatör-döngüsü, YALNIZ başarılı latis'te (zincir). CV bu soruyu güvenilir
        #    cevaplayamıyor (bkz. modülün başındaki "BİLİNEN SINIRLAMA" notu — 5 farklı
        #    yaklaşım ölçülüp reddedildi); bu yüzden iddialı bir uyarı YERİNE her zaman
        #    erişilebilir, nötr bir kontrol sunulur. Operatör dokunmazsa no-op.
        #    low_detection burada YOK (yukarıdaki ayrı blok işini görüyor) — latis henüz
        #    kurulmadığı için "eksik kanal nerede" sorusunun bu aşamada karşılığı yoktu.
        if grid.get("grid_source") == "zincir":
            current_k = expected_column_count if expected_column_count is not None else grid.get("column_count") or 7
            # Bandın DIŞINDA bir varsayılan (ör. latis 1 ya da 3 kanal bulduğunda)
            # st.number_input'u value<min_value ile çökertir — bandın içine sıkıştır.
            current_k = int(min(max(current_k, 4), 12))
            with st.expander("Kanal sayısı doğru mu?", expanded=False):
                new_k = st.number_input(
                    "Toplam kanal sayısı", min_value=4, max_value=12,
                    value=current_k, step=1, key=f"expected_k_input::{img_hash}",
                )
                k_apply, k_clear = st.columns(2)
                if k_apply.button("Uygula", use_container_width=True, key=f"k_apply::{img_hash}"):
                    st.session_state[expected_k_key] = int(new_k)
                    st.rerun()
                if k_clear.button("Temizle", use_container_width=True, key=f"k_clear::{img_hash}"):
                    st.session_state.pop(expected_k_key, None)
                    st.rerun()
            if expected_column_count is not None:
                st.markdown(
                    f'<div class="detection-meta">Kanal sayısı elle ayarlandı: '
                    f'<b>{expected_column_count}</b></div>',
                    unsafe_allow_html=True,
                )

            # ── Eksik kanal yönü (GÖREV 2) — K girişi tek başına yetmiyor (Tür-2'de
            #    K-kısıtlı comb da aynı güvenilmez kenarlara dayanıyor, bkz. modülün
            #    "BİLİNEN SINIRLAMA" notu). Operatör GÖRDÜĞÜ eksik kanalın yerini/sayısını
            #    girer; latis mevcut (güvenilir) pitch ile o yöne uzatılır — yeni CV yok.
            #    Ekran yönü → s-uzayı çevirisi TEK YERDE (``resolve_screen_extend``) —
            #    burada TEKRARLANMAZ, doğrudan çağrılır (bkz. ölçülmüş "s-uzayı ekran
            #    uzayıymış gibi kullanıldı" dersi, pitch_lattice_slots.md).
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            axis_u = grid.get("axis_u")
            if axis_u is None:
                axis_u = [1.0, 0.0]
            axis_ambiguous = abs(abs(axis_u[0]) - abs(axis_u[1])) < NUMBERING_AXIS_AMBIGUITY_EPS
            column_oriented = abs(axis_u[0]) >= abs(axis_u[1])

            if axis_ambiguous:
                st.markdown(
                    '<div class="detection-meta" style="color:#8195B5;">'
                    'Raf yönü (sütun/satır) belirsiz — ekran yönü güvenilir tanımlanamıyor, '
                    'kanal uzatma devre dışı.</div>',
                    unsafe_allow_html=True,
                )
            else:
                if column_oriented:
                    dir_labels = {"Yok": None, "Solda": SCREEN_LEFT, "Sağda": SCREEN_RIGHT, "İkisi de": SCREEN_BOTH}
                else:
                    dir_labels = {"Yok": None, "Yukarıda": SCREEN_UP, "Aşağıda": SCREEN_DOWN, "İkisi de": SCREEN_BOTH}
                label_by_value = {v: k for k, v in dir_labels.items()}
                dir_options = list(dir_labels.keys())
                default_label = label_by_value.get(manual_extend_direction, "Yok")
                direction_label = st.selectbox(
                    "Eksik kanal nerede?", dir_options,
                    index=dir_options.index(default_label) if default_label in dir_options else 0,
                    key=f"extend_dir::{img_hash}",
                )
                direction_value = dir_labels[direction_label]
                if direction_value is not None:
                    count = st.number_input(
                        "Kaç kanal (yön başına)", min_value=1, max_value=10,
                        value=max(manual_extend_count, 1), step=1, key=f"extend_count::{img_hash}",
                    )
                    ext_apply, ext_clear = st.columns(2)
                    if ext_apply.button("Uygula", use_container_width=True, key=f"ext_apply::{img_hash}"):
                        k_before = grid.get("column_count") or 0
                        add_low_s, add_high_s = resolve_screen_extend(direction_value, int(count), axis_u)
                        st.session_state[extend_dir_key] = direction_value
                        st.session_state[extend_count_key] = int(count)
                        save_manual_extension(
                            uploaded.name, add_low_s, add_high_s,
                            k_before, k_before + add_low_s + add_high_s,
                            direction_label=direction_label,
                        )
                        st.rerun()
                    if ext_clear.button("Temizle", use_container_width=True, key=f"ext_clear::{img_hash}"):
                        st.session_state.pop(extend_dir_key, None)
                        st.session_state.pop(extend_count_key, None)
                        st.rerun()
            if manual_extend_direction and debug_mode:
                meta = grid.get("lane_meta", {})
                add_low = meta.get("manual_extend_add_low_s", 0)
                add_high = meta.get("manual_extend_add_high_s", 0)
                dir_shown = {"Yok": None, "Solda": SCREEN_LEFT, "Sağda": SCREEN_RIGHT,
                             "İkisi de": SCREEN_BOTH, "Yukarıda": SCREEN_UP, "Aşağıda": SCREEN_DOWN}
                shown_label = next((k for k, v in dir_shown.items() if v == manual_extend_direction), manual_extend_direction)
                st.markdown(
                    f'<div class="detection-meta">Kanal uzatıldı — yön: <b>{shown_label}</b>, '
                    f'kanal: <b>{manual_extend_count}</b> (s-uzayı: düşük+{add_low}, yüksek+{add_high})</div>',
                    unsafe_allow_html=True,
                )
                if grid.get("lane_meta", {}).get("extend_frame_warning"):
                    st.markdown(
                        '<div class="detection-meta" style="color:#FFB547;">'
                        'Uyarı: eklenen kanal(lar) görüntü kadrajının dışına taşıyor olabilir — '
                        'yine de uygulandı, kırpılmadı.</div>',
                        unsafe_allow_html=True,
                    )

        anomalies = shelf_report.get("anomalies", [])
        if anomalies:
            with st.expander(f"{len(anomalies)} anomali tespit edildi"):
                for a in anomalies:
                    st.markdown(
                        f"**{format_class_name(a['category'])}** — güven {a['confidence']:.1%} — {a['note']}"
                    )

    # ── Right panel: results dashboard ──────────────────────────────────────
    with col_side:
        section_header(f"Raf Detayı · Raf {shelf_number}")
        if debug_mode:
            st.markdown(
                '<div class="detection-meta" style="color:#46536E;">geliştirici görünümü</div>',
                unsafe_allow_html=True,
            )
        slot_detail_table(slot_report)

    # ── Category count chart ────────────────────────────────────────────────
    if result["summary"]:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(category_count_bar(result["summary"]), use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="glow-divider"/>', unsafe_allow_html=True)

    # QR bir raf numarası ÜRETTİYSE okundu sayılır — "verified" (beklenen rafla
    # karşılaştırma) sidebar'da bir raf seçilmediyse (varsayılan "Otomatik" akışı)
    # zaten None döner; qr_okundu'yu ona bağlamak en yaygın akışta yanlışlıkla
    # false üretirdi. bkz. dashboard/export/user_report.py modül notu.
    qr_okundu = qr_shelf is not None
    metadata = {
        "dosya": uploaded.name,
        "tarih": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raf_numarasi": shelf_number,
        "qr_okundu": qr_okundu,
        "model_versiyonu": resolve_model_version(model_path),
    }
    user_report = build_user_report(
        slot_report, shelf_report, metadata, detection_summary=result.get("summary"),
    )

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="Rapor indir (JSON)",
            data=json.dumps(user_report, indent=2, ensure_ascii=False),
            file_name=single_export_filename(uploaded.name),
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        # JSON'daki "urun_dagilimi" ile AYNI kaynak — ayrı bir hesap yolu açmaz,
        # CSV'deki adet/kanal_sayisi JSON'dakiyle her zaman birebir eşleşir.
        rows = user_report.get("urun_dagilimi", [])
        csv_buf = io.StringIO()
        pd.DataFrame(rows).to_csv(csv_buf, index=False)
        csv_b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()
        st.markdown(
            f'<a href="data:text/csv;base64,{csv_b64}" download="urun_dagilimi.csv" '
            f'style="display:block;box-sizing:border-box;text-align:center;padding:0.5rem 1.5rem;'
            f'border:1px solid #8195B5;border-radius:6px;color:#8195B5;text-decoration:none;'
            f'font-size:0.85rem;font-weight:500;letter-spacing:0.02em;">CSV İndir</a>',
            unsafe_allow_html=True,
        )

    if debug_mode:
        dev_export_data = {
            "detection_result": {k: v for k, v in result.items() if k != "annotated_image"},
            "shelf_analysis": shelf_report,
            "slot_analysis": slot_report,
            "qr_info": qr_result,
        }
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Geliştirici verisi (JSON)",
            data=json.dumps(dev_export_data, indent=2),
            file_name="shelf_analysis_result.json",
            mime="application/json",
        )


run()
