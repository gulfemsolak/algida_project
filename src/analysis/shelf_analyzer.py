"""Higher-level shelf analysis on top of raw YOLO detections.

Groups detections by shelf row, computes per-row statistics,
generates a structured shelf map, and flags restocking needs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import yaml

from src.analysis.slot_assigner import (
    DEFAULT_COLUMN_COUNT,
    DEFAULT_SHELF_NUMBER,
    EMPTY_CLASS_NAME,
    build_slot_report,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _cluster_rows(
    detections: list[dict[str, Any]],
    eps: float = 50.0,
) -> dict[int, list[dict[str, Any]]]:
    """Group detections into shelf rows by y-centre proximity (simple 1-D DBSCAN).

    Args:
        detections: List of detection dicts (must contain ``bbox`` [x1,y1,x2,y2]).
        eps: Maximum y-centre distance (pixels) between two detections on the same row.

    Returns:
        Dict mapping row_index (0 = topmost) → list of detection dicts.
    """
    if not detections:
        return {}

    y_centres = np.array(
        [((d["bbox"][1] + d["bbox"][3]) / 2) for d in detections]
    )
    order = np.argsort(y_centres)
    sorted_y = y_centres[order]
    sorted_dets = [detections[i] for i in order]

    rows: list[list[dict[str, Any]]] = []
    current_row: list[dict[str, Any]] = [sorted_dets[0]]
    current_centre = sorted_y[0]

    for y, det in zip(sorted_y[1:], sorted_dets[1:]):
        if abs(y - current_centre) <= eps:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
            current_centre = y
    rows.append(current_row)

    # Sort each row left-to-right by x-centre
    for row in rows:
        row.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)

    return {i: row for i, row in enumerate(rows)}


def _row_stats(row: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(d["category"] for d in row)
    confs = [d["confidence"] for d in row]
    empty = counts.get("empty_slot", 0)
    products = sum(v for k, v in counts.items() if k != "empty_slot")
    total = products + empty
    return {
        "total_slots": total,
        "product_count": products,
        "empty_slots": empty,
        "fill_rate": round(products / total, 4) if total > 0 else 0.0,
        "category_counts": dict(counts),
        "avg_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
    }


def analyze_shelf(
    prediction_result: dict[str, Any],
    config_path: str = "config/config.yaml",
    image: Any | None = None,
    slot_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyse a predictor output dict and return a rich shelf report.

    Doluluk (empty/fill-rate) artık ham ``empty_slot`` dedeksiyonundan DEĞİL,
    lattice/slot katmanından (``slot_assigner.build_slot_report``) geliyor —
    empty_slot recall'u çok düşük olduğu için (bkz. predictor.py notu) model
    boş rafların çoğunu kaçırıyordu; tek doğru kaynak artık "bu slotta ürün
    dedeksiyonu var mı" sorusu.

    Args:
        prediction_result: Output of ``predictor.predict_shelf()``.
        config_path: Path to config.yaml.
        image: BGR görüntü — ``slot_report`` verilmediyse ızgara (tepsi köşe
            çapası) buradan çıkarılır. None ise ve ``slot_report`` de yoksa
            ızgara kurulamaz (belirsiz).
        slot_report: Hazır ``slot_assigner.build_slot_report`` çıktısı varsa
            yeniden hesaplanmaz (ör. Raf Analizi sayfası zaten kurmuş olabilir —
            tek kaynak, çifte hesap yok).

    Returns:
        Dict containing ``rows``, ``shelf_map``, ``restock_needed``,
        ``anomalies``, and aggregate statistics.
    """
    cfg = _load_config(config_path)
    shelf_cfg = cfg.get("shelf_analysis", {})
    slot_cfg = cfg.get("slot_analysis", {})
    row_eps: float = shelf_cfg.get("row_cluster_eps", 50.0)
    restock_threshold: float = shelf_cfg.get("restock_threshold", 0.5)
    low_conf_threshold: float = shelf_cfg.get("low_confidence_threshold", 0.4)
    empty_restock_n: int = shelf_cfg.get("empty_slot_restock_count", 3)

    detections: list[dict[str, Any]] = prediction_result.get("detections", [])
    row_map = _cluster_rows(detections, eps=row_eps)

    row_reports: dict[str, Any] = {}
    for row_idx, row_dets in row_map.items():
        row_reports[f"row_{row_idx}"] = _row_stats(row_dets)

    all_products = prediction_result.get("total_products", 0)

    if slot_report is None:
        shelf_number = int(slot_cfg.get("default_shelf_number", DEFAULT_SHELF_NUMBER))
        column_count = int(slot_cfg.get("column_count", DEFAULT_COLUMN_COUNT))
        slot_report = build_slot_report(
            detections, shelf_number=shelf_number, column_count=column_count,
            image=image, empty_class_name=EMPTY_CLASS_NAME,
        )

    slots = slot_report.get("slots", [])
    lattice_available = slot_report.get("grid_status") == "ok" and bool(slots)

    if lattice_available:
        all_empty = sum(1 for s in slots if s["is_empty"])
        total_slots = len(slots)
        shelf_fill_rate = round((total_slots - all_empty) / total_slots, 4) if total_slots else 0.0
    else:
        # Izgara kurulamadı (belirsiz/az tespit) — empty_slot dedeksiyonu artık
        # yok, dolayısıyla kanıtsız bir "boş" iddiası üretmiyoruz; ürün varsa
        # %100 varsayımı, yoksa %0. Alarm tetiklemez (restock_needed=False,
        # aşağıda ``lattice_available`` kontrolüyle).
        all_empty = 0
        total_slots = all_products
        shelf_fill_rate = 1.0 if all_products > 0 else 0.0

    # Restock decision — ızgara belirsizken alarm ÜRETME (kanıtsız iddia).
    restock_needed = lattice_available and (
        shelf_fill_rate < restock_threshold
        or all_empty >= empty_restock_n
    )

    # Shelf map (Dolum Planlama sayfası haritası): lattice'in TEK satırı (bkz.
    # slot_assigner.py "bir fotoğraf = bir raf") — is_empty=True slotlar
    # "empty_slot" sentinel kategorisiyle işaretlenir (widgets.shelf_map_grid
    # zaten bu string'e bakıyor; model class'ı DEĞİL, salt UI işareti).
    shelf_map: dict[str, Any] = {}
    if lattice_available:
        shelf_map["row_0"] = [
            {
                "position": s["column_no"] - 1,
                "category": EMPTY_CLASS_NAME if s["is_empty"] else s["product_name"],
                "confidence": s["confidence"],
                "x_centre": round(s["x"]),
            }
            for s in slots
        ]

    # Anomaly detection
    anomalies: list[dict[str, Any]] = []
    for det in detections:
        if det["confidence"] < low_conf_threshold:
            anomalies.append({
                "type": "low_confidence",
                "category": det["category"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "note": "Olası hasarlı ambalaj veya kısmi örtülme",
            })

    # Rows that need restocking
    rows_needing_restock = [
        row_id
        for row_id, stats in row_reports.items()
        if stats["empty_slots"] >= empty_restock_n or stats["fill_rate"] < restock_threshold
    ]

    return {
        "image_path": prediction_result.get("image_path", ""),
        "total_rows": len(row_map),
        "total_slots": total_slots,
        "total_products": all_products,
        "total_empty": all_empty,
        "shelf_fill_rate": shelf_fill_rate,
        "lattice_available": lattice_available,
        "restock_needed": restock_needed,
        "rows_needing_restock": rows_needing_restock,
        "rows": row_reports,
        "shelf_map": shelf_map,
        "slot_report": slot_report,
        "anomalies": anomalies,
        "overall_confidence": prediction_result.get("overall_confidence", 0.0),
    }
