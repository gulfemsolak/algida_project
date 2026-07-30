"""Clean inference pipeline: image → structured JSON result."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.utils.logger import get_logger
from src.utils.visualization import draw_detections

log = get_logger(__name__)

def filter_detections_by_confidence(
    detections: list[dict[str, Any]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Single global confidence filter applied to every inference result.

    One cutoff for every class, empty_slot included. The former per-class
    empty_slot threshold and product-coverage filter were removed: the model is
    run at ``conf_threshold`` directly (see ``predict_shelf``), so a plain
    confidence filter is all that remains — it still trims the demo-mode spread.
    """
    return [d for d in detections if d["confidence"] >= conf_threshold]


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str | Path):
    """Load YOLO weights once, raising on failure.

    Public and side-effect-free so the dashboard can wrap it in
    ``@st.cache_resource`` — a single shared model instance for the whole
    process. Streamlit reruns the script on every widget interaction, so
    without that cache the 52 MB checkpoint would reload on every click.

    When a model path is supplied the caller is asking for REAL detections, so a
    load failure must surface as an exception — never a silent demo fallback, or
    an operator would mistake fabricated boxes for genuine ones.
    """
    from ultralytics import YOLO
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"Model yüklenemedi: {model_path} ({exc})") from exc
    log.info("Model loaded from %s", model_path)
    return model


def downscale(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Shrink so the longest side is ``max_side`` px, preserving aspect ratio.

    Memory guard for the multi-image flow: a session of 20 phone photos held at
    full resolution is hundreds of MB of numpy arrays. The whole pipeline then
    works in this reduced space (detection, slots, annotation), so no coordinate
    remapping is needed. Never upscales. Run QR reading on the FULL-res decode
    first — small QR labels need the pixels — then downscale for everything else.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def run_detection(
    model,
    image: np.ndarray,
    conf_threshold: float,
    imgsz: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run YOLO on one BGR image; return ``(detections, class_names)``.

    Lean detection-only helper (no summary, no annotation) shared by
    ``predict_shelf`` and the orientation first pass — the latter passes a small
    ``imgsz`` (e.g. 320) since it only needs product centres. Labels come from
    the checkpoint's own ``model.names``, the single source of truth for IDs.
    """
    model_names: dict[int, str] = model.names
    class_names = [model_names[i] for i in sorted(model_names)]

    kwargs: dict[str, Any] = {"conf": conf_threshold, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    # empty_slot recall is catastrophic (~0.12-0.21) — the class is dropped at
    # inference time entirely; downstream (slot_assigner) infers "no product
    # detected" from the lattice instead. Filtered by name, not a hardcoded
    # index, so this keeps working if a retrain reorders model.names.
    keep_ids = [i for i in sorted(model_names) if model_names[i] != "empty_slot"]
    if len(keep_ids) != len(model_names):
        kwargs["classes"] = keep_ids
    results = model.predict(image, **kwargs)[0]

    detections: list[dict[str, Any]] = []
    if results.boxes is not None:
        for box in results.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "category": model_names.get(cid, str(cid)),
                "confidence": round(conf, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
    return detections, class_names


def _demo_result(image: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    """Grid-based demo detections that look like a real vending machine shelf.

    Products are arranged in rows and columns, sized proportionally to the
    image, and assigned a realistic spread of confidence scores so the
    threshold slider produces visible, graduated changes.

    Seeded from the image dimensions so the same image always yields the
    same layout — but the caller's conf_threshold filter then controls
    how many are actually shown.
    """
    import random

    h, w = image.shape[:2]
    # Seed from image shape so layout is stable for a given image size
    rng = random.Random(w * 10000 + h)

    non_empty = [c for c in class_names if c != "empty_slot"]

    num_rows = rng.randint(3, 5)
    # Vertical margin: 5% top, 5% bottom
    usable_h = 0.90
    row_h_frac = usable_h / num_rows

    detections: list[dict[str, Any]] = []

    for row in range(num_rows):
        items_in_row = rng.randint(4, 8)
        # Horizontal margin: 5% each side → 90% usable width
        item_w_frac = 0.90 / items_in_row

        # Each row tends to carry 1-2 product types (planogram grouping)
        row_cats = rng.sample(non_empty, k=rng.randint(1, 2))

        for col in range(items_in_row):
            is_empty = rng.random() < 0.15  # 15% empty slot probability
            category = "empty_slot" if is_empty else rng.choice(row_cats)

            # Confidence: spread from 0.40 → 0.97 with most detections 0.65–0.95
            # Use beta-like sampling: pick from two bands so slider has clear effect
            band = rng.random()
            if band < 0.15:
                conf = round(rng.uniform(0.40, 0.62), 2)   # low-conf tail
            elif band < 0.90:
                conf = round(rng.uniform(0.65, 0.95), 2)   # main cluster
            else:
                conf = round(rng.uniform(0.95, 0.98), 2)   # high-conf head

            # Normalised centre with small jitter
            cx_norm = 0.05 + col * item_w_frac + item_w_frac / 2 + rng.uniform(-0.005, 0.005)
            cy_norm = 0.05 + row * row_h_frac + row_h_frac / 2 + rng.uniform(-0.005, 0.005)

            # Box occupies 70–90% of its grid cell
            bw_norm = item_w_frac * rng.uniform(0.70, 0.90)
            bh_norm = row_h_frac * rng.uniform(0.60, 0.85)

            x1 = max(0, int((cx_norm - bw_norm / 2) * w))
            y1 = max(0, int((cy_norm - bh_norm / 2) * h))
            x2 = min(w, int((cx_norm + bw_norm / 2) * w))
            y2 = min(h, int((cy_norm + bh_norm / 2) * h))

            detections.append({
                "category": category,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

    return detections


def predict_shelf(
    image_path: str | Path | np.ndarray,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.5,
    config_path: str | Path = "config/config.yaml",
    model: Any = None,
    imgsz: int | None = None,
) -> dict[str, Any]:
    """Run inference on a shelf image and return a structured result.

    Args:
        image_path: Path to a shelf image, or a BGR numpy array.
        model_path: Path to ``best.pt`` weights. If None (and ``model`` is None),
            uses demo mode. If a path is given it must load or the call raises
            (no demo fallback).
        conf_threshold: Single global minimum confidence for every class.
        config_path: Path to config.yaml.
        model: Pre-loaded YOLO model (from ``load_model``, typically cached by
            the dashboard). When supplied it is used directly and ``model_path``
            is ignored — no per-call reload.
        imgsz: Optional YOLO inference size forwarded to ``model.predict``.

    Returns:
        Dict with keys ``image_path``, ``detections``, ``summary``,
        ``total_products``, ``overall_confidence``, ``annotated_image``.
        No ``shelf_fill_rate``/``total_empty`` — those are lattice concerns,
        see ``src.analysis.shelf_analyzer.analyze_shelf``.
    """
    cfg = _load_config(config_path)
    class_names: list[str] = cfg["classes"]

    # Load image
    if isinstance(image_path, np.ndarray):
        image = image_path
        src_path = "array_input"
    else:
        src_path = str(image_path)
        image = cv2.imread(src_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {src_path}")

    # Run inference. Demo detections are produced ONLY when no weights were
    # requested at all (model_path is None). If a path was given it must load
    # and run — a missing file or a load error raises, never a silent fallback
    # to fabricated boxes an operator could mistake for real detections.
    demo_mode = False
    if model is None and model_path is None:
        log.warning("No model supplied — using demo detections")
        raw_detections = _demo_result(image, class_names)
        demo_mode = True
    else:
        if model is None:
            if not Path(str(model_path)).exists():
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            model = load_model(model_path)

        # Trust the checkpoint's own class names — config.yaml's list can
        # belong to a different training run and silently mislabel IDs.
        raw_detections, class_names = run_detection(
            model, image, conf_threshold, imgsz=imgsz
        )

    # Trim by the global confidence cutoff and build the summary.
    detections = filter_detections_by_confidence(raw_detections, conf_threshold)

    by_category: dict[str, list[float]] = defaultdict(list)
    for det in detections:
        by_category[det["category"]].append(det["confidence"])

    summary: dict[str, dict[str, Any]] = {}
    for cat, confs in by_category.items():
        summary[cat] = {
            "count": len(confs),
            "avg_confidence": round(float(np.mean(confs)), 4),
        }

    # total_products excludes empty_slot for demo-mode compatibility (demo still
    # fabricates empty_slot boxes); real inference never emits the class at all
    # now, so this is just len(detections) in practice.
    total_products = sum(v["count"] for k, v in summary.items() if k != "empty_slot")

    all_confs = [d["confidence"] for d in detections]
    overall_confidence = round(float(np.mean(all_confs)), 4) if all_confs else 0.0

    annotated = draw_detections(image, detections, class_names, conf_threshold)

    # Doluluk (shelf_fill_rate) artık BURADA hesaplanmıyor — bu, tespit bulutu
    # değil lattice/slot (src/analysis/shelf_analyzer.py + slot_assigner.py) ilgi
    # alanı. empty_slot dedeksiyonu güvenilmez olduğundan (bkz. modül üstü not)
    # buradan kaldırıldı; tüketiciler analyze_shelf()'in lattice-tabanlı
    # shelf_fill_rate/total_empty alanlarını kullanmalı.
    return {
        "image_path": src_path,
        "demo_mode": demo_mode,
        "detections": detections,
        "summary": summary,
        "total_products": total_products,
        "overall_confidence": overall_confidence,
        "annotated_image": annotated,
    }


def main():
    import click

    @click.command()
    @click.argument("image_path", type=click.Path(exists=True))
    @click.option("--model", "model_path", default=None, type=click.Path())
    @click.option("--conf", "conf_threshold", default=0.5, show_default=True)
    @click.option("--config", default="config/config.yaml", show_default=True)
    @click.option("--output-image", default=None, type=click.Path(), help="Save annotated image here")
    def cli(image_path, model_path, conf_threshold, config, output_image):
        """Run shelf inference and print JSON result."""
        result = predict_shelf(image_path, model_path, conf_threshold, config)
        annotated = result.pop("annotated_image")
        print(json.dumps(result, indent=2))
        if output_image:
            cv2.imwrite(output_image, annotated)
            log.info("Annotated image saved → %s", output_image)

    cli()


if __name__ == "__main__":
    main()
