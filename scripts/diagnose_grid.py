"""Slot ızgarası teşhis scripti — gerçek raf fotoğraflarında homografi hattını ölçer.

Dashboard hattını birebir taklit eder: EXIF çıkarımı → deskew (EXIF-only) →
predict_shelf → estimate_grid (tepsi köşesi → homografi) → build_slot_report.
Her görsel için tek satır CSV + overlay üretir ve toplu istatistik raporlar.

Kullanım:
    venv.nosync/bin/python scripts/diagnose_grid.py \
        --images /Users/gulfemsolak/Desktop/train \
        --out /tmp/grid_diag --limit 0 --tag before
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from PIL import ExifTags, Image

from src.models.predictor import predict_shelf, load_model
from src.analysis.skew_corrector import deskew
from src.analysis.shelf_grid import estimate_grid
from src.analysis.slot_assigner import build_slot_report, EMPTY_CLASS_NAME
from src.utils.visualization import draw_detections, draw_slots

CAPACITY = 12  # kolon başına yaklaşık fiziksel kapasite
_EXIF_ORIENTATION_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for p in sorted(paths, key=lambda x: (len(x.name), x.name)):
        h = _md5(p)
        if h not in seen:
            seen[h] = p
    return sorted(seen.values(), key=lambda x: x.name)


def _exif_orientation(path: Path) -> int | None:
    try:
        exif = Image.open(path).getexif()
        return exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    except Exception:
        return None


def run(images_dir: Path, out_dir: Path, limit: int, tag: str) -> None:
    cfg = yaml.safe_load((Path("config/config.yaml")).read_text())
    slot_cfg = cfg.get("slot_analysis", {})
    column_count = int(slot_cfg.get("column_count", 7))

    paths = [p for p in images_dir.iterdir()
             if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
    paths = _dedupe(paths)
    if limit > 0:
        paths = paths[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / f"overlays_{tag}"
    overlay_dir.mkdir(exist_ok=True)

    model = load_model("models/best.pt")
    rows: list[dict] = []

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue

        # Yeni hat: EXIF-only coarse döndürme → tespit → tepsi köşesi → homografi.
        exif_tag = _exif_orientation(p)
        dk = deskew(img, exif_orientation=exif_tag)
        work = dk["image"]
        try:
            result = predict_shelf(image_path=work, model=model,
                                   conf_threshold=0.5, config_path="config/config.yaml")
        except Exception as exc:
            print(f"[{p.name}] tespit hatası: {exc}")
            continue

        dets = result["detections"]
        grid = estimate_grid(work, column_count)
        rep = build_slot_report(dets, shelf_number=1, column_count=column_count,
                               grid=grid, empty_class_name=EMPTY_CLASS_NAME)
        used = rep["grid"]

        slots = rep["slots"]
        counts = [s["count"] for s in slots]
        nonempty = sum(1 for c in counts if c > 0)
        max_count = max(counts) if counts else 0
        products = [d for d in dets if d.get("category") != EMPTY_CLASS_NAME]
        assigned_total = sum(counts)

        rows.append({
            "file": p.name,
            "img_w": img.shape[1], "img_h": img.shape[0],
            "exif_tag": exif_tag if exif_tag is not None else "",
            "coarse": dk["coarse"],
            "orient_source": dk["orient_source"],
            "grid_source": used.get("grid_source"),
            "grid_status": rep["grid_status"],
            "shelf_number_from_bar": used.get("shelf_number_from_bar"),
            "pitch": round(used["pitch"], 1) if used.get("pitch") else "",
            "rect_width": round(used["rect_width"], 1) if used.get("rect_width") else "",
            "n_detections": len(dets),
            "n_products": len(products),
            "nonempty_slots": nonempty,
            "max_slot_count": max_count,
            "slot_counts": "|".join(str(c) for c in counts),
            "alignment_suspects": len(rep.get("alignment_suspects", [])),
            "sum_eq_len": (assigned_total == len(products)) if rep["grid_status"] == "ok" else True,
        })

        # Overlay: tespit + ızgara (belirsizde yalnız tespit)
        base = result.get("annotated_image")
        base = base if base is not None else work
        overlay = draw_slots(base, rep)
        cv2.imwrite(str(overlay_dir / p.name), overlay)

        if (i + 1) % 10 == 0:
            print(f"...{i + 1}/{len(paths)} işlendi")

    # CSV yaz
    csv_path = out_dir / f"diag_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _report(rows, csv_path, overlay_dir, tag)


def _report(rows: list[dict], csv_path: Path, overlay_dir: Path, tag: str) -> None:
    n = len(rows)
    print("\n" + "=" * 64)
    print(f"TEŞHİS RAPORU [{tag}] — {n} tekil görsel")
    print("=" * 64)
    print(f"CSV: {csv_path}")
    print(f"Overlay: {overlay_dir}")

    src_dist = Counter(r["grid_source"] for r in rows)
    print("\ngrid_source dağılımı:")
    for k, v in src_dist.most_common():
        print(f"  {k:12s} {v:4d}  ({100 * v / n:.0f}%)")

    status_dist = Counter(r["grid_status"] for r in rows)
    print("\ngrid_status dağılımı:")
    for k, v in status_dist.most_common():
        print(f"  {k:12s} {v:4d}  ({100 * v / n:.0f}%)")

    coarse_dist = Counter(r["coarse"] for r in rows)
    print("\ncoarse (EXIF) dağılımı:")
    for k in sorted(coarse_dist):
        print(f"  {k:3d}°  {coarse_dist[k]:4d}")
    exif_rotated = sum(1 for r in rows if r["coarse"] != 0)
    print(f"EXIF ile döndürülen: {exif_rotated} görsel (hedef: yalnız gerçekten EXIF'i öyle olanlar)")

    bar_read = sum(1 for r in rows if r["shelf_number_from_bar"] not in (None, ""))
    print(f"\nraf no tepsi barından okundu: {bar_read}/{n} (%{100*bar_read/n:.0f})")

    dense = [r for r in rows if r["max_slot_count"] >= 15]
    print(f"\ntek slotta 15+ ürün : {len(dense)} görsel")
    for r in dense[:20]:
        print(f"  {r['file']:16s} max={r['max_slot_count']:2d}  counts={r['slot_counts']}")

    ne = Counter(r["nonempty_slots"] for r in rows)
    print("\nboş-olmayan slot sayısı dağılımı:")
    for k in sorted(ne):
        print(f"  {k} slot dolu : {ne[k]:4d} görsel")

    broken = [r for r in rows if not r["sum_eq_len"]]
    print(f"\nsum(counts) != len(products) : {len(broken)} görsel (0 olmalı)")

    susp = sum(1 for r in rows if r["alignment_suspects"] > 0)
    print(f"alignment_suspect işaretli   : {susp} görsel")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    run(Path(a.images), Path(a.out), a.limit, a.tag)


if __name__ == "__main__":
    main()
