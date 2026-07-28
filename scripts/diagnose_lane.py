"""Lane (zincir/latis) hattı teşhis scripti — CANLI dashboard hattını taklit eder.

diagnose_grid.py'den farkı: bu, üretim yolunu (estimate_grid_lane_based →
build_slot_report) çalıştırır ve her görsel için SINIR-MERKEZ İHLALİ sayısını
ölçer (bir lane sınır doğrusu bir tespit kutusunun içinden/merkezinden geçiyor mu).
P1 (blok-birleşme), P2 (kayma/karışım), P3 (boş) sinyallerini CSV'ye yazar.

Kullanım:
    venv.nosync/bin/python scripts/diagnose_lane.py \
        --images /Users/gulfemsolak/Desktop/train \
        --out /tmp/lane_diag --limit 0 --tag before
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from PIL import ExifTags, Image

from src.models.predictor import predict_shelf, load_model
from src.analysis.skew_corrector import deskew
from src.analysis.product_anchored_grid import estimate_grid_lane_based
from src.analysis.slot_assigner import build_slot_report, EMPTY_CLASS_NAME
from src.utils.visualization import draw_slots

_EXIF_ORIENTATION_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for p in sorted(paths, key=lambda x: (len(x.name), x.name)):
        h = _md5(p)
        if h not in seen:
            seen[h] = p
    return sorted(seen.values(), key=lambda x: (len(x.name), x.name))


def _exif_orientation(path: Path) -> int | None:
    try:
        exif = Image.open(path).getexif()
        return exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    except Exception:
        return None


def _boundary_center_violations(grid: dict, detections: list[dict]) -> int:
    """Kaç tespit kutusunun İÇİNDEN bir iç lane sınırı geçiyor (P2 kök göstergesi).

    Her tespiti (t,s) uzayına taşı; kutunun u-yönü yarı-genişliğini hesapla; herhangi
    bir İÇ sınır (ilk/son hariç) o t'de kutunun s-aralığına düşüyorsa ihlal say.
    """
    if grid.get("grid_status") != "ok" or not grid.get("boundary_lines"):
        return 0
    origin, v, u = grid["axis_origin"], grid["axis_v"], grid["axis_u"]
    boundaries = grid["boundary_lines"]
    if len(boundaries) <= 2:
        return 0
    interior = boundaries[1:-1]
    viol = 0
    for d in detections:
        if d.get("category") == EMPTY_CLASS_NAME:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        t = (cx - origin[0]) * v[0] + (cy - origin[1]) * v[1]
        s = (cx - origin[0]) * u[0] + (cy - origin[1]) * u[1]
        half = (abs(u[0]) * (x2 - x1) + abs(u[1]) * (y2 - y1)) / 2.0
        # "merkezden geçme": sınır kutunun MERKEZİ çevresine (±%20 yarı-genişlik) düşerse.
        # Kenara yakın teğet geçişler (ürünler görüntüde üst üste biner) ihlal SAYILMAZ.
        core = 0.2 * half
        for b in interior:
            bs = b["a"] + b["b"] * t
            if abs(bs - s) <= core:
                viol += 1
                break
    return viol


def _numbering_direction_wrong(grid: dict, slots: list[dict]) -> bool | None:
    """KALICI metrik (GÖREV 1): atanmış ``column_no`` gerçek görüntü konumuyla doğru
    yönde mi artıyor? Düzen-farkında (sütun→x, satır→y — bkz. ``_lane_grid``'deki
    ``NUMBERING_AXIS_AMBIGUITY_EPS`` mantığıyla AYNI eksen seçimi) ve BAĞIMSIZ
    doğrulama: ``grid["numbering_reversed"]`` bayrağının kendisine değil, atanmış
    ``slots`` listesinin GERÇEK sonucuna bakar (bayrağın kendisinde bug olsa da yakalar).
    Yalnız ~45° belirsizlik bandı dışında hesaplanır; içinde/uygulanamaz durumda ``None``."""
    if grid.get("grid_status") != "ok" or grid.get("grid_source") != "zincir" or not slots:
        return None
    u = grid.get("axis_u")
    origin = grid.get("axis_origin")
    if not u or not origin:
        return None
    if abs(abs(u[0]) - abs(u[1])) < 0.1:
        return None
    column_oriented = abs(u[0]) >= abs(u[1])
    ordered = sorted(slots, key=lambda s: s["column_no"])
    if column_oriented:
        proj = [origin[0] + s["x"] * u[0] for s in ordered]
    else:
        proj = [origin[1] + s["x"] * u[1] for s in ordered]
    n = len(proj)
    if n < 2 or len(set(proj)) <= 1:
        return None
    mean_i = (n - 1) / 2.0
    mean_p = sum(proj) / n
    cov = sum((i - mean_i) * (p - mean_p) for i, p in enumerate(proj))
    return cov < 0


def _axis_dump_row(fname: str, grid: dict) -> dict | None:
    """GÖREV 1 teşhisi: slot numaralandırma yönünün hangi eksene bağlı olduğunu dök.
    ``grid_source != 'zincir'`` ise (homografi/manuel yol farklı numaralandırma kullanır)
    None döner."""
    if grid.get("grid_source") != "zincir" or not grid.get("centers"):
        return None
    meta = grid.get("lane_meta", {})
    origin = grid["axis_origin"]
    u = grid["axis_u"]
    centers = grid["centers"]
    theta = meta.get("angle_deg")
    phi = round(math.degrees(math.atan2(u[1], u[0])), 2)
    n_x_sign = 1 if u[0] > 0 else (-1 if u[0] < 0 else 0)

    # Slot merkezlerinin GERÇEK görüntü x'i (t=0 varsayımıyla — merkezler yalnız s
    # boyunca değişir, x = origin_x + s*u_x tam olarak doğrusal).
    xs = [origin[0] + s * u[0] for s in centers]
    # s-sırası (centers zaten artan) ile x-sırasının korelasyon işareti (Pearson).
    n = len(xs)
    idx = list(range(n))
    if n >= 2 and len(set(xs)) > 1:
        mean_i = sum(idx) / n
        mean_x = sum(xs) / n
        cov = sum((i - mean_i) * (x - mean_x) for i, x in zip(idx, xs))
        corr_sign = 1 if cov > 0 else (-1 if cov < 0 else 0)
    else:
        corr_sign = 0

    return {
        "file": fname,
        "theta_deg": theta,
        "phi_deg": phi,
        "n_x_sign": n_x_sign,
        "s_vs_x_corr_sign": corr_sign,
        "first_slot_x": round(xs[0], 1),
        "last_slot_x": round(xs[-1], 1),
        "dominant_branch": "horizontal" if abs(u[0]) >= abs(u[1]) else "vertical",
    }


def run(images_dir: Path, out_dir: Path, limit: int, tag: str, only: list[str],
        expected_k: int | None = None, dump_axis: bool = False) -> None:
    cfg = yaml.safe_load((Path("config/config.yaml")).read_text())
    slot_cfg = cfg.get("slot_analysis", {})
    column_count = int(slot_cfg.get("column_count", 7))

    paths = [p for p in images_dir.iterdir()
             if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
    paths = _dedupe(paths)
    if only:
        only_set = set(only)
        paths = [p for p in paths if p.name in only_set or p.stem in only_set]
    if limit > 0:
        paths = paths[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / f"overlays_{tag}"
    overlay_dir.mkdir(exist_ok=True)

    model = load_model("models/best.pt")
    rows: list[dict] = []
    axis_rows: list[dict] = []

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue

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
        grid = estimate_grid_lane_based(work, column_count, dets, expected_column_count=expected_k)
        rep = build_slot_report(dets, shelf_number=1, column_count=column_count,
                                grid=grid, empty_class_name=EMPTY_CLASS_NAME)

        if dump_axis:
            axis_row = _axis_dump_row(p.name, grid)
            if axis_row is not None:
                axis_rows.append(axis_row)
        used = rep["grid"]
        slots = rep["slots"]
        counts = [s["count"] for s in slots]
        products = [d for d in dets if d.get("category") != EMPTY_CLASS_NAME]
        viol = _boundary_center_violations(used, dets)
        n_mixed = sum(1 for s in slots if s.get("mixed"))
        n_empty = sum(1 for s in slots if s.get("is_empty"))
        meta = used.get("lane_meta", {})
        dir_wrong = _numbering_direction_wrong(used, slots)

        rows.append({
            "file": p.name,
            "grid_source": used.get("grid_source"),
            "grid_status": rep["grid_status"],
            "angle_deg": meta.get("angle_deg"),
            "pitch": meta.get("pitch"),
            "pitch_comb_used": meta.get("pitch_comb_used"),
            "theta_search_used": meta.get("theta_search_used"),
            "fixed_k_used": meta.get("fixed_k_used"),
            "n_boundary_snaps": meta.get("n_boundary_snaps"),
            "n_products": len(products),
            "n_slots": len(slots),
            "nonempty_slots": sum(1 for c in counts if c > 0),
            "empty_slots": n_empty,
            "max_slot_count": max(counts) if counts else 0,
            "mixed_slots": n_mixed,
            "boundary_center_violations": viol,
            "alignment_suspects": len(rep.get("alignment_suspects", [])),
            "slot_counts": "|".join(str(c) for c in counts),
            "numbering_direction_wrong": dir_wrong,
        })

        base = result.get("annotated_image")
        base = base if base is not None else work
        overlay = draw_slots(base, rep)
        cv2.imwrite(str(overlay_dir / p.name), overlay)

        if (i + 1) % 10 == 0:
            print(f"...{i + 1}/{len(paths)} işlendi")

    if not rows:
        print("hiç görsel işlenmedi")
        return

    csv_path = out_dir / f"lane_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print("\n" + "=" * 64)
    print(f"LANE TEŞHİS [{tag}] — {n} görsel · CSV: {csv_path}")
    print("=" * 64)
    print(f"grid_source: {dict(Counter(r['grid_source'] for r in rows))}")
    print(f"grid_status: {dict(Counter(r['grid_status'] for r in rows))}")
    tot_viol = sum(r["boundary_center_violations"] for r in rows)
    imgs_viol = sum(1 for r in rows if r["boundary_center_violations"] > 0)
    print(f"\nP2 sınır-merkez ihlali: toplam={tot_viol}, {imgs_viol}/{n} görselde")
    tot_mixed = sum(r["mixed_slots"] for r in rows)
    imgs_mixed = sum(1 for r in rows if r["mixed_slots"] > 0)
    print(f"P2 karışık slot: toplam={tot_mixed}, {imgs_mixed}/{n} görselde")
    dir_applicable = [r for r in rows if r["numbering_direction_wrong"] is not None]
    dir_wrong_n = sum(1 for r in dir_applicable if r["numbering_direction_wrong"])
    print(f"Numaralandırma yönü YANLIŞ: {dir_wrong_n}/{len(dir_applicable)} "
          f"(uygulanamaz/belirsiz: {n - len(dir_applicable)})")
    dense = [r for r in rows if r["max_slot_count"] >= 8]
    print(f"\nP1 tek slotta 8+ ürün (blok-birleşme şüphesi): {len(dense)} görsel")
    for r in sorted(dense, key=lambda x: -x["max_slot_count"])[:15]:
        print(f"  {r['file']:16s} max={r['max_slot_count']:2d} slots={r['n_slots']} "
              f"viol={r['boundary_center_violations']} counts={r['slot_counts']}")

    if dump_axis and axis_rows:
        axis_csv_path = out_dir / f"axis_{tag}.csv"
        with open(axis_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(axis_rows[0].keys()))
            w.writeheader()
            w.writerows(axis_rows)
        reversed_rows = [r for r in axis_rows if r["s_vs_x_corr_sign"] < 0]
        print(f"\nGÖREV 1 eksen teşhisi: {len(axis_rows)} foto (zincir) · CSV: {axis_csv_path}")
        print(f"s-x korelasyonu TERS (sağdan-sola numaralanmış): {len(reversed_rows)}/{len(axis_rows)}")
        print(f"  dominant_branch dağılımı (ters vakalar): "
              f"{dict(Counter(r['dominant_branch'] for r in reversed_rows))}")
        print(f"  n_x_sign dağılımı (ters vakalar): "
              f"{dict(Counter(r['n_x_sign'] for r in reversed_rows))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--expected-k", type=int, default=None,
                     help="QR/tepsi-barından okunmuş gibi K (kanal sayısı) simüle et — "
                          "gerçek okuma bağlanana kadar K-kısıtlı comb'u elle test aracı.")
    ap.add_argument("--dump-axis", action="store_true",
                     help="GÖREV 1 teşhisi: θ/φ/n_x işareti/s-x korelasyonu foto başına dök.")
    a = ap.parse_args()
    run(Path(a.images), Path(a.out), a.limit, a.tag, a.only, expected_k=a.expected_k,
        dump_axis=a.dump_axis)


if __name__ == "__main__":
    main()
