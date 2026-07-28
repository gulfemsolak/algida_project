"""Visualization helpers: bounding boxes, annotation grids, comparison layouts."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# Per-category BGR colours, matching dashboard.theme.PRODUCT_COLORS (hex RGB → BGR)
# so a box drawn on the annotated image matches its dot color in the dashboard UI.
_CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "algida_boom_gold":        (74,  201, 255),
    "algida_klasik":           (230, 209, 90),
    "algida_klasik_cikolata":  (42,  74,  122),
    "cornetto_cilek":          (121, 79,  255),
    "cornetto_findik_krema":   (88,  137, 198),
    "cornetto_fistik":         (74,  195, 139),
    "cornetto_karamel":        (60,  155, 216),
    "cornetto_kaymak":         (196, 226, 242),
    "cornetto_mango":          (48,  166, 255),
    "cornetto_oreo":           (116, 104, 107),
    "empty_slot":              (80,  48,  42),
    "frigola_bar":             (151, 220, 61),
    "frigola_klasik":          (160, 180, 0),
    "magnum_ahududu":          (91,  53,  224),
    "magnum_badem":            (106, 155, 205),
    "magnum_beyaz":            (230, 239, 242),
    "magnum_cookie":           (68,  107, 156),
    "magnum_double_cikolata":  (32,  54,  90),
    "magnum_dubai":            (201, 99,  145),
    "magnum_karadut":          (142, 58,  122),
    "magnum_karamel":          (51,  115, 184),
    "magnum_klasik":           (19,  69,  139),
    "maras_usulu_kutu":        (138, 201, 183),
    "nogger_karamel":          (93,  154, 199),
    "nogger_vanilya":          (180, 217, 235),
    "twister_kavun":           (76,  217, 180),
    "twister_orman":           (99,  43,  160),
    "viennetta":               (126, 199, 232),
}

# Fallback palette for unknown classes
_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 215, 255),
    (255, 0, 255), (255, 255, 0), (0, 102, 255), (255, 51, 153),
    (136, 255, 0), (102, 51, 255), (255, 204, 51),
]


def class_color(class_id: int, class_name: str = "") -> tuple[int, int, int]:
    """Return a consistent BGR colour — category-specific when possible."""
    if class_name in _CATEGORY_COLORS:
        return _CATEGORY_COLORS[class_name]
    return _PALETTE[class_id % len(_PALETTE)]


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white text depending on background brightness."""
    # Perceived luminance using standard coefficients (on BGR values)
    b, g, r = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    class_names: list[str],
    conf_threshold: float = 0.0,
    line_thickness: int | None = None,
    font_scale: float | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of *image*.

    Thickness and font scale auto-scale with image dimensions when not
    explicitly provided, so boxes look correct regardless of resolution.

    Args:
        image: BGR numpy array.
        detections: List of dicts with keys ``category``, ``confidence``, ``bbox``.
        class_names: Ordered list of class name strings.
        conf_threshold: Skip detections below this confidence.
        line_thickness: Box border width; None = auto (scales with image size).
        font_scale: OpenCV font scale; None = auto.

    Returns:
        Annotated BGR image copy.
    """
    out = image.copy()
    h, w = out.shape[:2]
    short_side = min(w, h)

    # Auto-scale: ~0.3% of short side for thickness, clamped to [3, 8]
    thickness = line_thickness if line_thickness is not None else max(3, int(short_side * 0.003))
    # Auto-scale font: ~0.06% of short side, clamped to [0.45, 1.2]
    fscale = font_scale if font_scale is not None else max(0.45, min(1.2, short_side * 0.0006))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(1, thickness // 2)

    for det in detections:
        if det["confidence"] < conf_threshold:
            continue

        cat = det["category"]
        # Katman 5: yalnızca ÜRÜNLER kutulanır. empty_slot kutuları görüntüyü
        # karıştırır ve yalnızca slot tespitinde (Katman 2/3) içeride kullanılır.
        if cat == "empty_slot":
            continue
        cid = class_names.index(cat) if cat in class_names else 0
        color = class_color(cid, cat)

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Build label: "category_name 0.XX"
        label = f"{cat} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, fscale, font_thickness)

        # Label background rectangle sits fully above the box top edge
        pad = 4
        label_y1 = max(0, y1 - th - baseline - pad * 2)
        label_y2 = y1
        label_x2 = x1 + tw + pad * 2

        # Filled background rectangle
        cv2.rectangle(out, (x1, label_y1), (label_x2, label_y2), color, -1)

        # Text colour chosen for readability against the filled background
        txt_color = _text_color(color)
        cv2.putText(
            out, label,
            (x1 + pad, label_y2 - baseline - 2),
            font, fscale, txt_color, font_thickness, cv2.LINE_AA,
        )

    return out


# Soğuk Zincir paleti (dashboard/theme.py hex'lerinin BGR karşılığı).
_ICE_BGR = (255, 184, 92)     # #5CB8FF — nötr şerit/ızgara rengi
_ALGIDA_RED_BGR = (85, 51, 255)   # #FF3355 — dolu slot
_AMBER_BGR = (71, 181, 255)   # #FFB547 — hizalama şüpheli / uyarı


def _slot_color(slot: dict[str, Any], default: tuple[int, int, int]) -> tuple[int, int, int]:
    if slot.get("alignment_suspect"):
        return _AMBER_BGR
    if slot.get("is_empty"):
        return _ICE_BGR
    return default


def _dashed_polyline(
    img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int],
    thickness: int, dash: int = 14, gap: int = 10,
) -> None:
    """Boş slotlar için soluk/kesikli çerçeve (cv2'de yerleşik dashed-line yok)."""
    n = len(pts)
    for i in range(n):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[(i + 1) % n], dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-6:
            continue
        direction = (p2 - p1) / length
        step = dash + gap
        k = 0
        while k * step < length:
            start = p1 + direction * (k * step)
            end = p1 + direction * min(k * step + dash, length)
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)
            k += 1


def draw_slots(
    image: np.ndarray,
    slot_report: dict[str, Any],
    color: tuple[int, int, int] = _ALGIDA_RED_BGR,
    line_thickness: int | None = None,
    font_scale: float | None = None,
) -> np.ndarray:
    """Slot (lane) sınırlarını *image* üzerine çizer — mevcut tespit çizimini korur.

    TEK KAYNAK İLKESİ: çizilen sınırlar/numaralar, atama ile TAM OLARAK aynı ``grid``
    değerlerinden üretilir. İki grid şeması desteklenir:

      * ``grid_source == "zincir"`` (PRIMARY yol): her lane KENDİ doğrultusunda bir
        trapez şerit olarak çizilir (``boundary_lines`` + ``axis_origin/v/u`` —
        homografi YOK). Şeritler afin/lokal-doğrusal olduğu için birbirini kesmez;
        komşu lane'ler hafif yelpaze açabilir ama HER ZAMAN paralel-benzeri kalır.
      * Diğerleri (``kontur``/``manuel``): eski ``H_inv`` reprojeksiyonu.

    Dolu slotlar kırmızı (ALGIDA_RED), hizalama şüpheli olanlar amber, boş slotlar
    soluk/kesikli ice çerçeve ile çizilir (Soğuk Zincir teması). Slot kimliği şeridin
    üst ucuna yazılır.

    Args:
        image: BGR numpy array (genellikle ``draw_detections`` çıktısı — üstüne eklenir).
        slot_report: ``slot_assigner.build_slot_report`` / ``analyze_slots`` çıktısı.
        color: dolu slotların varsayılan çizgi/etiket rengi (BGR).

    Returns:
        Slot sınırları eklenmiş BGR kopya. Izgara yoksa (belirsiz) görüntü değişmeden döner.
    """
    out = image.copy()
    slots = slot_report.get("slots", [])
    grid = slot_report.get("grid")
    if not slots or not grid:
        return out

    if grid.get("grid_source") == "zincir" and grid.get("boundary_lines"):
        return _draw_slots_chain(out, slots, grid, color, line_thickness, font_scale)

    if grid.get("H_inv") is not None:
        return _draw_slots_homography(out, slots, grid, color, line_thickness, font_scale)

    return out


def _draw_slots_homography(
    out: np.ndarray, slots: list[dict[str, Any]], grid: dict[str, Any],
    color: tuple[int, int, int], line_thickness: int | None, font_scale: float | None,
) -> np.ndarray:
    """Eski yol: rektifiye uzaydan ``H_inv`` ile geri-izdüşüm (yalnız ``kontur``/``manuel``)."""
    from src.analysis.shelf_grid import transform_point

    h, w = out.shape[:2]
    short_side = min(w, h)
    thickness = line_thickness if line_thickness is not None else max(2, int(short_side * 0.004))
    fscale = font_scale if font_scale is not None else max(0.6, min(2.4, short_side * 0.0017))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = max(2, thickness)

    h_inv = grid["H_inv"]
    rect_h = grid["rect_height"]

    def _reproject(x_rect: float, y_rect: float) -> tuple[int, int]:
        x, y = transform_point(x_rect, y_rect, h_inv)
        return int(round(x)), int(round(y))

    for slot in slots:
        x_l = slot.get("x_left")
        x_r = slot.get("x_right")
        if x_l is None or x_r is None:
            continue

        tl = _reproject(x_l, 0)
        tr = _reproject(x_r, 0)
        br = _reproject(x_r, rect_h)
        bl = _reproject(x_l, rect_h)
        pts = np.array([tl, tr, br, bl], dtype=np.int32).reshape(-1, 1, 2)
        slot_color = _slot_color(slot, color)
        cv2.polylines(out, [pts], isClosed=True, color=slot_color, thickness=thickness, lineType=cv2.LINE_AA)

        label = str(slot["slot_id"])
        (tw, th), _ = cv2.getTextSize(label, font, fscale, font_thick)
        cx = (tl[0] + tr[0]) // 2
        cy = (tl[1] + tr[1]) // 2
        tx = max(2, int(cx - tw / 2))
        ty = int(cy + th + 8)
        cv2.putText(out, label, (tx, ty), font, fscale, (0, 0, 0), font_thick + 3, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, fscale, slot_color, font_thick, cv2.LINE_AA)

    return out


def _draw_slots_chain(
    out: np.ndarray, slots: list[dict[str, Any]], grid: dict[str, Any],
    color: tuple[int, int, int], line_thickness: int | None, font_scale: float | None,
) -> np.ndarray:
    """Yeni yol: her lane kendi doğrultusunda trapez — homografi YOK, kesişme YOK."""
    h, w = out.shape[:2]
    short_side = min(w, h)
    thickness = line_thickness if line_thickness is not None else max(2, int(short_side * 0.004))
    fscale = font_scale if font_scale is not None else max(0.6, min(2.4, short_side * 0.0017))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = max(2, thickness)

    origin, v, u = grid["axis_origin"], grid["axis_v"], grid["axis_u"]
    t_min, t_max = grid["t_min"], grid["t_max"]
    boundaries = grid["boundary_lines"]
    edge_left, edge_right = grid.get("edge_left"), grid.get("edge_right")

    def _point(t: float, s: float) -> tuple[int, int]:
        x = origin[0] + t * v[0] + s * u[0]
        y = origin[1] + t * v[1] + s * u[1]
        return int(round(x)), int(round(y))

    def _clip(s: float, t: float, edge: dict[str, Any] | None, side: str) -> float:
        # Hiçbir şerit tepsi kenarının DIŞINA çizilmez — kenar biliniyorsa s'yi kırp.
        if edge is None:
            return s
        edge_s = edge["a"] + edge["b"] * t
        return max(s, edge_s) if side == "left" else min(s, edge_s)

    for i, slot in enumerate(slots):
        if i + 1 >= len(boundaries):
            continue
        bl, br = boundaries[i], boundaries[i + 1]
        s_l_min, s_l_max = bl["a"] + bl["b"] * t_min, bl["a"] + bl["b"] * t_max
        s_r_min, s_r_max = br["a"] + br["b"] * t_min, br["a"] + br["b"] * t_max
        s_l_min = _clip(s_l_min, t_min, edge_left, "left")
        s_l_max = _clip(s_l_max, t_max, edge_left, "left")
        s_r_min = _clip(s_r_min, t_min, edge_right, "right")
        s_r_max = _clip(s_r_max, t_max, edge_right, "right")

        p1 = _point(t_min, s_l_min)   # sol-alt (t_min ucu)
        p2 = _point(t_min, s_r_min)   # sağ-alt
        p3 = _point(t_max, s_r_max)   # sağ-üst (t_max ucu)
        p4 = _point(t_max, s_l_max)   # sol-üst
        pts = [p1, p2, p3, p4]

        slot_color = _slot_color(slot, color)
        if slot.get("is_empty") and not slot.get("alignment_suspect"):
            _dashed_polyline(out, pts, slot_color, max(1, thickness - 1))
        else:
            cv2.polylines(out, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                           isClosed=True, color=slot_color, thickness=thickness, lineType=cv2.LINE_AA)

        # Etiket: t_min/t_max uçlarından resimde daha küçük y'ye (yukarıda) düşen ucun ortası.
        mid_min = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        mid_max = ((p3[0] + p4[0]) // 2, (p3[1] + p4[1]) // 2)
        label_pos = mid_min if mid_min[1] <= mid_max[1] else mid_max

        label = str(slot["slot_id"])
        (tw, th), _ = cv2.getTextSize(label, font, fscale, font_thick)
        tx = max(2, int(label_pos[0] - tw / 2))
        ty = int(label_pos[1] + th + 8)
        cv2.putText(out, label, (tx, ty), font, fscale, (0, 0, 0), font_thick + 3, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, fscale, slot_color, font_thick, cv2.LINE_AA)

    return out


def create_comparison_grid(
    images: list[np.ndarray],
    titles: list[str],
    cols: int = 2,
    target_height: int = 400,
) -> np.ndarray:
    """Arrange images in a grid with titles.

    Args:
        images: List of BGR numpy arrays.
        titles: List of title strings (same length as images).
        cols: Number of columns.
        target_height: Resize each image to this height before tiling.

    Returns:
        Single BGR grid image.
    """
    rows_count = (len(images) + cols - 1) // cols
    resized: list[np.ndarray] = []
    for img, title in zip(images, titles):
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        r = cv2.resize(img, (new_w, target_height))
        cv2.putText(
            r, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
        )
        resized.append(r)

    # Pad to fill last row
    while len(resized) % cols:
        blank = np.zeros_like(resized[0])
        resized.append(blank)

    rows: list[np.ndarray] = []
    for r in range(rows_count):
        row_imgs = resized[r * cols: (r + 1) * cols]
        # Pad widths to match within each row
        max_w = max(i.shape[1] for i in row_imgs)
        padded = []
        for img in row_imgs:
            pad_w = max_w - img.shape[1]
            padded.append(np.pad(img, ((0, 0), (0, pad_w), (0, 0))))
        rows.append(np.hstack(padded))

    # Pad row widths to match
    max_row_w = max(r.shape[1] for r in rows)
    final_rows = []
    for r in rows:
        pad_w = max_row_w - r.shape[1]
        final_rows.append(np.pad(r, ((0, 0), (0, pad_w), (0, 0))))

    return np.vstack(final_rows)


def save_annotated_image(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    class_names: list[str],
    output_path: Path | str,
    conf_threshold: float = 0.0,
) -> None:
    """Draw detections on image and save to disk."""
    annotated = draw_detections(image, detections, class_names, conf_threshold)
    cv2.imwrite(str(output_path), annotated)
    log.info("Saved annotated image → %s", output_path)
