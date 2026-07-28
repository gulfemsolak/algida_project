"""Kolon ızgarası çapası — HOMOGRAFİ ile rektifikasyon.

Tepsi HER ZAMAN ``column_count`` kolon içerir, ama yukarıdan açılı çekildiği için
lane'ler perspektifle yelpaze gibi açılır — paralel DEĞİLDİR. Sabit-pitch dikey
ızgara (afin/shear modeli) bunu temsil edemez: soldaki lane'lere hizalanan ızgara
sağda kayar. Bu modül tepsinin 4 köşesini bulup ``cv2.getPerspectiveTransform`` ile
onu bir dikdörtgene eşleyen **H** homografisini hesaplar. Atama REKTİFİYE uzayda
yapılır — lane'ler orada gerçekten eşit aralıklıdır; modelin doğru olduğu tek yer.

Köşe kaynakları, öncelik sırasıyla:
  (a) ``kontur``  Koyu tepsi gövdesinin en büyük bağlı bileşeninin konturu →
                  ``approxPolyDP`` ile 4 köşeli dışbükey poligon. Perspektif
                  trapezoidi doğrudan yakalar (minAreaRect'in aksine).
  (b) ``manuel``  Operatörün elle işaretlediği 4 köşe — otomatik kadar güvenilir.
  (c) ``belirsiz`` Hiçbiri güvenilir değilse ATAMA YAPILMAZ.

Ray (etiket rayı) hem GEOMETRİ hem RAF NUMARASI için KULLANILMAZ. ``train/``
setinde ölçüldü: numaralı ray fiziksel olarak bir üstteki rafın etiket şeridine
ait (parallax) — ör. ``9.jpeg``'de ray "21,22,23" gösterirken fotoğraflanan
tepsinin KENDİ barı "3" yazıyor; ``39.jpeg``'de ray "11..17" gösterirken tepsinin
kendi barı "2" yazıyor. Yanlış rafın geometrisini/numarasını okumak sistematik
kaymaya yol açar (off-by-one'ın kaynağı da bu olabilir). Raf numarası bunun yerine
tepsinin KENDİ barındaki tek haneli basılı sayıdan OCR edilir.

Görüntü alan tek CV modülü budur; başarısız olduğunda YOLO tespitleri yine gösterilir.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:  # cv2 yoksa (saf birim testi) modül yine import edilebilsin.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from src.utils.logger import get_logger

log = get_logger(__name__)

# Tepsi konturu guard'ları — biri düşerse köşeler reddedilir (belirsiz).
MIN_TRAY_AREA_FRAC = 0.10
MAX_TRAY_AREA_FRAC = 0.80
# Taraf oranı GENİŞ tutulur: tepsi yukarıdan açılı çekildiği için gerçek perspektifte
# sol/sağ (ya da üst/alt) kenar uzunlukları doğal olarak çok asimetrik olabilir — bu
# HATA değil, tam olarak homografinin düzeltmeye çalıştığı şey. Guard yalnız gerçekten
# saçma (dejenere) şekilleri eler.
SIDE_RATIO_MIN = 0.30
SIDE_RATIO_MAX = 3.3
EDGE_MARGIN_FRAC = 0.01  # köşe kadraj kenarına bu kadar (kısa kenarın %'si) yakınsa reddet
# Koyu tepsi maskesi: v eşiği düşük tutulur (80 değil 65) — aksi halde üstteki rafın
# etiket rayı/kovuğu (kısmen karanlık, eşik 80'i geçen bölgeler içeriyor) bu rafın
# tepsisiyle birleşip tek dev bulut oluşturuyor (bkz. train/39.jpeg ölçümü).
TRAY_MASK_V_MAX = 65
TRAY_MASK_S_MAX = 90


# ── Yardımcılar ────────────────────────────────────────────────────────────────
def _cx(det: dict[str, Any]) -> float:
    x1, _, x2, _ = det["bbox"]
    return (x1 + x2) / 2.0


def _cy(det: dict[str, Any]) -> float:
    _, y1, _, y2 = det["bbox"]
    return (y1 + y2) / 2.0


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """4 noktayı TL, TR, BR, BL sırasına koy (toplam/fark ile)."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(d))]
    bl = pts[int(np.argmax(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _corners_valid(corners: np.ndarray, w: int, h: int) -> tuple[bool, str]:
    """Dışbükey, makul en-boy oranlı, kadraja tam sığmış 4 köşe mi?"""
    if not cv2.isContourConvex(corners.astype(np.int32).reshape(-1, 1, 2)):
        return False, "dışbükey değil"

    area = cv2.contourArea(corners)
    frame_area = float(w * h)
    if area < MIN_TRAY_AREA_FRAC * frame_area:
        return False, "alan çok küçük"
    if area > MAX_TRAY_AREA_FRAC * frame_area:
        return False, "alan çok büyük"

    tl, tr, br, bl = corners
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    if not (SIDE_RATIO_MIN <= top / max(bottom, 1e-6) <= SIDE_RATIO_MAX):
        return False, "üst/alt kenar oranı aşırı"
    if not (SIDE_RATIO_MIN <= left / max(right, 1e-6) <= SIDE_RATIO_MAX):
        return False, "sol/sağ kenar oranı aşırı"

    margin = EDGE_MARGIN_FRAC * min(w, h)
    for x, y in corners:
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
            return False, "köşe kadraj kenarına yapışık"

    return True, ""


# ── (a) Tepsi konturu → 4 köşe ───────────────────────────────────────────────────
def detect_tray_corners(image: np.ndarray) -> np.ndarray | None:
    """Koyu tepsi gövdesinin en büyük bağlı bileşeninin 4 köşesini bul.

    Returns: (4, 2) float32 dizi [TL, TR, BR, BL] ya da guard'lardan biri
    düşerse None.
    """
    if cv2 is None:
        return None
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((v < TRAY_MASK_V_MAX) & (s < TRAY_MASK_S_MAX)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    approx = None
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        a = cv2.approxPolyDP(hull, frac * peri, True)
        if len(a) == 4:
            approx = a
            break
    if approx is None:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect)

    corners = _order_corners(np.asarray(approx, dtype=np.float32))
    ok, reason = _corners_valid(corners, w, h)
    if not ok:
        log.info("tepsi köşesi reddedildi: %s", reason)
        return None
    return corners


def shelf_number_from_tray_bar(image: np.ndarray, corners: np.ndarray) -> int | None:
    """Tepsinin KENDİ üst barındaki tek haneli basılı sayıyı OCR et (pytesseract varsa).

    Ray'den TAMAMEN bağımsız — ray bir üstteki rafa ait olduğu için raf numarası
    kaynağı olarak KULLANILAMAZ (bkz. modül docstring'i).
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None
    if cv2 is None:
        return None

    tl, tr, br, bl = corners
    h_img, w_img = image.shape[:2]
    x0 = int(max(0, min(tl[0], bl[0]) - 10))
    x1 = int(min(w_img, max(tr[0], br[0]) + 10))
    y_top = int(min(tl[1], tr[1]))
    tray_side = min(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    band_h = max(10, int(0.12 * tray_side))
    y0 = max(0, y_top - band_h)
    y1 = min(h_img, y_top + band_h)

    band = image[y0:y1, x0:x1]
    if band.size == 0:
        return None
    try:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        txt = pytesseract.image_to_string(
            gray, config="--psm 6 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None

    import re
    digits = re.findall(r"\d", txt)
    if not digits:
        return None
    vals, counts = np.unique(np.array(digits), return_counts=True)
    return int(vals[int(counts.argmax())])


# ── Homografi ────────────────────────────────────────────────────────────────────
def _homography_from_corners(
    corners: np.ndarray, column_count: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """4 köşeden H (orijinal→rektifiye) ve H_inv (rektifiye→orijinal) hesapla."""
    tl, tr, br, bl = corners
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    rect_w = max(top, bottom)
    rect_h = max(left, right)

    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners, dst)
    h_inv = cv2.getPerspectiveTransform(dst, corners)
    return h_mat, h_inv, rect_w, rect_h


def transform_point(x: float, y: float, h_mat: list[list[float]] | np.ndarray) -> tuple[float, float]:
    """Bir noktayı verilen 3x3 homografi ile dönüştür."""
    h_mat = np.asarray(h_mat, dtype=np.float64)
    pt = np.array([[[x, y]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, h_mat)[0, 0]
    return float(out[0]), float(out[1])


# ── Izgara birleştirici ────────────────────────────────────────────────────────
def _build_grid(corners: np.ndarray, column_count: int, *,
                grid_source: str, shelf_number: int | None = None,
                confident: bool = True) -> dict[str, Any]:
    h_mat, h_inv, rect_w, rect_h = _homography_from_corners(corners, column_count)
    pitch = rect_w / column_count
    boundaries = [i * pitch for i in range(column_count + 1)]
    centers = [pitch / 2.0 + i * pitch for i in range(column_count)]
    return {
        "grid_status": "ok",
        "column_count": int(column_count),
        "pitch": float(pitch),
        "rect_width": float(rect_w), "rect_height": float(rect_h),
        "centers": centers, "boundaries": boundaries,
        "corners": corners.tolist(),
        "H": h_mat.tolist(), "H_inv": h_inv.tolist(),
        "grid_source": grid_source,
        "shelf_number_from_bar": shelf_number,
        "confident": bool(confident),
    }


def _uncertain_grid(column_count: int, reason: str) -> dict[str, Any]:
    """Çapa güvenilir değil → atama yapılmaz. Kendinden emin yanlış yerine belirsizlik."""
    log.info("ızgara çapası: belirsiz (%s)", reason)
    return {
        "grid_status": "belirsiz", "reason": reason,
        "column_count": int(column_count),
        "pitch": None, "rect_width": None, "rect_height": None,
        "centers": [], "boundaries": [], "corners": None,
        "H": None, "H_inv": None,
        "grid_source": "belirsiz",
        "shelf_number_from_bar": None,
        "confident": False,
    }


def estimate_grid(
    image: np.ndarray | None,
    column_count: int,
    *,
    corners_override: np.ndarray | list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """KANONİK görüntüden tepsi köşelerini bul → homografi → ızgara.

    Bulunamazsa ``grid_status="belirsiz"``.

    Args:
        image: BGR görüntü (döndürülmemiş — homografi rotasyonu/perspektifi
            zaten kapsar). None → belirsiz.
        column_count: rafın fiziksel sütun sayısı.
        corners_override: operatörün elle işaretlediği 4 köşe
            ``[(x,y), (x,y), (x,y), (x,y)]`` — verilirse doğrudan kullanılır
            (sıra önemli değil, ``_order_corners`` düzeltir).
    """
    column_count = max(1, int(column_count))

    if corners_override is not None:
        corners = _order_corners(np.asarray(corners_override, dtype=np.float32))
        log.info("ızgara çapası: manuel köşeler")
        return _build_grid(corners, column_count, grid_source="manuel")

    if image is None:
        return _uncertain_grid(column_count, "görüntü yok")
    if cv2 is None:
        return _uncertain_grid(column_count, "cv2 yok")

    corners = detect_tray_corners(image)
    if corners is None:
        return _uncertain_grid(column_count, "tepsi konturu bulunamadı")

    shelf_no = shelf_number_from_tray_bar(image, corners)
    log.info("ızgara çapası: kontur (raf_no=%s)", shelf_no)
    return _build_grid(corners, column_count, grid_source="kontur", shelf_number=shelf_no)


def slot_index_for(x_center: float, y_center: float, grid: dict[str, Any]) -> int:
    """Tespit merkezini H ile rektifiye uzaya taşı, oradan kolon indeksini bul.

    ``slot_index = floor(x_rect / pitch)``, ``[0, count-1]``'e kırpılır. Tespit
    ASLA düşmez.
    """
    x_rect, _y_rect = transform_point(x_center, y_center, grid["H"])
    raw = int(np.floor(x_rect / grid["pitch"]))
    return int(min(max(raw, 0), grid["column_count"] - 1))
