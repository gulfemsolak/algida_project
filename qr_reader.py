"""
QR Kod Tespit ve Raf Doğrulama Modülü
Vending machine raf fotoğraflarındaki QR kodları okur ve raf kimliğini doğrular.

Vending makinesi raflarındaki QR etiketleri fotoğrafta ÇOK küçük (~30-70px),
yansımalı/eğik yüzeyde ve WhatsApp JPEG sıkıştırmasıyla bozuk oluyor. Bu yüzden
tek bir decoder yetmiyor; katmanlı ve çok ölçekli bir yaklaşım kullanılıyor:

  1. zxing-cpp        — birincil, en sağlam decoder (saf wheel, sistem bağımlılığı yok)
  2. WeChat (opencv)  — ikincil, zxing'in kaçırdığı bazı kareleri yakalar
  3. cv2.QRCodeDetector — üçüncül yedek

Her decoder görüntüyü birkaç büyütme oranında dener (küçük QR'lar büyütülünce okunur).
Herhangi bir katman kuruluysa çalışır; hiçbiri yoksa kod çökmeden "devre dışı" der.

pyzbar KULLANILMIYOR: Homebrew zbar 0.23.x ile pyzbar 0.1.9 native ABI uyumsuzluğu
her decode()'da segfault veriyordu.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ── zxing-cpp (birincil) ───────────────────────────────────────────────────────
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception:
    zxingcpp = None
    ZXING_AVAILABLE = False

# ── WeChat dedektörü (ikincil) ─────────────────────────────────────────────────
try:
    _wechat = cv2.wechat_qrcode.WeChatQRCode()
    WECHAT_AVAILABLE = True
except Exception:
    _wechat = None
    WECHAT_AVAILABLE = False

# ── Standart cv2 dedektörü (üçüncül yedek) ─────────────────────────────────────
_cv2_detector = cv2.QRCodeDetector()

# Küçük/bulanık QR'lar için denenecek büyütme oranları.
_ZXING_SCALES = (1.0, 2.0, 3.0)
_WECHAT_SCALES = (1.0, 2.0)


def _to_bgr(image_input) -> np.ndarray | None:
    """Girişi BGR numpy array'e çevirir; okunamazsa None döner."""
    if isinstance(image_input, (str, Path)):
        return cv2.imread(str(image_input))
    if isinstance(image_input, np.ndarray):
        return image_input
    return None


def _rect_from_points(pts: np.ndarray) -> tuple[int, int, int, int]:
    """Köşe noktalarından (x, y, w, h) bbox üretir."""
    xs, ys = pts[:, 0], pts[:, 1]
    x, y = int(xs.min()), int(ys.min())
    return x, y, int(xs.max() - x), int(ys.max() - y)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _zxing_pass(gray: np.ndarray, scale: float) -> list[dict]:
    """zxing-cpp ile tek ölçekte QR ara. Düz griye ek olarak CLAHE (kontrast
    eşitleme) varyantını da dener — alüminyum/yansımalı yüzeydeki düşük kontrastlı
    küçük QR'lar CLAHE ile okunabilir hale gelebiliyor."""
    g = _resize(gray, scale)
    barcodes = []
    for variant in (g, _clahe.apply(g)):
        try:
            found = zxingcpp.read_barcodes(variant)
        except Exception:
            found = []
        if found:
            barcodes = found
            break
    results = []
    for bc in barcodes:
        if not bc.text:
            continue
        # zxing pozisyonu 4 köşe noktası verir
        try:
            p = bc.position
            corners = np.array([
                [p.top_left.x, p.top_left.y],
                [p.top_right.x, p.top_right.y],
                [p.bottom_right.x, p.bottom_right.y],
                [p.bottom_left.x, p.bottom_left.y],
            ], dtype=float)
            x, y, w, h = _rect_from_points(corners)
            bbox = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        except Exception:
            bbox = (0, 0, 0, 0)
        results.append({
            "data": bc.text,
            "bbox": bbox,
            "type": "QRCODE",
            "strategy": f"zxing_{scale:g}x",
        })
    return results


def _wechat_pass(bgr: np.ndarray, scale: float) -> list[dict]:
    """WeChat dedektörüyle tek ölçekte QR ara."""
    im = _resize(bgr, scale)
    try:
        texts, points = _wechat.detectAndDecode(im)
    except Exception:
        return []
    results = []
    for i, data in enumerate(texts):
        if not data:
            continue
        if points is not None and i < len(points):
            x, y, w, h = _rect_from_points(np.asarray(points[i], dtype=float))
            bbox = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        else:
            bbox = (0, 0, 0, 0)
        results.append({
            "data": data,
            "bbox": bbox,
            "type": "QRCODE",
            "strategy": f"wechat_{scale:g}x",
        })
    return results


def _cv2_pass(gray: np.ndarray) -> list[dict]:
    """cv2.QRCodeDetector ile son yedek. (Büyük görüntüde 3x upscale + multi-decode
    çok yavaş olduğundan burada upscale yok — zxing/WeChat zaten çok ölçekli deniyor.)"""
    variants = [gray, _clahe.apply(gray)]
    for g in variants:
        try:
            ok, infos, points, _ = _cv2_detector.detectAndDecodeMulti(g)
        except Exception:
            continue
        if not ok or points is None:
            continue
        results = []
        for data, quad in zip(infos, points):
            if not data:
                continue
            x, y, w, h = _rect_from_points(np.asarray(quad, dtype=float))
            results.append({"data": data, "bbox": (x, y, w, h),
                            "type": "QRCODE", "strategy": "cv2"})
        if results:
            return results
    return []


def detect_qr_codes(image_input) -> list[dict]:
    """
    Fotoğraftaki tüm QR kodları tespit eder ve decode eder.

    Args:
        image_input: str/Path (dosya yolu) VEYA numpy array (OpenCV image)

    Returns:
        Liste of dict: [{"data": str, "bbox": (x,y,w,h), "type": str}]
        Bulunamazsa boş liste [].
    """
    img = _to_bgr(image_input)
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) zxing-cpp — çok ölçekli
    if ZXING_AVAILABLE:
        for scale in _ZXING_SCALES:
            hits = _zxing_pass(gray, scale)
            if hits:
                return hits

    # 2) WeChat — çok ölçekli
    if WECHAT_AVAILABLE:
        for scale in _WECHAT_SCALES:
            hits = _wechat_pass(img, scale)
            if hits:
                return hits

    # 3) cv2 — son yedek
    return _cv2_pass(gray)


def verify_shelf_qr(image_input, expected_shelf: str | None = None) -> dict:
    """
    QR kodu okur ve raf kimligiyle karsilastirir.

    Args:
        image_input: dosya yolu veya numpy array
        expected_shelf: beklenen raf degeri (or: "31", "Raf1").
                        None ise sadece tespit yapar, dogrulama yapmaz.

    Returns:
        dict: {
            "qr_found": bool,
            "qr_count": int,
            "qr_values": list[str],
            "qr_bboxes": list[tuple],
            "verified": True | False | None,
            "message": str  (Turkce kullanici mesaji)
        }
    """
    if not (ZXING_AVAILABLE or WECHAT_AVAILABLE):
        return {
            "qr_found": False, "qr_count": 0, "qr_values": [], "qr_bboxes": [],
            "verified": None,
            "message": "QR okuma kütüphaneleri yüklü değil — QR okuma devre dışı",
        }

    detections = detect_qr_codes(image_input)

    if not detections:
        return {
            "qr_found": False, "qr_count": 0, "qr_values": [], "qr_bboxes": [],
            "verified": None,
            "message": "QR kod tespit edilemedi",
        }

    qr_values = [d["data"] for d in detections]
    qr_bboxes = [d["bbox"] for d in detections]

    verified = None
    if expected_shelf is not None:
        verified = any(
            expected_shelf.lower() in v.lower() or v.lower() in expected_shelf.lower()
            for v in qr_values
        )

    values_str = ", ".join(qr_values)
    if verified is True:
        message = f"QR doğrulandı: {values_str}"
    elif verified is False:
        message = f"QR uyumsuz! Beklenen: {expected_shelf}, Okunan: {values_str}"
    else:
        message = f"QR tespit edildi: {values_str}"

    return {
        "qr_found": True,
        "qr_count": len(detections),
        "qr_values": qr_values,
        "qr_bboxes": qr_bboxes,
        "verified": verified,
        "message": message,
    }
