"""Oryantasyon normalizasyonu — raf fotoğrafını KANONİK (90°'nin katı) yönelime oturtur.

Yönelim İÇERİKTEN TAHMİN EDİLMEZ. Önceki turlarda ``estimate_coarse_orientation`` tepsi
konturunun en-boy oranına bakıp portre/yatay kararı veriyordu; bu güvenilmez çıktı —
somut vakada (WhatsApp Image 2026-07-17 at 12.03.36.jpeg) gerçekten YATAY çekilmiş düz
bir fotoğrafta tepsi bloğunun kadraj içindeki şekli (kamera açısına bağlı, 0.83 en-boy)
"tepsi-dikey" sanılıp görsel 90° döndürüldü — yanlış. Tepsi bloğunun kadrajdaki şekli,
fotoğrafın gerçek yönelimini GÜVENİLİR yansıtmaz.

Yönelim kaynağı artık TEK: EXIF Orientation etiketi. Yoksa (bu projenin görsellerinde
hemen hiç yok — WhatsApp/telefon çoğu zaman EXIF'i siler) döndürme YAPILMAZ (0°). Şüphe
varsa döndürme — kendinden emin yanlış > açık default.

Kalan perspektif/eğim (tepsi çekimin doğası gereği hep bir miktar açılı olur) artık bu
modülün işi DEĞİL: ``shelf_grid`` homografisi tepsinin 4 köşesinden doğrudan rektifikasyon
yapar, ayrı bir "ince eğim" adımına gerek bırakmaz.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

_COARSE_ROT = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# PIL/EXIF Orientation tag değeri → kayıpsız 90° katı döndürme. 2/4/5/7 ayna
# (mirror) gerektirir — bu bir döndürme değildir, burada ele alınmaz (0 sayılır).
_EXIF_TO_COARSE = {1: 0, 3: 180, 6: 90, 8: 270}


def rotate_coarse(image: np.ndarray, angle: int) -> np.ndarray:
    """90°'nin katı kayıpsız döndürme (kanonik yönelime oturtma)."""
    code = _COARSE_ROT.get(int(angle) % 360)
    return image if code is None else cv2.rotate(image, code)


def exif_coarse_angle(exif_orientation: int | None) -> int:
    """EXIF Orientation etiketinden (1..8) kanonik döndürme açısı (0/90/180/270).

    Etiket yoksa ya da ayna (mirror, 2/4/5/7) ise 0 — içerikten tahmin YOK.
    """
    if exif_orientation is None:
        return 0
    return _EXIF_TO_COARSE.get(int(exif_orientation), 0)


def deskew(image: np.ndarray, exif_orientation: int | None = None) -> dict[str, Any]:
    """Görüntüyü EXIF'e göre kanonik yönelime oturt. İçerik tabanlı tahmin YOK.

    Args:
        image: BGR görüntü.
        exif_orientation: dosyanın EXIF Orientation etiketi (1..8) — çağıran
            (dashboard/script) dosya baytlarından/yolundan çıkarır. None → 0°.

    Returns:
        ``{"image", "angle", "coarse", "orient_source"}``. ``image`` EXIF varsa
        döndürülmüş kopya, yoksa orijinalin kendisi.
    """
    coarse = exif_coarse_angle(exif_orientation)
    osrc = "exif" if coarse else ("exif" if exif_orientation is not None else "varsayılan")
    rotated = rotate_coarse(image, coarse)
    return {
        "image": rotated,
        "angle": float(coarse),
        "coarse": coarse,
        "orient_source": osrc,
    }
