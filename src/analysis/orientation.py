"""Oryantasyon normalizasyonu (Katman 1) — raf fotoğrafını slotlar DİKEY olacak
şekilde döndürür.

Fotoğraflar elle çekiliyor: bazen 90° yan, bazen eğik. Slot atama (Katman 2)
slotların dikey olduğunu varsayar. Tavuk-yumurta problemi (yönü bilmek için
tespit, tespit için yön) ürünlerin rotasyona dayanıklılığıyla çözülür:

    1. Orijinal görüntüde ürün tespiti (empty_slot hariç) — ÇAĞIRAN yapar (1. geçiş).
    2. Ürün merkezlerinin EN YAKIN KOMŞU vektörlerinin açı histogramı → mod = slot
       ekseni. Aynı slottaki ürünler bitişik (kısa NN vektörü); slotlar arası mesafe
       daha büyük, bu yüzden en yakın komşu genelde AYNI slottan → vektör slot ekseni
       boyuncadır. Kısa vektörler ağırlıklandırılır (slotlar-arası uzun vektörler
       bastırılır).
    3. Slot ekseni dikey olacak şekilde döndürme açısı (cv2 konvansiyonu) döndürülür.
       Bu hem 90° yan çekimi hem eğik çekimi TEK mekanizmayla çözer — ayrı bir
       COLUMN_SHEAR kalibrasyonuna gerek yoktur.
    4. Normalize görüntüde tam tespit — ÇAĞIRAN yapar (2. geçiş).

Ürün sayısı çok azsa (< 3) açı güvenilmez → ``confidence = 0.0`` döner; çağıran
döndürme yapmaz, sonucu düşük güvenle işaretler.

İzole ve saf: ``estimate_orientation`` yalnızca tespit listesi alır (görüntü değil),
``rotate_image`` yalnızca görüntü + açı alır. İkisi de birim testte doğrudan çağrılır.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Slot ekseni bu açıya (derece, image koordinatı) getirilmeye çalışılır: dikey.
_VERTICAL_DEG = 90.0


def _product_centers(detections: list[dict[str, Any]], empty_class_name: str) -> np.ndarray:
    """empty_slot HARİÇ ürün kutularının merkezleri (N, 2). empty_slot yönü taşımaz."""
    pts = [
        [(d["bbox"][0] + d["bbox"][2]) / 2.0, (d["bbox"][1] + d["bbox"][3]) / 2.0]
        for d in detections
        if d.get("category") != empty_class_name
    ]
    return np.asarray(pts, dtype=float)


def _nearest_neighbor_vectors(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Her nokta için en yakın komşuya vektör (dx, dy) ve mesafe döndür.

    Returns:
        vecs (N, 2), dists (N,). Tek nokta varsa boş diziler.
    """
    n = len(centers)
    if n < 2:
        return np.empty((0, 2)), np.empty((0,))

    # İkili mesafe matrisi (N küçük — birkaç yüz kutu — O(N^2) yeterli).
    diff = centers[:, None, :] - centers[None, :, :]        # (N, N, 2)
    d2 = (diff ** 2).sum(axis=2)                            # (N, N)
    np.fill_diagonal(d2, np.inf)                            # kendini komşu sayma
    nn = np.argmin(d2, axis=1)                              # (N,)

    vecs = centers[nn] - centers                            # (N, 2) her nokta→komşusu
    dists = np.sqrt((vecs ** 2).sum(axis=1))
    return vecs, dists


def estimate_orientation(
    detections: list[dict[str, Any]],
    empty_class_name: str = "empty_slot",
    min_products: int = 3,
    bin_deg: float = 5.0,
) -> tuple[float, float]:
    """Ürün diziliminden slot eksenini dik yapan döndürme açısını tahmin et.

    Args:
        detections: 1. geçiş tespitleri (``bbox`` + ``category`` içeren dict'ler).
        empty_class_name: yön hesabından dışlanacak sınıf adı.
        min_products: bunun altında ürün varsa açı güvenilmez → (0.0, 0.0).
        bin_deg: açı histogramı bin genişliği (derece).

    Returns:
        ``(angle_deg, confidence)``:
          - ``angle_deg``: ``rotate_image(image, angle_deg)``'e verilince slot
            ekseni dikleşir (cv2 konvansiyonu, (-90, 90] aralığında).
          - ``confidence``: 0..1; histogram tepe yoğunluğu × yeterli-örnek. 0.0 =
            güvenilmez (çağıran döndürmemeli).
    """
    centers = _product_centers(detections, empty_class_name)
    if len(centers) < min_products:
        return 0.0, 0.0

    vecs, dists = _nearest_neighbor_vectors(centers)
    if len(vecs) == 0:
        return 0.0, 0.0

    # Açılar mod 180 (yön belirsiz: yukarı/aşağı aynı eksen). Kısa NN vektörleri
    # (aynı slot) ağırlıklı; uzun vektörler (slotlar arası) bastırılır.
    angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) % 180.0
    weights = 1.0 / (dists + 1e-6)

    # Ağırlıklı histogram → tepe bin (mod).
    n_bins = int(np.ceil(180.0 / bin_deg))
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    hist, _ = np.histogram(angles, bins=edges, weights=weights)
    total = float(hist.sum())
    if total <= 0:
        return 0.0, 0.0
    peak = int(np.argmax(hist))

    # Tepe bin çevresinde (± bir bin) ağırlıklı DAİRESEL ortalama ile bin-altı
    # hassasiyet. mod-180 sarmasını yönetmek için açılar 2× ile tam çembere taşınır.
    lo, hi = edges[max(0, peak - 1)], edges[min(n_bins, peak + 2)]
    sel = (angles >= lo) & (angles < hi)
    if sel.sum() == 0:
        axis_angle = (edges[peak] + edges[peak + 1]) / 2.0
    else:
        a2 = np.radians(angles[sel] * 2.0)
        w = weights[sel]
        mean2 = np.arctan2((w * np.sin(a2)).sum(), (w * np.cos(a2)).sum())
        axis_angle = (np.degrees(mean2) / 2.0) % 180.0

    # Yoğunluk (tepe ± komşu binlerin toplam ağırlık payı) × örnek yeterliliği.
    peak_mass = float(hist[max(0, peak - 1):peak + 2].sum()) / total
    sample_factor = min(1.0, len(centers) / (2.0 * min_products))
    confidence = round(peak_mass * sample_factor, 4)

    # Slot eksenini (axis_angle) dikeye (90°) taşıyan cv2 döndürme açısı.
    # cv2 açı `a` ile döndürme bir vektörün açısını `a` kadar AZALTIR
    # (x'=cos·x+sin·y, y'=-sin·x+cos·y ⇒ yeni açı = eski − a). Dolayısıyla
    # yeni_eksen = axis_angle − a = 90 ⇒ a = axis_angle − 90.
    angle = axis_angle - _VERTICAL_DEG
    # (-90, 90] aralığına indir (en küçük dönüş).
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    return round(float(angle), 2), confidence


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """Görseli ``angle`` derece döndür (cv2 konvansiyonu), köşeleri kırpmadan.

    Tuval, döndürülmüş içeriği tam kapsayacak şekilde büyütülür (kenarlar
    ``border_value`` ile doldurulur). ``estimate_orientation`` çıktısı buraya
    verilince slot ekseni dikleşir.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += new_w / 2.0 - center[0]
    m[1, 2] += new_h / 2.0 - center[1]

    return cv2.warpAffine(
        image, m, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
