"""Tepsi köşe modeli — fotoğraftan tepsinin 4 köşesini TEPSİ SIRASIYLA bulur.

Neden: kanal ızgarası köşe yokken ürün dizilişinden kurulur (``product_anchored_grid``)
ve az ürünlü / arası boş raflarda kanalları kaydırır (bkz. o modülün "BİLİNEN
SINIRLAMA #2" notu). Köşeler bilinirse 7 kanal doğrudan homografiyle çizilir.

Model: yolo11n-pose, tek sınıf (tabla), 4 anahtar nokta. Eğitimi
``_test-verisi/kose-etiketleme/egit.py``. Nokta sırası tepsiye göredir:
0 sol üst, 1 sağ üst, 2 sağ alt, 3 sol alt ("üst" = QR'lı metal bar). Bu sıra
``shelf_grid.estimate_grid(corners_in_tray_order=True)`` ile olduğu gibi kullanılır.

Kural: emin olunmayan köşe VERİLMEZ (``confident=False`` + ``reason``). Çağıran bu
durumda eski yönteme döner. Yanlış köşe, köşesizlikten kötüdür: bütün kanalları
kendinden emin biçimde yanlış atar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULTS = {
    "min_box_conf": 0.50,     # tabla tespit güveni
    "min_kpt_conf": 0.50,     # her köşenin görünürlük güveni (kadraj dışı köşe düşük çıkar)
    "min_area_frac": 0.15,    # tepsi kadrajın en az bu kadarını kaplamalı
    "max_side_ratio": 2.0,    # karşılıklı kenar oranı (perspektif sınırı)
}


def load_corner_model(weights_path: str | Path) -> Any:
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def detect_tray_corners_model(
    image: np.ndarray, model: Any, params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """BGR görüntüde tepsi köşelerini bul.

    Returns:
        ``{confident, corners ([[x,y]×4] tepsi sırası, ya da None), reason,
           box_conf, kpt_conf}``
    """
    p = {**DEFAULTS, **(params or {})}
    out: dict[str, Any] = {"confident": False, "corners": None, "reason": "",
                           "box_conf": None, "kpt_conf": None}
    h, w = image.shape[:2]

    r = model.predict(image, verbose=False, conf=0.10)[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        out["reason"] = "tabla bulunamadı"
        return out

    i = int(r.boxes.conf.argmax())
    box_conf = float(r.boxes.conf[i])
    kp = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
    kc = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
          else np.ones(4, dtype=np.float32))
    out["box_conf"] = round(box_conf, 3)
    out["kpt_conf"] = [round(float(c), 3) for c in kc]
    out["corners"] = kp.tolist()  # emin olunmasa da teşhis için döner

    if box_conf < p["min_box_conf"]:
        out["reason"] = f"tabla güveni düşük ({box_conf:.2f})"
        return out
    if float(kc.min()) < p["min_kpt_conf"]:
        out["reason"] = f"köşe kadrajda değil / belirsiz (en düşük {kc.min():.2f})"
        return out
    if cv2 is not None and not cv2.isContourConvex(kp.astype(np.int32).reshape(-1, 1, 2)):
        out["reason"] = "köşeler dışbükey değil"
        return out
    area = abs(float(cv2.contourArea(kp))) if cv2 is not None else 0.0
    if area < p["min_area_frac"] * w * h:
        out["reason"] = f"tepsi alanı küçük (%{100 * area / (w * h):.0f})"
        return out
    tl, tr, br, bl = kp
    for a, b, ad in ((np.linalg.norm(tr - tl), np.linalg.norm(br - bl), "üst/alt"),
                     (np.linalg.norm(bl - tl), np.linalg.norm(br - tr), "sol/sağ")):
        ratio = max(a, b) / max(min(a, b), 1e-6)
        if ratio > p["max_side_ratio"]:
            out["reason"] = f"{ad} kenar oranı aşırı ({ratio:.2f})"
            return out

    out["confident"] = True
    return out


def refine_corners_with_products(
    corners: list[list[float]],
    detections: list[dict[str, Any]],
    column_count: int = 7,
    *,
    empty_class_name: str = "empty_slot",
    max_shift_channels: float = 0.6,
    coarse_steps: int = 9,
    min_products: int = 4,
    prior_weight: float = 0.5,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Köşe modelinin kalan kaymasını ürün konumlarıyla düzelt.

    Neden: kanal ayırıcıları dik duran plakalardır, ürünler tepsinin önünden taşar; açılı
    çekimde tepsinin 4 köşesinden çizilen düz ızgara fotoğrafın üstüne ve altına AYNI ANDA
    oturmaz (ölçüldü: raf 2'de üstte ayırıcılarla örtüşüp altta ~0.5 kanal sola kalıyor).
    Gerçek kanal sınırları ürün sütunlarının ARASINDAN geçer. Bu yüzden 4 ayrı kayma aranır:
    üst kenarın sol ve sağ ucu, alt kenarın sol ve sağ ucu (her biri en fazla
    ``max_shift_channels`` kanal). Üstteki ürünler üst kenarı, öndeki ürünler alt kenarı
    belirler; ürün olmayan bölgede ``prior_weight`` modelin köşesini korur.

    Kanal SAYISI köşelerden geldiği için eski latis yönteminin az ürünlü raflardaki çöküşü
    burada olmaz. Kayma sınırı bir kanalın altında tutulur (tam bir kanal kayması ürün
    sütunlarına aynı iyilikte oturur, onu ayırt edemeyiz).

    Ürün sayısı ``min_products`` altındaysa ayar yapılmaz.
    """
    info: dict[str, Any] = {"uygulandi": False}
    prods = [d for d in detections if d.get("category") != empty_class_name]
    if len(prods) < min_products or cv2 is None:
        info["neden"] = f"ürün az ({len(prods)})"
        return corners, info

    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    rect = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    h_mat = cv2.getPerspectiveTransform(c, rect)
    centers = np.float32([[((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2)
                          for d in prods]])
    tv = cv2.perspectiveTransform(centers, h_mat)[0]
    keep = (tv[:, 0] > -0.1) & (tv[:, 0] < 1.1) & (tv[:, 1] > -0.2) & (tv[:, 1] < 1.3)
    t, v = tv[keep, 0].astype(np.float64), tv[keep, 1].astype(np.float64)
    if len(t) < min_products:
        info["neden"] = "tepsi içinde yeterli ürün yok"
        return corners, info

    unit = 1.0 / column_count          # SABİT birim (aday genişliğe bölmek genişletmeyi ödüllendirir)
    kk = (np.arange(column_count) + 0.5) / column_count

    def cost_of(cand: np.ndarray) -> np.ndarray:
        """cand: (M, 4) = [üst-sol, üst-sağ, alt-sol, alt-sağ] kayma (tepsi genişliği birimi)."""
        a, b, cc, dd = (cand[:, i][:, None] for i in range(4))          # (M,1)
        left = a * (1 - v) + cc * v                                      # (M,N)
        right = 1 + b * (1 - v) + dd * v
        ch = left[..., None] + (right - left)[..., None] * kk            # (M,N,K)
        dist = np.abs(t[None, :, None] - ch).min(-1) / unit              # kanal birimi
        prior = prior_weight * ((cand / unit) ** 2).sum(1) / len(t)
        return (dist ** 2).mean(1) + prior

    lim = max_shift_channels * unit
    g = np.linspace(-lim, lim, coarse_steps)
    cand = np.stack(np.meshgrid(g, g, g, g, indexing="ij"), -1).reshape(-1, 4)
    best = cand[int(np.argmin(cost_of(cand)))]
    step = g[1] - g[0]
    for _ in range(3):                  # yerel inceltme: adımı yarıla, en iyinin çevresinde ara
        step /= 2
        local = np.linspace(-step * 2, step * 2, 5)
        cand = best + np.stack(np.meshgrid(local, local, local, local, indexing="ij"), -1).reshape(-1, 4)
        cand = np.clip(cand, -lim, lim)
        best = cand[int(np.argmin(cost_of(cand)))]

    a, b, cc, dd = (float(x) for x in best)
    tl, tr, br, bl = c
    new = np.array([tl + (tr - tl) * a, tl + (tr - tl) * (1 + b),
                    bl + (br - bl) * (1 + dd), bl + (br - bl) * cc], dtype=np.float32)
    info.update({"uygulandi": True,
                 "kayma_kanal": {"ust_sol": round(a / unit, 2), "ust_sag": round(b / unit, 2),
                                 "alt_sol": round(cc / unit, 2), "alt_sag": round(dd / unit, 2)},
                 "sinirda": bool(np.any(np.isclose(np.abs(best), lim)))})
    return new.tolist(), info


def corners_cover_products(
    corners: list[list[float]],
    detections: list[dict[str, Any]],
    *,
    empty_class_name: str = "empty_slot",
    margin: float = 0.12,
    max_outside_frac: float = 0.10,
) -> tuple[bool, str, float]:
    """Ürünlerin çoğu modelin bulduğu tepsinin içinde mi?

    Tam dolu rafta ürünler köşeleri kapatır; model köşeyi görmeden tahmin üretir ve köşe
    güveni bunu yakalamaz (ölçüldü: canlı raf 1'de ızgara rafın yalnız ortasını kapladı,
    en sol sütun tümüyle dışarıda kaldı, köşe güvenleri yine de > 0.5). Ürün merkezlerinin
    ``max_outside_frac``'tan fazlası tepsinin yanlarından ``margin`` (tepsi genişliği
    birimi) daha dışarıdaysa köşeler güvenilmez sayılır.
    """
    prods = [d for d in detections if d.get("category") != empty_class_name]
    if not prods or cv2 is None:
        return True, "", 0.0
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    h_mat = cv2.getPerspectiveTransform(c, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    centers = np.float32([[((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2)
                          for d in prods]])
    t = cv2.perspectiveTransform(centers, h_mat)[0][:, 0]
    frac = float(np.mean((t < -margin) | (t > 1 + margin)))
    if frac > max_outside_frac:
        return False, f"ürünlerin %{100 * frac:.0f}'i tepsinin dışında kaldı", frac
    return True, "", frac
