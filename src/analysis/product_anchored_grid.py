"""Latis-tabanlı lane tespiti — kimlikten BAĞIMSIZ eşit-pitch fiziksel kanallar.

KAVRAM (bu tasarımın özü): Lane KİMLİKTEN kurulmaz. Lane = tepsinin eşit aralıklı
fiziksel spiral kanalı. Önce tek bir açı θ ve buna dik sıralama ekseni u kurulur
(kimlikten bağımsız, yön-agnostik). Sonra PITCH (kanal genişliği) iki bağımsız
kaynaktan tahmin edilir; u-ekseninde eşit aralıklı bir LATİS üretilir; latisin FAZI,
tespit merkezlerinin kanal merkezlerine toplam uzaklığını minimize edecek şekilde
hizalanır (sınırlar ürünlerin ARASINA düşer, ÜSTÜNDEN değil). Her tespit en yakın
kanala düşer; kanalın etiketi = içindeki ÇOĞUNLUK tür (bu ``slot_assigner``'da yapılır).

Bu üç eski hatayı KÖKTEN çözer:
  P1 BLOK-BİRLEŞME: aynı ürün iki kanala yayılırsa latis onu doğal olarak iki kanala
     böler (10 Badem → ~2 slot). Ayrı "geniş zincir bölme" kod yolu GEREKMEZ — latis
     kimlikten bağımsız olduğu için blok kaç kanala denk geliyorsa o kadar slot olur.
  P2 KAYMA/KARIŞIM: kanal sınırları latis fazıyla ürünlerin ARASINA hizalandığından
     bir sınır ASLA bir ürün zincirinin üstünden geçmez; komşu farklı ürünler doğru
     kanala düşer, çoğunluk kanalı belirler → karışım minimuma iner.
  P3 BOŞ ALGISI: latis sabit ve tepsi kenarına kadar uzadığı için ürün düşmeyen ara/uç
     kanallar doğal olarak BOŞ slot çıkar (ekstra tahmin gerekmez).

Tek bir küresel θ kullanılır (perspektif "yelpaze" YOK): kanallar v'ye paralel, yani
(t, s) uzayında hepsi ``b = 0`` — bu yüzden sınırlar YAPISAL olarak kesişemez.

Üretilen grid ``assign_slots``/``draw_slots`` için gereken ortak alanları
(``column_count``, ``pitch``, ``centers``, ``boundaries``, ``grid_status``,
``grid_source``, ``axis_origin/v/u``, ``boundary_lines``) korur; atama/çizim
``chain_slot_index_for`` (sınır doğrusu testi) üzerinden yapılır.

BİLİNEN SINIRLAMA (ölçülmüş, 2026-07): en dış tespit edilen ürünün ÖTESİNDEKİ tamamen
boş kanallar (tepsinin bir ucu/bölümü hiç doldurulmamışsa) otomatik olarak tespit
EDİLEMİYOR — latis yalnızca ürün varlığından öğrenebildiği kadarıyla kurulur, orada
tanım gereği ürün (veri) yoktur. "Tepsi fiziksel sınırını bul" ailesinden BEŞ farklı
yaklaşım ölçülüp reddedildi:
  1. Ray (etiket rayı) çapası — parallax nedeniyle yanlış rafa ait (bkz. ``shelf_grid.py``).
  2. Spiral-CV otokorelasyonu dolu bölgede — gerçek fotonun %87'sinde yanlış (bkz. eski
     ``shelf_grid.py`` tasarım notu, kaldırıldı).
  3. Tepsi köşe-konturu (``detect_tray_corners``) — bu foto tipinde (üstten açılı spiral
     tepsi) sık sık reddediliyor ("köşe kadraj kenarına yapışık" vb.).
  4. Görüntü kadrajının kendisi (son çare) — tepsi etrafında duvar/çerçeve boşluğu olan
     fotolarda kadraj gerçek tepsi genişliğinden çok daha büyük çıkıyor; tam-dolu
     tepsilerde bile hayalet boş uç ürettiği ölçüldü (17.jpeg: 7→8 slot, `0|1|5|4|4|5|7|9`
     — baştaki 0 uydurma). Net regresyon, uygulanmadı.
  5. Boş-bant Canny+otokorelasyon sondası (en dış üründen öteye, dolu bölgeye hiç
     girmeden) — teşhis AŞAMASINDA (koda bağlanmadan) reddedildi: tam-dolu tepsilerin
     (108.jpeg, 5.jpeg) otokorelasyon netliği (0.16, 0.30), gerçek boş-bantlı fotoların
     çoğundan (çoğu 0.00–0.15) YÜKSEK çıktı — iki dağılım tamamen iç içe, ayrışmıyor.
Kök neden hepsinde aynı: boş kanalda veri YOK, bu yüzden veri-türevli hiçbir sinyal
(ürün, ray, kontur, kadraj, CV periyodu) güvenilir ayırt edemiyor. Çözüm insan-döngüde:
bkz. ``expected_column_count`` (QR/operatör K girişi, Artış A) — operatör toplam kanal
sayısını düzeltirse K-kısıtlı comb (``estimate_pitch_comb_fixed_k``) boş kanalları da
doğru sayıda üretir. Otomatik/CV tabanlı bir çözüm ÖLÇÜLMEDEN denenmemeli.

BİLİNEN SINIRLAMA #2 — "sessizce yanlış" bölge (ölçülmüş, 2026-07, HENÜZ DÜZELTİLMEDİ):
``N < COMB_MIN_PRODUCTS`` (< 6 ürün) altında ``grid_status="belirsiz"`` dönülür (bkz.
``_lane_grid`` — mimari gerekçe: comb o noktada ÇALIŞAMAZ, A/B pitch'e sessizce düşerdi).
Ancak 6–25 ürün aralığında grid_status="ok" dönüyor ve latis GERÇEKTE hâlâ güvenilmez
(147+94 fotonun ``|n_slots-7|`` sapması):

    N ürün    foto   ort.|slot-7|   %tam-7
    6–8         1        4.00         0%
    8–10        4        3.25         0%
    10–15       6        2.50        17%
    15–25      19        1.16        37%
    ≥25       202        0.25        80%

Bu turun kapsamı DIŞINDA bırakıldı (düzeltme, ≥6 ürünlü ~29 fotoyu etkileyip regresyon
riski yaratırdı — bkz. ``low_detection`` bayrağının YALNIZ <6'da tetiklenmesi).

ÖLÇÜLDÜ (2026-07-22): bu 30 foto (6≤N<25) tek tek incelendi — YENİ bir hata sınıfı
DEĞİL, Tür-2'nin (en dış ürünün ötesindeki tespit edilemeyen boş/seyrek kanal) 6.
tezahürü:
  (i) Comb SKORU ayırt edici DEĞİL — yanlış çıkan fotolarda (n_slots≠7) skor genelde
      DOĞRU çıkanlardan YÜKSEK (ör. skor=0.855→3 slot yanlış; skor=0.375→7 slot doğru).
      "Skorla güvenilmezlik tespiti" bu yüzden ELENDİ — eşik koyacak ayırt edici sinyal yok.
  (ii) Comb bu aralıkta neredeyse hiç reddetmiyor (29/30 comb_used=True) — ``≥25``
       grubuyla aynı oran (%99); "comb sık reddediyor" da değil.
  (iii) Hata yönü TEK TARAFLI: 30/30 fotoda yalnız EKSİK-bölünme, hiç aşırı-bölünme yok.
  (iv) K-KISITLI COMB DA KURTARMIYOR (kritik test): 8 örnekte ``expected_column_count=7``
       verildi, hepsinde ``fixed_k_used=False`` — ``estimate_pitch_comb_fixed_k``'nın
       ``p0=(sağ_kenar−sol_kenar)/K`` hesabı AYNI güvenilmez kenarlara (spiral/fallback;
       ``kontur`` bu foto tipinde hiç başarılı olmuyor) dayanıyor. Darboğaz K DEĞİL,
       SPAN/KENAR — az üründe seyrek kolonlar olunca kenar tespiti yoğun alt-bölgeye
       sıkışıp gerçek tepsi genişliğini kaçırıyor.
Sonuç: bu bucket otomatik olarak ne TESPİT edilebiliyor (ayırt edici sinyal yok — (i)),
ne de mevcut operatör-K yoluyla DÜZELTİLEBİLİYOR (K yetmiyor — (iv)). Reddedilen
yaklaşımların listesine ekle: ray-çapa, spiral-CV dolu bölgede, köşe-kontur, kadraj-kenarı,
boş-bant otokorelasyon, comb-skoru filtresi, K-girişi (span olmadan) — YEDİ ölçülmüş
başarısızlık. Çözüm yönü muhtemelen operatörden K'NIN YANINDA SPAN'ı da almak (mevcut
``manual_corners`` mekanizması bunu zaten sağlıyor — yeni bir UI icat ETME, onu her zaman
erişilebilir bir yola genişletmek yeterli olabilir). Bu bir TASARIM kararı; ayrı onayla
planlanmalı, rastgele "iyileştirme" denemeden önce bu tabloyu güncel veriyle doğrula.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from src.analysis.shelf_grid import (
    _uncertain_grid,
    detect_tray_corners,
    estimate_grid as estimate_homography_grid,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Açı kestirimi için zincirler (SADECE θ; lane sınırı zincirden KURULMAZ) ────────
MIN_PRODUCTS_FOR_CHAINS = 2       # bunun altında geometri tanımsız → belirsiz
LINK_DIST_FACTOR = 1.8            # aynı-sınıf bağlama mesafesi = medyan ürün boyutu * bu
SINGLETON_ABSORB_FACTOR = 1.8     # tekil (muhtemelen yanlış etiketli) zincir emme mesafesi
MIN_CHAIN_LEN_FOR_PCA = 3         # bu uzunluktaki zincirler eksen (θ) oyuna katılır
DEFAULT_AXIS_ANGLE = math.pi / 2  # hiçbir zincir/ürün yoksa son çare: dikey varsay

# ── Pitch (kanal genişliği) tahmini ───────────────────────────────────────────────
PITCH_CONSISTENCY_TOL = 0.30      # |pitch_B - pitch_A| bunu * pitch_A aşarsa tutarsız → min(A,B)
TRANSITION_MIN_FACTOR = 0.5       # sınıf-geçiş mesafesi bu * pitch_A altındaysa aynı-kanal gürültüsü
TRANSITION_MAX_FACTOR = 1.6       # bu * pitch_A üstündeyse çok-kanal atlaması → tek-adım değil
MIN_TRANSITIONS = 2               # pitch_B için gereken asgari kimlik-geçişi (altında güvenilmez)

# ── Latis / kenar sağlık kapıları ─────────────────────────────────────────────────
MAX_CHANNELS = 25                 # bundan çok kanal → pitch tahmini çökmüş, belirsize düş
EDGE_MAX_REACH_FACTOR = 1.0       # kenar en dıştaki üründen en fazla bu * pitch dışarı (aşırıysa fallback)

# ── Tepsi kenarı tespiti (boş desen / spiral sezgiseli) ──────────────────────────
EMPTY_PATTERN_MAX_STEPS = 3       # kenar aramasının son üründen dışa gideceği azami adım (pitch birimi)
EMPTY_PATTERN_DROP_RATIO = 0.5    # kenar dışı yoğunluk temel yoğunluğun bu oranı altına düşerse "kenar"
EMPTY_PATTERN_MIN_BASELINE = 0.02 # bu değerden az kenar-yoğunluğu varsa sezgisel güvenilmez (sinyal yok)

# ── Karışık slot / hizalama şüphesi (assign_slots ile aynı eşik, telemetri) ──────
SECONDARY_CLASS_FRAC = 0.25


def _cx(det: dict[str, Any]) -> float:
    x1, _, x2, _ = det["bbox"]
    return (x1 + x2) / 2.0


def _cy(det: dict[str, Any]) -> float:
    _, y1, _, y2 = det["bbox"]
    return (y1 + y2) / 2.0


def _bbox_size(det: dict[str, Any]) -> float:
    x1, y1, x2, y2 = det["bbox"]
    return max(x2 - x1, y2 - y1)


def _dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(_cx(a) - _cx(b), _cy(a) - _cy(b))


def _u_extent(det: dict[str, Any], u: tuple[float, float]) -> float:
    """Bir kutunun u-ekseni (dik) yönündeki izdüşüm genişliği — eksen-hizalı bir
    dikdörtgenin birim ``u`` yönüne izdüşümü tam olarak ``|u_x|*w + |u_y|*h``'dir."""
    x1, y1, x2, y2 = det["bbox"]
    return abs(u[0]) * (x2 - x1) + abs(u[1]) * (y2 - y1)


def _project(point: tuple[float, float], origin: tuple[float, float],
             axis: tuple[float, float]) -> float:
    px, py = point
    return (px - origin[0]) * axis[0] + (py - origin[1]) * axis[1]


# ── 1) Zincir kurma — YALNIZCA açı (θ) kestirimi için ─────────────────────────────
def build_chains(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aynı sınıftan, mekânsal olarak ardışık tespitleri zincirle (union-find).

    Bu zincirler ARTIK lane sınırı KURMAZ; yalnızca baskın eksen θ'yı kestirmek için
    kullanılır (aynı-ürün dikey komşu çiftlerin eğimi rafın diklik yönünü verir).
    """
    if not products:
        return []

    sizes = [_bbox_size(d) for d in products]
    median_size = float(np.median(sizes)) or 1.0
    max_dist = LINK_DIST_FACTOR * median_size

    n = len(products)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if products[i]["category"] != products[j]["category"]:
                continue
            if _dist(products[i], products[j]) <= max_dist:
                union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(products[i])

    chains = [
        {"members": members, "category": Counter(d["category"] for d in members).most_common(1)[0][0]}
        for members in groups.values()
    ]
    return _absorb_singletons(chains, max_dist * SINGLETON_ABSORB_FACTOR / LINK_DIST_FACTOR)


def _absorb_singletons(chains: list[dict[str, Any]], max_dist: float) -> list[dict[str, Any]]:
    """Tekil zincirleri (muhtemel yanlış etiket) en yakın çok-elemanlı komşuya em —
    böylece θ kestirimi tek gürültülü noktadan etkilenmez."""
    singles = [c for c in chains if len(c["members"]) == 1]
    others = [c for c in chains if len(c["members"]) >= 2]
    if not singles or not others:
        return chains

    absorbed = set()
    for s in singles:
        p = s["members"][0]
        best, best_d = None, math.inf
        for o in others:
            for m in o["members"]:
                d = _dist(p, m)
                if d < best_d:
                    best_d, best = d, o
        if best is not None and best_d <= max_dist:
            best["members"].append(p)
            absorbed.add(id(s))

    result = [c for c in chains if id(c) not in absorbed]
    for c in result:
        c["category"] = Counter(d["category"] for d in c["members"]).most_common(1)[0][0]
    return result


# ── 2) Baskın yön θ (yön-bağımsızlık burada) ──────────────────────────────────────
def _pca_angle(points: list[tuple[float, float]]) -> float | None:
    """Nokta bulutunun ana ekseninin açısı, (-π/2, π/2] aralığına kanonikleştirilmiş."""
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=float)
    centered = pts - pts.mean(axis=0)
    if np.allclose(centered, 0):
        return None
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.atan2(principal[1], principal[0])
    if angle <= -math.pi / 2 or angle > math.pi / 2:
        angle = angle - math.pi if angle > 0 else angle + math.pi
    return angle


def _circular_median_mod_pi(angles: list[float]) -> float:
    """π-periyodik (yönsüz doğru) açılar için gerçek medyan: toplam dairesel mesafeyi
    minimize eden GİRDİ değerini seçer (aykırı bir-iki zincire dayanıklı)."""
    def circ_dist(a: float, b: float) -> float:
        d = abs(a - b) % math.pi
        return min(d, math.pi - d)

    best, best_cost = angles[0], math.inf
    for a in angles:
        cost = sum(circ_dist(a, b) for b in angles)
        if cost < best_cost:
            best_cost, best = cost, a
    return best


def dominant_axis(chains: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Zincirlerin PCA ana eksenlerinin medyanından raf-dikey yönü v'yi (ve v'ye dik
    sıralama ekseni u'yu) çıkar. Fotoğraf 90° dönükse v resim uzayında yatay çıkar.

    Not (ölçülmüş): "aynı-sınıf en-yakın-komşu vektörü" alternatifi Magnum yelpaze
    açısını düzeltiyordu (77°→85°) ama TEK komşu vektörü gürültülü olduğundan 94-foto
    genelinde karışık-slotu artırdı (21→34). PCA-medyanı aggregate'te daha kararlı
    olduğu için korundu; θ'nın gerçek çözümü ayrı ele alınmalı.

    Returns:
        ``(v, u, angle_deg)`` — v, u birim vektör (x, y); angle_deg telemetri için.
    """
    angles = []
    for c in chains:
        if len(c["members"]) >= MIN_CHAIN_LEN_FOR_PCA:
            a = _pca_angle([(_cx(d), _cy(d)) for d in c["members"]])
            if a is not None:
                angles.append(a)

    if angles:
        v_angle = _circular_median_mod_pi(angles)
    else:
        all_pts = [(_cx(d), _cy(d)) for c in chains for d in c["members"]]
        a = _pca_angle(all_pts)
        v_angle = a if a is not None else DEFAULT_AXIS_ANGLE

    v = (math.cos(v_angle), math.sin(v_angle))
    u = _canonicalize_u((-v[1], v[0]))
    return v, u, round(math.degrees(v_angle), 2)


def _canonicalize_u(u: tuple[float, float]) -> tuple[float, float]:
    """Numaralandırma "soldan sağa" (u+ yönünde) olmalı: baskın bileşeni pozitif yönde
    kanonikleştir (baskın-OLMAYAN küçük bileşenin işaretine bakmak ekseni ters çevirir).

    ÖLÇÜLMÜŞ BUG (GÖREV 1, 2026-07): bu kural yalnızca ``dominant_axis`` içinde
    uygulanıyordu — ``_search_axis_angle`` θ'yı değiştirdiğinde kendi ``u``'sunu ham
    rotasyon formülünden (``u=(-v_y,v_x)``) TÜRETİYOR, bu kanonikleştirmeyi TEKRAR
    UYGULAMIYORDU. Sonuç: 147 fotonun 138'inde (zincir yolu) 33'ü (%24) s-artan ile
    x-artan arasında TERS korelasyon veriyordu (slot 17..11 gibi sağdan sola
    numaralanıyordu) — 22/33'ü ``theta_search_used=True`` (çoğunluk kök), 11/33'ü
    ``theta_search_used=False`` (``dominant_axis``'ın kendi kuralı da "vertical" dalda
    baskın-OLMAYAN bileşenin işaretini serbest bırakıyordu — ikinci, küçük bir sızıntı).
    Düzeltme: bu fonksiyon TEK yerden çağrılır — hem ``dominant_axis`` hem
    ``_search_axis_angle`` sonrası (bkz. ``_lane_grid``) — final u HER ZAMAN aynı
    kurala tabi olur, kaynağı ne olursa olsun."""
    if abs(u[0]) >= abs(u[1]):
        if u[0] < 0:
            return (-u[0], -u[1])
    else:
        if u[1] < 0:
            return (-u[0], -u[1])
    return u


# ── 3) Pitch — iki bağımsız kaynak, kimlik-geçişi (B) tercihli ────────────────────
def estimate_pitch(
    products: list[dict[str, Any]], u: tuple[float, float], origin: tuple[float, float],
) -> tuple[float, float, float | None, bool]:
    """Kanal genişliği (pitch) tahmini — iki bağımsız kaynak.

    Kaynak A (ürün genişliği): tüm kutuların u-yönü genişliğinin medyanı.
    Kaynak B (kimlik geçişi): u boyunca sıralı tespitlerde SINIF DEĞİŞEN komşu çiftlerin
    arasındaki mesafeler tek-kanal geçişleridir (≈ 1 pitch). Aynı-kanal gürültüsü
    (< A'nın yarısı) ve çok-kanal atlaması (> 1.6 A) elenir; kalanların medyanı pitch_B.

    SEÇİM: pitch_A ürün kutularının u-enidir; paketler ÜST ÜSTE BİNDİĞİNDE eni gerçek
    kanal aralığını ŞİŞİRİR (105-foto: pitch_A>pitch_B 95/103). Ancak pitch_B de yansız
    DEĞİL: lane-içi s-yayılımı sınıf-geçişi mesafesini KÜÇÜLTÜR, yani pitch_B çoğu temiz
    tepside gerçek pitch'in ALTINA düşer (ör. 108.jpeg 7 temiz lane: pA=156 doğru, pB=90).
    Ölçüm (349→698 ihlal, 36 foto kötüleşti) ``min(A,B)``'nin temiz tepsileri AŞIRI-BÖLDÜĞÜNÜ
    ve iki kaynağı ayırt edecek güvenilir sinyal OLMADIĞINI gösterdi. Bu yüzden GÜVENLİ
    varsayılan korunur:
      - B güvenilir + tutarlı (fark ≤ %30) → A ve B'nin ortalaması,
      - B güvenilir ama tutarsız → pitch_A (kararlı taban),
      - B GÜVENİLMEZ (çok az kimlik-geçişi) → pitch_A, bayrak.
    (pitch'in gerçek çözümü A/B seçimi değil; s-tarağı periyodikliğinden doğrudan
    kestirim gerektirir — bu daha büyük bir değişiklik, ayrı ele alınmalı.)

    Returns: ``(pitch, pitch_A, pitch_B, pitch_B_unreliable)``.
    """
    widths = [_u_extent(d, u) for d in products]
    pitch_A = float(np.median(widths)) if widths else 1.0
    if pitch_A <= 0:
        pitch_A = 1.0

    seq = sorted(
        ((_project((_cx(d), _cy(d)), origin, u), d["category"]) for d in products),
        key=lambda x: x[0],
    )
    trans = []
    for (s1, c1), (s2, c2) in zip(seq, seq[1:]):
        if c1 == c2:
            continue
        g = s2 - s1
        if TRANSITION_MIN_FACTOR * pitch_A <= g <= TRANSITION_MAX_FACTOR * pitch_A:
            trans.append(g)
    pitch_B = float(np.median(trans)) if trans else None

    unreliable = pitch_B is None or len(trans) < MIN_TRANSITIONS
    if unreliable:
        return pitch_A, pitch_A, pitch_B, True
    if abs(pitch_B - pitch_A) <= PITCH_CONSISTENCY_TOL * pitch_A:
        pitch = 0.5 * (pitch_A + pitch_B)
    else:
        pitch = pitch_A
    return pitch, pitch_A, pitch_B, False


# ── 3b) Pitch — s-tarağı periyodikliğinden doğrudan kestirim (comb) ──────────────
COMB_MIN_PRODUCTS = 6             # bunun altında periyot tanımsız → None (fallback)
COMB_SEARCH_STEPS = 60            # p taraması adım sayısı
COMB_LO_FACTOR = 0.45             # p_lo = max(bu * pitch_A, span/N)
COMB_HI_FACTOR = 1.15             # p_hi = min(bu * pitch_A, span)
COMB_TOOTH_HALF_WIDTH = 0.35      # diş doluluğu ±(bu * p) penceresinde sayılır
COMB_SUBHARMONIC_GUARD = 0.97     # en iyi skorun bu oranı içindeki adaylardan en büyüğü seçilir
COMB_MIN_SCORE = 0.35             # R*occ bunun altındaysa comb güvenilmez → None


def estimate_pitch_comb(
    s_sorted: list[float], pitch_A: float, pitch_B: float | None,
) -> tuple[float, float, float] | None:
    """Pitch'i A/B seçiminden değil, doğrudan s-tarağının periyodikliğinden kestir.

    ``pitch_A`` (ürün eni) yelpaze/üst-üste-binme durumunda sistematik şişer (bkz.
    ``estimate_pitch`` docstring — 105 fotonun 95'inde A>B). Bu fonksiyon s
    izdüşümlerinin ne kadar düzenli bir p-periyoduyla dizildiğini doğrudan tarar:
    her aday p için dairesel konsantrasyon (R, tespitler dişlere ne kadar sıkı
    oturuyor) ve diş-doluluk oranını (occ, tepsi boyunca kaç diş dolu) çarpar.

    Alt-harmonik guard: p/2, p/3 ... de yüksek skor verebilir (her gerçek diş
    yarım/üçte-bir periyotta da "dolu" görünür) — bu yüzden en iyi skorun ~%97'si
    içindeki adaylardan EN BÜYÜK p seçilir (gerçek periyot alt-harmoniklerden büyük).

    Returns: ``(pitch_comb, phi_comb, score)`` ya da güvenilmezse ``None``.
    ``phi_comb`` ``_phase_offset`` ile AYNI tanım/konvansiyonu kullanır (dairesel
    ortalama üstünden ``s=0`` referanslı faz, ``[0, p)`` aralığına sarılmış) — bu
    yüzden ``build_lattice``'e doğrudan ``phase_override`` olarak verilebilir.
    """
    n = len(s_sorted)
    if n < COMB_MIN_PRODUCTS:
        return None

    span = s_sorted[-1] - s_sorted[0]
    if span <= 0 or pitch_A <= 0:
        return None

    p_lo = max(COMB_LO_FACTOR * pitch_A, span / n)
    p_hi = min(COMB_HI_FACTOR * pitch_A, span)
    if p_hi <= p_lo:
        return None

    candidates = []
    for i in range(COMB_SEARCH_STEPS + 1):
        p = p_lo + (p_hi - p_lo) * i / COMB_SEARCH_STEPS
        if p <= 0:
            continue
        angles = [2.0 * math.pi * (s % p) / p for s in s_sorted]
        mean_sin = sum(math.sin(a) for a in angles) / n
        mean_cos = sum(math.cos(a) for a in angles) / n
        r = math.hypot(mean_sin, mean_cos)
        mean_angle = math.atan2(mean_sin, mean_cos)
        phi = ((mean_angle / (2.0 * math.pi)) * p) % p

        k_lo = math.floor((s_sorted[0] - phi) / p)
        k_hi = math.ceil((s_sorted[-1] - phi) / p)
        n_teeth = max(1, k_hi - k_lo + 1)
        half = COMB_TOOTH_HALF_WIDTH * p
        occupied = 0
        for k in range(k_lo, k_hi + 1):
            center = phi + k * p
            if any(abs(s - center) <= half for s in s_sorted):
                occupied += 1
        occ = occupied / n_teeth
        score = r * occ
        candidates.append((p, phi, score))

    if not candidates:
        return None

    best_score = max(c[2] for c in candidates)
    if best_score < COMB_MIN_SCORE:
        return None

    plateau = [c for c in candidates if c[2] >= COMB_SUBHARMONIC_GUARD * best_score]
    pitch_comb, phi_comb, score = max(plateau, key=lambda c: c[0])
    return pitch_comb, phi_comb, score


# ── 3a2) Pitch — QR/tepsi-barından okunan KESİN kanal sayısı K biliniyorken ───────
COMB_FIXED_K_TOL = 0.12   # p0 = (sağ_kenar−sol_kenar)/K çevresinde ±bu oran taranır


def estimate_pitch_comb_fixed_k(
    s_sorted: list[float], left_edge_s: float, right_edge_s: float, k: int,
) -> tuple[float, float, int, float] | None:
    """K (kanal sayısı) DIŞARIDAN (QR/tepsi-barı) verilmişken pitch'i kilitle.

    Serbest ``estimate_pitch_comb``'dan FARKI: diş sayısı veriden tahmin EDİLMEZ,
    ``k`` olarak SABİTTİR — comb yalnız hizayı (faz φ) ve dar bir pitch toleransını
    (``p0=(right_edge_s-left_edge_s)/k`` çevresinde ±``COMB_FIXED_K_TOL``) arar.
    Occupancy (R·occ) aynı tanım: R dairesel konsantrasyon, occ = K diş içinde en az
    bir tespit içeren oran.

    Returns: ``(pitch, phi, k_lo, score)`` ya da güvenilmezse (skor < ``COMB_MIN_SCORE``)
    ``None``. ``k_lo``: en soldaki (kenara en yakın) dişin indeksi — merkezler
    ``[phi + (k_lo + j) * pitch for j in range(k)]`` ile üretilir (tam ``k`` kanal,
    ``build_lattice``'in dinamik uzatma/daraltma mantığı DEVREDE DEĞİL — K kesin)."""
    if k < 1:
        return None
    span = right_edge_s - left_edge_s
    if span <= 0:
        return None
    p0 = span / k
    if p0 <= 0:
        return None
    p_lo = (1.0 - COMB_FIXED_K_TOL) * p0
    p_hi = (1.0 + COMB_FIXED_K_TOL) * p0
    n = len(s_sorted)
    if n < 1:
        return None

    best: tuple[float, float, int, float] | None = None
    for i in range(COMB_SEARCH_STEPS + 1):
        p = p_lo + (p_hi - p_lo) * i / COMB_SEARCH_STEPS
        if p <= 0:
            continue
        angles = [2.0 * math.pi * (s % p) / p for s in s_sorted]
        mean_sin = sum(math.sin(a) for a in angles) / n
        mean_cos = sum(math.cos(a) for a in angles) / n
        r = math.hypot(mean_sin, mean_cos)
        mean_angle = math.atan2(mean_sin, mean_cos)
        phi = ((mean_angle / (2.0 * math.pi)) * p) % p

        # +0.5: en soldaki diş MERKEZİ left_edge_s + p/2'ye hizalanır (left_edge_s'in
        # kendisine değil) — diş kenarı değil merkezi kenara oturur, blok sağa doğru
        # tam K*p kadar uzanır (bkz. ölçülmüş hata: bu +0.5 olmadan blok bir yarım-pitch
        # sola kayıyor, sağ uçtaki ürünler son dişin dışında kalıp son slota yığılıyordu).
        k_lo = round((left_edge_s - phi) / p + 0.5)
        half = COMB_TOOTH_HALF_WIDTH * p
        occupied = 0
        for j in range(k):
            center = phi + (k_lo + j) * p
            if any(abs(s - center) <= half for s in s_sorted):
                occupied += 1
        occ = occupied / k
        score = r * occ
        if best is None or score > best[3]:
            best = (p, phi, k_lo, score)

    if best is None or best[3] < COMB_MIN_SCORE:
        return None
    return best


# ── Numaralandırma yönü (GÖREV 1) — sütun/satır ayrımı ~45° sınırında belirsiz ────
# |u[0]|≈|u[1]| bandında hangi eksenin (x/y) "doğru" olduğu kararsız; bu bantta
# MUHAFAZAKAR davranılır — numbering_reversed hesaplanmaz (False kalır), yanlış
# eksende ölçüp yanlışlıkla ters çevirmektense dokunmamak tercih edilir. Ölçülmüş
# (147 foto, 2026-07-22): bu ε ile 0 foto bu banda düşüyor (bkz. pitch_lattice_slots.md).
NUMBERING_AXIS_AMBIGUITY_EPS = 0.1  # ~45°'nin ±4°'si (bileşen farkı cinsinden)


# ── 3c) θ arama — comb skoruyla küçük açı düzeltmesi (yelpaze/perspektif için) ────
THETA_SEARCH_RANGE_DEG = 12.0
THETA_SEARCH_STEP_DEG = 1.0
THETA_ANCHOR_PENALTY_PER_DEG = 0.05 / THETA_SEARCH_RANGE_DEG  # skor birimi / derece
THETA_IMPROVEMENT_MARGIN = 0.02   # θ0'dan sapmak için gereken asgari net (çıpalı) kazanç


def _comb_score_for_axis(
    products: list[dict[str, Any]], u: tuple[float, float], origin: tuple[float, float],
) -> float:
    """Verilen ``u`` ekseninde comb skorunu hesapla (θ taramasında adayları karşılaştırmak
    için). Comb güvenilmezse (``None``) 0.0 döner — arama bu θ'yı cezalandırır, çökmez."""
    ss = sorted(_project((_cx(d), _cy(d)), origin, u) for d in products)
    _pitch, pitch_A, pitch_B, _unreliable = estimate_pitch(products, u, origin)
    comb = estimate_pitch_comb(ss, pitch_A, pitch_B)
    return comb[2] if comb is not None else 0.0


def _search_axis_angle(
    products: list[dict[str, Any]], v0: tuple[float, float], u0: tuple[float, float],
    angle0_deg: float, origin: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    """θ0 (PCA-medyan, ``dominant_axis``) etrafında ±``THETA_SEARCH_RANGE_DEG`` tara;
    comb skorunu θ0'a hafif çıpalı (``THETA_ANCHOR_PENALTY_PER_DEG``) maksimize eden
    açıyı seç. Doğru θ perspektif yelpazede lane'leri dikleştirip comb'u sıkılaştırır
    (bkz. modül docstring — ``dominant_axis`` içindeki ölçülmüş not). θ0'ı belirgin
    (``THETA_IMPROVEMENT_MARGIN``) geçmezse θ0'da kalınır — PCA'dan sürüklenmeyi önler.

    ``origin`` θ'dan bağımsız (ürün merkezlerinin görüntü-uzayı ortalaması) olduğu için
    her aday yalnızca ``v,u,ss`` yeniden izdüşümü + comb taraması gerektirir; kenar
    tespiti ve latis kurma DÖNGÜ DIŞINDA (çağıran, seçilen θ ile bir kez çalıştırır).

    Returns: ``(v, u, angle_deg, changed)``.
    """
    angle0_rad = math.radians(angle0_deg)
    score0 = _comb_score_for_axis(products, u0, origin)

    steps = int(round(THETA_SEARCH_RANGE_DEG / THETA_SEARCH_STEP_DEG))
    best_offset_deg = 0.0
    best_adjusted = score0
    best_v, best_u = v0, u0
    for i in range(-steps, steps + 1):
        offset_deg = i * THETA_SEARCH_STEP_DEG
        if offset_deg == 0.0:
            continue
        angle_rad = angle0_rad + math.radians(offset_deg)
        v = (math.cos(angle_rad), math.sin(angle_rad))
        u = (-v[1], v[0])
        raw = _comb_score_for_axis(products, u, origin)
        adjusted = raw - THETA_ANCHOR_PENALTY_PER_DEG * abs(offset_deg)
        if adjusted > best_adjusted:
            best_adjusted = adjusted
            best_offset_deg = offset_deg
            best_v, best_u = v, u

    if best_offset_deg == 0.0 or (best_adjusted - score0) < THETA_IMPROVEMENT_MARGIN:
        return v0, u0, angle0_deg, False
    return best_v, best_u, round(angle0_deg + best_offset_deg, 2), True


# ── 4) Latis — eşit aralıklı kanallar, faz ürünlere hizalı ────────────────────────
def _phase_offset(ss: list[float], pitch: float) -> float:
    """Latis fazı φ ∈ [0, pitch): kanal merkezlerinin (φ + k·pitch) tespit s-değerlerine
    toplam uzaklığını minimize eder. Dairesel ortalama ile çözülür — her s'nin pitch'e
    göre kalıntısı bir açıya taşınır, açıların ortalaması geri φ'ye çevrilir. Böylece
    kanal merkezleri ürünlere oturur, sınırlar (merkez ± pitch/2) ürünlerin ARASINA düşer."""
    if pitch <= 0 or not ss:
        return 0.0
    angles = [2.0 * math.pi * (s % pitch) / pitch for s in ss]
    mean_sin = sum(math.sin(a) for a in angles) / len(angles)
    mean_cos = sum(math.cos(a) for a in angles) / len(angles)
    mean_angle = math.atan2(mean_sin, mean_cos)
    phi = (mean_angle / (2.0 * math.pi)) * pitch
    return phi % pitch


def build_lattice(
    ss: list[float], pitch: float,
    left_edge_s: float | None, right_edge_s: float | None,
    *, phase_override: float | None = None,
) -> list[float]:
    """Faz-hizalı eşit-pitch kanal merkezleri üret (u-ekseni s uzayında, artan sırada).

    Ürünleri kapsayan tüm kanallar üretilir (aradaki ürünsüz kanallar = iç boş, doğal).
    Kenar biliniyorsa latis, kanalın TAMAMI kenarın içinde kaldığı sürece (merkez ∓
    pitch/2 kenarı geçmeyecek şekilde) dışa uzatılır — kenara tam sığmayan kısmi kanal
    ÜRETİLMEZ (taşmaktansa bir eksik). Kenar yoksa uydurma dolgu yapılmaz.

    ``phase_override`` verilirse (ör. ``estimate_pitch_comb``'un ürettiği ``phi_comb``,
    AYNI dairesel-ortalama konvansiyonuyla) ``_phase_offset`` yerine doğrudan kullanılır."""
    if pitch <= 0 or not ss:
        return list(ss)
    phi = phase_override if phase_override is not None else _phase_offset(ss, pitch)
    s_lo, s_hi = min(ss), max(ss)
    k_lo = round((s_lo - phi) / pitch)
    k_hi = round((s_hi - phi) / pitch)
    if k_hi < k_lo:
        k_lo, k_hi = k_hi, k_lo

    # Kenara doğru uzat: kanal tümüyle kenarın içinde kalmalı (kısmi kanal yok).
    if left_edge_s is not None:
        while phi + (k_lo - 1) * pitch - pitch / 2.0 >= left_edge_s - 1e-6:
            k_lo -= 1
    if right_edge_s is not None:
        while phi + (k_hi + 1) * pitch + pitch / 2.0 <= right_edge_s + 1e-6:
            k_hi += 1

    return [phi + k * pitch for k in range(k_lo, k_hi + 1)]


def extend_lattice_manual(
    centers: list[float], pitch: float, add_low_s: int, add_high_s: int,
) -> list[float]:
    """GÖREV 2 (operatör-döngüsü): en dış üründen ötedeki tamamen boş/seyrek kanallar
    CV ile tespit edilemiyor (bkz. modülün "BİLİNEN SINIRLAMA" notu — 7 ölçülmüş
    başarısızlık). Tek güvenilir yol: operatörün GÖRDÜĞÜ eksik kanal sayısını girmesi.
    Latis, mevcut (zaten güvenilir) pitch ile o yöne uzatılır — YENİ CV/tahmin YOK,
    yalnızca ekstrapolasyon. Yeni kanallar tespitsiz kalır → ``assign_slots`` onları
    doğal olarak ``is_empty=True`` ("ürün tespit edilemedi") işaretler.

    SAF S-UZAYINDA çalışır — ``add_low_s``/``add_high_s`` EKRAN yönü İDDİA ETMEZ
    (düşük-s ucu / yüksek-s ucu). Ekran yönünden (Solda/Sağda/Yukarıda/Aşağıda) bu
    uçlara çeviri TEK YERDE yapılır: ``resolve_screen_extend`` (bkz. o fonksiyonun
    docstring'i — "s-uzayı ekran uzayıymış gibi kullanılmasın" dersi, pitch_lattice_slots.md)."""
    new_centers = list(centers)
    for _ in range(max(0, add_low_s)):
        new_centers.insert(0, new_centers[0] - pitch)
    for _ in range(max(0, add_high_s)):
        new_centers.append(new_centers[-1] + pitch)
    return new_centers


# GÖREV 2 — operatörün seçtiği EKRAN yönü (görüntüdeki gerçek sol/sağ/yukarı/aşağı).
SCREEN_LEFT = "SCREEN_LEFT"
SCREEN_RIGHT = "SCREEN_RIGHT"
SCREEN_UP = "SCREEN_UP"
SCREEN_DOWN = "SCREEN_DOWN"
SCREEN_BOTH = "BOTH"


def resolve_screen_extend(
    direction: str | None, count: int, u: tuple[float, float],
) -> tuple[int, int]:
    """EKRAN yönünü (operatörün seçtiği) s-uzayı uçlarına (``add_low_s``, ``add_high_s``)
    çevirir — bu çeviri BAŞKA HİÇBİR YERDE tekrarlanmaz (tek kaynak).

    ÖLÇÜLMÜŞ DERS (2026-07-22, üçüncü tekrar — bkz. pitch_lattice_slots.md): "s-uzayı
    ekran uzayıymış gibi kullanıldı" örüntüsü burada da neredeyse tekrarlanıyordu —
    ``add_left``/``add_right`` (eski adıyla) s-artan/azalan uçlardı, GERÇEK ekran
    solu/sağı DEĞİL; ``u[0]``'ın işareti kanonikleştirilmediği için (bkz. GÖREV 1,
    Aday A reddi) bazı fotoğraflarda "Solda" seçimi fiilen SAĞA uzatıyordu (ör.
    20.jpeg: dx=+311, ölçülmüş). Bu fonksiyon her çağrıda ``u``'nun GERÇEK işaretine
    bakar, asla varsaymaz.

    Sütun-düzenli (``|u[0]|>=|u[1]|``): ``u[0]>0`` ise s artan = x artan (sağ) →
    SOL=``add_low_s``, SAĞ=``add_high_s``; ``u[0]<0`` ise TERSİ.
    Satır-düzenli: ``u[1]>0`` ise s artan = y artan (aşağı, görüntüde y aşağı büyür) →
    YUKARI=``add_low_s``, AŞAĞI=``add_high_s``; ``u[1]<0`` ise TERSİ.
    ``BOTH``: anlamı düzene göre değişir (sütun-düzenli→sol+sağ, satır-düzenli→
    yukarı+aşağı) ama sonuç HER İKİ DÜZEN İÇİN DE aynı — iki s-ucuna da ``count``
    eklemek, hangi ucun hangi ekran yönüne karşılık geldiğinden BAĞIMSIZ olarak
    "her iki ekran tarafını da genişlet" anlamına gelir; bu yüzden yön/işarete
    bakmadan doğrudan ``(count, count)`` döner.
    Eşleşmeyen/geçersiz ``direction`` (ör. sütun-düzenli fotoda ``SCREEN_UP``) veya
    ``count<=0`` → ``(0, 0)`` (güvenli no-op).

    DÜZEN-FARKINDALIK (2026-07-22): yukarıdaki no-op yalnız "yanlış eksen için yanlış
    yön" durumunu kapsıyordu — ama ``u``'nun YÖNÜ belirlemede kullanılan bileşeni
    gürültü düzeyinde küçükse (ör. satır-düzenli bir fotoda ``u[0]≈0.025``), önceki
    sürüm yine de ``LEFT``/``RIGHT`` sorulduğunda o gürültülü işarete göre sessizce
    bir yöne çeviriyordu — SONUÇ TANIMSIZ, ama sessiz. UI bunu üretmez (``axis_ambiguous``
    kapısı + düzene göre yalnız ilgili yön etiketlerini gösterme ile), fakat bu
    fonksiyon API'den doğrudan da çağrılabildiği için kapıyı burada da TEKRARLA:
    ``LEFT``/``RIGHT`` için ``|u[0]|``, ``UP``/``DOWN`` için ``|u[1]|`` eşiğin
    (``NUMBERING_AXIS_AMBIGUITY_EPS``) altındaysa sessiz no-op yerine açık hata.
    """
    if not direction or count <= 0:
        return 0, 0
    if direction == SCREEN_BOTH:
        return count, count
    if direction in (SCREEN_LEFT, SCREEN_RIGHT) and abs(u[0]) < NUMBERING_AXIS_AMBIGUITY_EPS:
        raise ValueError(
            f"resolve_screen_extend: {direction} için u[0]={u[0]:.4f} eşiğin altında "
            f"(|u[0]|<{NUMBERING_AXIS_AMBIGUITY_EPS}) — bu muhtemelen satır-düzenli bir "
            "fotoğraf, SOL/SAĞ ekran yönü bu eksenden güvenilir çevrilemez."
        )
    if direction in (SCREEN_UP, SCREEN_DOWN) and abs(u[1]) < NUMBERING_AXIS_AMBIGUITY_EPS:
        raise ValueError(
            f"resolve_screen_extend: {direction} için u[1]={u[1]:.4f} eşiğin altında "
            f"(|u[1]|<{NUMBERING_AXIS_AMBIGUITY_EPS}) — bu muhtemelen sütun-düzenli bir "
            "fotoğraf, YUKARI/AŞAĞI ekran yönü bu eksenden güvenilir çevrilemez."
        )
    column_oriented = abs(u[0]) >= abs(u[1])
    if column_oriented:
        if direction == SCREEN_LEFT:
            return (count, 0) if u[0] > 0 else (0, count)
        if direction == SCREEN_RIGHT:
            return (0, count) if u[0] > 0 else (count, 0)
    else:
        if direction == SCREEN_UP:
            return (count, 0) if u[1] > 0 else (0, count)
        if direction == SCREEN_DOWN:
            return (0, count) if u[1] > 0 else (count, 0)
    return 0, 0


def extension_exceeds_frame(
    image_shape: tuple[int, int], origin: tuple[float, float], v: tuple[float, float],
    u: tuple[float, float], t_min: float, t_max: float, pitch: float,
    old_centers: list[float], new_centers: list[float],
) -> bool:
    """Operatörün eklediği kanallar görüntü kadrajının dışına taşıyor mu? Yalnız
    UYARI amaçlı — sessizce kırpma YAPILMAZ (operatör kararı kalıcıdır)."""
    h, w = image_shape[:2]
    added_left = old_centers[0] - new_centers[0] > 1e-6
    added_right = new_centers[-1] - old_centers[-1] > 1e-6
    edges_to_check = []
    if added_left:
        edges_to_check.append(new_centers[0] - pitch / 2.0)
    if added_right:
        edges_to_check.append(new_centers[-1] + pitch / 2.0)
    for s_edge in edges_to_check:
        for t in (t_min, t_max):
            x = origin[0] + t * v[0] + s_edge * u[0]
            y = origin[1] + t * v[1] + s_edge * u[1]
            if x < 0 or x > w or y < 0 or y > h:
                return True
    return False


# ── 5) Tepsi kenarı tespiti (kontur → spiral deseni → güvenli marj) ───────────────
def _line_through_two_points(t1: float, s1: float, t2: float, s2: float) -> dict[str, Any] | None:
    if abs(t2 - t1) < 1e-6:
        return None
    b = (s2 - s1) / (t2 - t1)
    a = s1 - b * t1
    return {"a": a, "b": b}


def _detect_edge_contour(
    image: np.ndarray | None, v: tuple[float, float], u: tuple[float, float], origin: tuple[float, float],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(a) Tepsi konturu: ``detect_tray_corners`` 4 köşe bulursa, sol kenar (TL-BL) ve
    sağ kenar (TR-BR) çiftlerinden gerçek (a,b) sınır DOĞRULARI kurulur."""
    if image is None:
        return None, None
    try:
        corners = detect_tray_corners(image)
    except Exception:
        return None, None
    if corners is None:
        return None, None
    tl, tr, br, bl = corners

    def proj(pt: Any) -> tuple[float, float]:
        return _project((float(pt[0]), float(pt[1])), origin, v), _project((float(pt[0]), float(pt[1])), origin, u)

    t_tl, s_tl = proj(tl)
    t_bl, s_bl = proj(bl)
    t_tr, s_tr = proj(tr)
    t_br, s_br = proj(br)
    left = _line_through_two_points(t_tl, s_tl, t_bl, s_bl)
    right = _line_through_two_points(t_tr, s_tr, t_br, s_br)
    if left is None or right is None:
        return None, None
    if left["a"] > right["a"]:
        left, right = right, left
    return left, right


def _detect_edge_empty_pattern(
    image: np.ndarray | None, v: tuple[float, float], u: tuple[float, float], origin: tuple[float, float],
    products: list[dict[str, Any]], pitch: float, t_min: float, t_max: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(b) Boş spiral/tepsi deseni sezgiseli: son üründen dışa doğru taranan bir şeritteki
    Canny kenar-yoğunluğu, ürün-bandı içindeki temel yoğunluğun belirgin altına düşerse
    o nokta "kenar" sayılır. Güvenilir bulunamazsa None döner."""
    if image is None or pitch <= 0:
        return None, None
    try:
        import cv2
    except Exception:
        return None, None
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    ss = [_project((_cx(d), _cy(d)), origin, u) for d in products]
    if not ss:
        return None, None

    def density_at(s_center: float) -> float:
        half = pitch / 2.0
        count = total = 0
        for frac in np.linspace(-half, half, 7):
            for tf in np.linspace(t_min, t_max, 5):
                x = origin[0] + tf * v[0] + (s_center + frac) * u[0]
                y = origin[1] + tf * v[1] + (s_center + frac) * u[1]
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    total += 1
                    if edges[yi, xi] > 0:
                        count += 1
        return (count / total) if total else 0.0

    def scan(s0: float, direction: int) -> float | None:
        baseline = density_at(s0 - direction * pitch * 0.5)
        if baseline <= EMPTY_PATTERN_MIN_BASELINE:
            return None
        for step in range(1, EMPTY_PATTERN_MAX_STEPS + 1):
            s = s0 + direction * step * pitch
            if density_at(s) < EMPTY_PATTERN_DROP_RATIO * baseline:
                return s0 + direction * (step - 0.5) * pitch
        return None

    left_s = scan(min(ss), -1)
    right_s = scan(max(ss), 1)
    left = {"a": left_s, "b": 0.0} if left_s is not None else None
    right = {"a": right_s, "b": 0.0} if right_s is not None else None
    return left, right


def _estimate_edges(
    image: np.ndarray | None, v: tuple[float, float], u: tuple[float, float], origin: tuple[float, float],
    products: list[dict[str, Any]], pitch: float, s_lo: float, s_hi: float,
    t_min: float, t_max: float,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Sol/sağ kenarı sırayla dene: (a) tepsi konturu → (b) boş desen sezgiseli →
    (c) güvenli marj (uydurma DEĞİL, en dış üründen yarım-pitch dışarı, taşmayı önler).
    İki taraf BAĞIMSIZ seçilir.

    ÖLÇÜLMÜŞ REGRESYON (denendi, uygulanmadı): kadraj-kenarı (görüntü çerçevesinin u
    eksenine izdüşümü) üçüncü kaynak olarak eklenmişti — kontur+spiral ikisi de
    başarısız olduğunda son çare. 147+94 fotoda ölçüldü: aşırı-bölünme arttı (train
    12→17, temmuz 1→9) ve eksik-bölünme (n=6) neredeyse hiç iyileşmedi (train 5→5
    sabit, temmuz 21→19). Kök: tepsi etrafında (duvar/çerçeve) boşluk olan fotolarda
    kadraj köşeleri gerçek tepsi genişliğinden çok daha geniş bir dikdörtgen veriyor,
    tam-dolu tepsilerde bile hayalet boş uçlar üretiyordu (ör. 17.jpeg: 7→8, sayım
    ``0|1|5|4|4|5|7|9`` — baştaki 0 hayalet). 108.jpeg güvende kaldı ama net kazanç
    negatifti, revert edildi. Gerçek tepsi-sınırı (kadraj DEĞİL) olmadan bu yol
    güvenilir değil — sonraki tur ayrı ele alınmalı."""
    kontur_left, kontur_right = _detect_edge_contour(image, v, u, origin)
    pattern_left = pattern_right = None
    if kontur_left is None or kontur_right is None:
        pattern_left, pattern_right = _detect_edge_empty_pattern(
            image, v, u, origin, products, pitch, t_min, t_max)

    if kontur_left is not None:
        left, left_source = kontur_left, "kontur"
    elif pattern_left is not None:
        left, left_source = pattern_left, "spiral"
    else:
        left, left_source = {"a": s_lo - 0.5 * pitch, "b": 0.0}, "fallback"

    if kontur_right is not None:
        right, right_source = kontur_right, "kontur"
    elif pattern_right is not None:
        right, right_source = pattern_right, "spiral"
    else:
        right, right_source = {"a": s_hi + 0.5 * pitch, "b": 0.0}, "fallback"

    # Aşırı-uzatma sınırı: kontur/spiral kaynağı en dıştaki üründen EDGE_MAX_REACH_FACTOR
    # · pitch'ten fazla dışarı düşerse (ör. spiral s=−1124, ~3 kanal hayalet boş) o kaynağı
    # reddedip güvenli marja düş — taşmaktansa (hayalet boş kanal) eksik üret.
    max_reach = EDGE_MAX_REACH_FACTOR * pitch
    if left_source != "fallback" and left["a"] + left["b"] * 0.0 < s_lo - max_reach:
        left, left_source = {"a": s_lo - 0.5 * pitch, "b": 0.0}, "fallback"
    if right_source != "fallback" and right["a"] + right["b"] * 0.0 > s_hi + max_reach:
        right, right_source = {"a": s_hi + 0.5 * pitch, "b": 0.0}, "fallback"

    return left, right, left_source, right_source


# ── 5b) Yerel sınır-snap — kanal içi görünür sınıf-geçişi (kanıt varsa +1) ────────
BOUNDARY_SNAP_MIN_SUBSET = 2        # ayrımın her iki tarafı da en az bu kadar tespit içermeli
BOUNDARY_SNAP_MIN_GAP_FACTOR = 0.4  # ayrım boşluğu >= bu * pitch olmalı (aksi halde gürültü)


def _snap_class_transition_boundaries(
    products: list[dict[str, Any]], ss_all: list[float], pitch: float, boundaries: list[float],
) -> list[float]:
    """Latis kurulduktan SONRA yerel bir rafineri: bir kanala atanmış, u-ekseninde komşu
    ve FARKLI sınıf iki tespit arasında yeterince büyük bir boşluk varsa, bu kanalın
    gerçek bir fiziksel sınırı içine aldığının KANITIdır (etiketleme hatası DEĞİL —
    modül önseli: bir slot bir üründür). O kanalı sınıf-geçiş noktasından böl.

    Global pitch/θ'ya DOKUNMAZ (ölçülmüş pitch_B tercihi gibi genel bir kaynak DEĞİL —
    bkz. ``estimate_pitch`` docstring, o regresyon yapmıştı); yalnızca KANIT olan (görünür
    sınıf değişimi + yeterli boşluk + her iki alt-küme de yeterince kalabalık) kanallarda
    YEREL olarak +1 sınır ekler. Bir kanalda birden çok geçiş varsa EN BÜYÜK boşluklu
    (en güvenilir) tek geçişten böler — çoklu bölme bu artışın kapsamı dışında.

    Returns: (varsa ekleme yapılmış) yeni ``boundaries`` listesi, artan sırada.
    """
    if pitch <= 0 or len(boundaries) < 2:
        return boundaries

    by_channel: dict[int, list[tuple[float, str]]] = {}
    for d, s in zip(products, ss_all):
        idx = None
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= s <= boundaries[i + 1]:
                idx = i
                break
        if idx is None:
            idx = 0 if s < boundaries[0] else len(boundaries) - 2
        by_channel.setdefault(idx, []).append((s, d["category"]))

    inserts: list[float] = []
    for items in by_channel.values():
        cats = {c for _, c in items}
        if len(cats) < 2:
            continue
        items.sort(key=lambda x: x[0])
        candidates = []
        for (s1, c1), (s2, c2) in zip(items, items[1:]):
            if c1 == c2:
                continue
            gap = s2 - s1
            if gap < BOUNDARY_SNAP_MIN_GAP_FACTOR * pitch:
                continue
            split = (s1 + s2) / 2.0
            left_n = sum(1 for s, _ in items if s <= split)
            right_n = sum(1 for s, _ in items if s > split)
            if left_n < BOUNDARY_SNAP_MIN_SUBSET or right_n < BOUNDARY_SNAP_MIN_SUBSET:
                continue
            candidates.append((gap, split))
        if candidates:
            _, split = max(candidates, key=lambda c: c[0])
            inserts.append(split)

    if not inserts:
        return boundaries
    return sorted(boundaries + inserts)


# ── 6) Atama testi — nokta → kanal (sınır doğrusu, kırpma) ────────────────────────
def chain_slot_index_for(x: float, y: float, grid: dict[str, Any]) -> int:
    """Bir noktayı (axis_origin/v/u ile) t,s'ye izdüşür, ``boundary_lines`` içinde s'yi
    kapsayan aralığı bulup sütun indeksini döndürür. Tespit ASLA düşmez (kırpma)."""
    origin, v, u = grid["axis_origin"], grid["axis_v"], grid["axis_u"]
    t = (x - origin[0]) * v[0] + (y - origin[1]) * v[1]
    s = (x - origin[0]) * u[0] + (y - origin[1]) * u[1]
    b_vals = [b["a"] + b["b"] * t for b in grid["boundary_lines"]]

    idx = None
    for i in range(len(b_vals) - 1):
        if b_vals[i] <= s <= b_vals[i + 1]:
            idx = i
            break
    if idx is None:
        idx = 0 if s < b_vals[0] else len(b_vals) - 2
    return int(max(0, min(idx, grid["column_count"] - 1)))


# ── 7) Bütünü birleştiren latis ızgara kurucu ─────────────────────────────────────
def _lane_grid(
    detections: list[dict[str, Any]], column_count_fallback: int,
    empty_class_name: str, image: np.ndarray | None,
    expected_column_count: int | None = None,
    manual_extend_direction: str | None = None, manual_extend_count: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    products = [d for d in detections if d.get("category") != empty_class_name]

    if len(products) < MIN_PRODUCTS_FOR_CHAINS:
        uncertain = _uncertain_grid(column_count_fallback, "latis: yetersiz tespit")
        uncertain["low_detection"] = True
        return uncertain, {"angle_deg": 0.0, "n_chains": 0, "fallback": True}

    # N < COMB_MIN_PRODUCTS (2..5 ürün): comb MİMARİ OLARAK çalışamaz (COMB_MIN_PRODUCTS
    # eşiği), A/B pitch'e sessizce düşer — bu, TANIM GEREĞİ "kendinden emin ama yanlış" bir
    # latis üretir (ölçülmüş: 147+94 fotoda 2-4 ürünlü örneklerde ort. |slot-7| sapması 6.0,
    # bkz. modül docstring'i). Kendinden emin yanlıştansa dürüst belirsizlik: geometri
    # KURMA, ``low_detection`` bayrağıyla işaretle — çağıran (``slot_assigner``) operatör
    # K girerse geometrisiz SAYIM tabanlı rapora düşer.
    if len(products) < COMB_MIN_PRODUCTS:
        uncertain = _uncertain_grid(column_count_fallback, "latis: az tespit — kanal yapısı güvenilir çıkarılamaz")
        uncertain["low_detection"] = True
        return uncertain, {"angle_deg": 0.0, "n_chains": 0, "fallback": True}

    # (1) Tek açı θ ve dik eksen u (kimlikten bağımsız; zincirler yalnız θ için).
    chains = build_chains([dict(d) for d in products])  # açı kestirimi mutate etmesin
    v0, u0, angle0_deg = dominant_axis(chains)
    origin = (float(np.mean([_cx(d) for d in products])), float(np.mean([_cy(d) for d in products])))

    # (1b) θ arama: comb skoruyla θ0'a hafif çıpalı küçük düzeltme (perspektif yelpaze).
    # origin θ'dan bağımsız olduğu için burada ucuz — kenar/latis DÖNGÜ DIŞINDA kalır.
    v, u, angle_deg, theta_search_used = _search_axis_angle(products, v0, u0, angle0_deg, origin)

    ts_all = [_project((_cx(d), _cy(d)), origin, v) for d in products]
    ss_all = [_project((_cx(d), _cy(d)), origin, u) for d in products]
    t_min, t_max = min(ts_all), max(ts_all)
    s_lo, s_hi = min(ss_all), max(ss_all)

    # (2) Pitch: Kaynak A (ürün eni) + Kaynak B (kimlik geçişi), B tercihli — TABAN/öncül.
    pitch, pitch_A, pitch_B, pitch_B_unreliable = estimate_pitch(products, u, origin)

    comb_used = False
    comb_score = None
    fixed_k_used = False
    fixed_k_score = None
    phase_override = None
    centers_override = None

    # (2a) K (kanal sayısı) DIŞARIDAN biliniyorsa (QR/tepsi-barı) — K-kısıtlı comb.
    # Kenarlar bu yolun p0=(sağ−sol)/K hesabı için ÖNCE gerekli; taban (A/B) pitch'iyle
    # bulunur (serbest comb henüz uygulanmadı — bu dal onu hiç kullanmaz).
    if expected_column_count is not None and expected_column_count >= 1:
        k_left_edge, k_right_edge, k_left_source, k_right_source = _estimate_edges(
            image, v, u, origin, products, pitch, s_lo, s_hi, t_min, t_max)
        k_left_s = k_left_edge["a"] if k_left_edge else None
        k_right_s = k_right_edge["a"] if k_right_edge else None
        fixed = None
        if k_left_s is not None and k_right_s is not None:
            fixed = estimate_pitch_comb_fixed_k(ss_all, k_left_s, k_right_s, expected_column_count)
        if fixed is not None:
            pitch, phase_override, k_lo, fixed_k_score = fixed
            fixed_k_used = True
            centers_override = [phase_override + (k_lo + j) * pitch for j in range(expected_column_count)]
            left_edge, right_edge = k_left_edge, k_right_edge
            left_source, right_source = k_left_source, k_right_source

    # (2b) K yoksa/kilitlenemediyse — serbest comb: s-tarağı periyodikliğinden doğrudan
    # kestirim; başarılıysa A/B taban pitch'inin yerini alır (edge taraması da düzeltilmiş
    # pitch'i görsün diye burada, kenar tespitinden ÖNCE uygulanır).
    if not fixed_k_used:
        # sorted(): estimate_pitch_comb s_sorted[0]/[-1]'i min/maks varsayar (span, k_lo/k_hi
        # bunlardan türer) — ss_all ürün-tespit sırasında (YOLO tarama sırası), s-sıralı DEĞİL;
        # sıralanmadan geçilirse span negatif/anlamsız çıkabilir (ölçülmüş: comb'un asılsız
        # reddinin/yanlış-p seçiminin kök nedeni).
        comb = estimate_pitch_comb(sorted(ss_all), pitch_A, pitch_B)
        comb_used = comb is not None
        if comb_used:
            pitch, phase_override, comb_score = comb

        # (3) Kenarlar (kontur → spiral → güvenli marj, aşırı-uzatma klample), pitch birimiyle.
        left_edge, right_edge, left_source, right_source = _estimate_edges(
            image, v, u, origin, products, pitch, s_lo, s_hi, t_min, t_max)

    left_edge_s = (left_edge["a"] + left_edge["b"] * 0.0) if left_edge else None
    right_edge_s = (right_edge["a"] + right_edge["b"] * 0.0) if right_edge else None

    # (4) Latis: K-kısıtlı yolda merkezler ZATEN tam K adet üretildi (build_lattice'in
    # dinamik uzatma/daraltma mantığı bu yolda DEVREDE DEĞİL — K kesin). Aksi halde
    # faz-hizalı eşit-pitch serbest latis (sınırlar ürünlerin arasına düşer).
    if centers_override is not None:
        centers = centers_override
    else:
        centers = build_lattice(ss_all, pitch, left_edge_s, right_edge_s, phase_override=phase_override)

    # (4b) GÖREV 2: operatörün gördüğü ama latis'in kaçırdığı boş kanalları, mevcut
    # (güvenilir) pitch ile MEVCUT eksene uzat — yeni CV/tahmin yok. Ekran yönü → s-uzayı
    # çevirisi TEK YERDE (``resolve_screen_extend``), burada TEKRARLANMAZ.
    manual_extend_left, manual_extend_right = resolve_screen_extend(
        manual_extend_direction, manual_extend_count, u,
    )
    extend_frame_warning = False
    if manual_extend_left or manual_extend_right:
        pre_extend_centers = list(centers)
        centers = extend_lattice_manual(centers, pitch, manual_extend_left, manual_extend_right)
        extend_frame_warning = extension_exceeds_frame(
            image.shape, origin, v, u, t_min, t_max, pitch, pre_extend_centers, centers,
        ) if image is not None else False

    column_count = len(centers)

    # Sağlık kapısı: pitch tahmini çökerse kanal sayısı patlar → belirsize düş.
    if column_count > MAX_CHANNELS or column_count < 1:
        log.warning("latis kapısı: %d kanal (pitch=%.1f) — pitch tahmini çökmüş, belirsize düşülüyor",
                    column_count, pitch)
        uncertain = _uncertain_grid(column_count_fallback, f"latis kanal sayısı anlamsız ({column_count})")
        return uncertain, {"angle_deg": angle_deg, "n_chains": column_count, "fallback": True,
                           "pitch_A": round(pitch_A, 2)}

    # (5) Sınırlar = kanal merkezlerinin ortası (hepsi b=0 → yapısal olarak kesişmez).
    boundaries = [centers[0] - pitch / 2.0]
    for c1, c2 in zip(centers, centers[1:]):
        boundaries.append((c1 + c2) / 2.0)
    boundaries.append(centers[-1] + pitch / 2.0)

    # (5b) Yerel sınır-snap: kanal içi görünür sınıf-geçişi kanıtı varsa +1 sınır.
    # K-kısıtlı VE serbest yoldan gelen latis'e aynı şekilde uygulanır.
    n_boundary_snaps = 0
    snapped = _snap_class_transition_boundaries(products, ss_all, pitch, boundaries)
    if len(snapped) > len(boundaries) and len(snapped) - 1 <= MAX_CHANNELS:
        n_boundary_snaps = len(snapped) - len(boundaries)
        boundaries = snapped
        centers = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(len(boundaries) - 1)]
        column_count = len(centers)

    boundary_lines = [{"a": b, "b": 0.0} for b in boundaries]
    lane_lines = [{"a": c, "b": 0.0, "chain": None, "t0": 0.0} for c in centers]

    # GÖREV 1 (Aday B, post-hoc, DÜZEN-FARKINDA): numaralandırma yönü. GEOMETRİYE
    # (centers/boundaries/atama) DOKUNMAZ — comb/pitch zaten hesaplandı, buradan sonrası
    # SADECE etiketleme (bkz. slot_assigner._slot_record: ``numbering_reversed``).
    #
    # ÖLÇÜLMÜŞ DÜZELTME (2026-07-22): ilk sürüm s-artan sıra ile GERÇEK GÖRÜNTÜ X'i
    # arasındaki korelasyona bakıyordu — 147 fotoda 33 "ters" vaka buldu. Ama 15'i
    # SATIR-düzenli tepsilerdi (|u_y|>|u_x|, θ≈0-6° — ör. üstten çekilmiş spiral tepsi,
    # kanallar yatay, doğru numaralandırma YUKARIDAN-AŞAĞIYA olmalı): bunlarda x zaten
    # anlamsız bir eksen, gerçek metrik (y-korelasyonu) B'den ÖNCE 15/15 doğruydu — kör
    # x-kontrolü bunları YANLIŞLIKLA ters çevirip 15/15'ini BOZDU (sessiz regresyon,
    # hiçbir agregat metrik yakalamadı). Gerçek sütun-düzenli ters vaka sayısı 33 DEĞİL,
    # 18'di. Ders: fix'i, fix'in kendisinin bir alt kümede geçersiz saydığı metrikle
    # doğrulamak döngüsel doğrulamadır (bkz. pitch_lattice_slots.md).
    #
    # DÜZELTME: önce dominant_axis'in KENDİ dal ayrımını (``|u[0]|>=|u[1]|``) kullanarak
    # tepsinin sütun-düzenli mi satır-düzenli mi olduğuna karar ver, SONRA korelasyonu
    # YALNIZ o eksende hesapla. |u[0]|≈|u[1]| (~45°) sınır durumunda MUHAFAZAKAR davran:
    # hangi eksenin "doğru" olduğu belirsizse müdahale ETME (numbering_reversed=False) —
    # yanlış eksende ölçüp yanlışlıkla ters çevirmektense dokunmamak güvenli.
    numbering_reversed = False
    ambiguous_orientation = abs(abs(u[0]) - abs(u[1])) < NUMBERING_AXIS_AMBIGUITY_EPS
    if not ambiguous_orientation:
        column_oriented = abs(u[0]) >= abs(u[1])
        proj = [origin[0] + s * u[0] for s in centers] if column_oriented else \
               [origin[1] + s * u[1] for s in centers]
        if len(proj) >= 2 and len(set(proj)) > 1:
            n_c = len(proj)
            mean_i = (n_c - 1) / 2.0
            mean_p = sum(proj) / n_c
            cov = sum((i - mean_i) * (p - mean_p) for i, p in enumerate(proj))
            numbering_reversed = cov < 0

    # (6) Telemetri — TEK KAYNAK: atama ``chain_slot_index_for`` ile (assign_slots ile
    # aynı sınır-doğrusu testi); burada AYRI bir "en yakın merkez" ataması YOK. Bins
    # yalnız boş-kanal ve ihlal sayımı içindir; yetkili slot dağılımı assign_slots'ta.
    axis_grid = {"axis_origin": [origin[0], origin[1]], "axis_v": [v[0], v[1]],
                 "axis_u": [u[0], u[1]], "boundary_lines": boundary_lines,
                 "column_count": column_count}
    filled = [False] * column_count
    for d in products:
        filled[chain_slot_index_for(_cx(d), _cy(d), axis_grid)] = True

    filled_k = [k for k in range(column_count) if filled[k]]
    first_f = filled_k[0] if filled_k else 0
    last_f = filled_k[-1] if filled_k else column_count - 1
    n_head = first_f
    n_tail = (column_count - 1) - last_f
    n_interior = sum(1 for k in range(first_f, last_f + 1) if not filled[k])

    # Sınır-merkez ihlali (DOĞRULAMA): bir sınır bir kutunun MERKEZİ çevresine (±%20
    # yarı-genişlik) düşerse. Faz-hizalı latiste sınırlar ürün arasına düştüğü için ~0
    # beklenir; kenara yakın teğet geçişler (görüntüde üst üste binen ürünler) sayılmaz.
    n_sinif_gecis = 0
    for d, s in zip(products, ss_all):
        half = _u_extent(d, u) / 2.0
        for b in boundaries[1:-1]:
            if abs(b - s) <= 0.2 * half:
                n_sinif_gecis += 1
                break

    lane_meta = {
        "angle_deg": angle_deg,
        "n_chains": column_count,
        "fallback": False,
        "pitch": round(pitch, 2),
        "pitch_A": round(pitch_A, 2),
        "pitch_B": round(pitch_B, 2) if pitch_B is not None else None,
        "pitch_B_unreliable": pitch_B_unreliable,
        "pitch_comb_used": comb_used,
        "pitch_comb_score": round(comb_score, 3) if comb_score is not None else None,
        "fixed_k_used": fixed_k_used,
        "fixed_k_score": round(fixed_k_score, 3) if fixed_k_score is not None else None,
        "expected_column_count": expected_column_count,
        "n_boundary_snaps": n_boundary_snaps,
        "numbering_reversed": numbering_reversed,
        "manual_extend_direction": manual_extend_direction,
        "manual_extend_count": manual_extend_count,
        "manual_extend_add_low_s": manual_extend_left,
        "manual_extend_add_high_s": manual_extend_right,
        "extend_frame_warning": extend_frame_warning,
        "theta_search_used": theta_search_used,
        "angle0_deg": angle0_deg,
        "phase": round(centers[0] - (centers[0] // pitch) * pitch, 2) if pitch > 0 else 0.0,
        "n_interior_empty": n_interior,
        "n_head_empty": n_head,
        "n_tail_empty": n_tail,
        "edge_source_left": left_source,
        "edge_source_right": right_source,
        "column_count": column_count,
        "rail_anchor_used": False,
        "boundary_center_violations": n_sinif_gecis,
    }
    log.info(
        "latis: v=%.1f°, pitch=%.1f (A=%.1f, B=%s, comb=%s, K=%s), %d kanal (iç boş=%d, baş=%d[%s], "
        "son=%d[%s]), sınır-merkez ihlali=%d",
        angle_deg, pitch, pitch_A, f"{pitch_B:.1f}" if pitch_B is not None else "yok",
        f"{comb_score:.2f}" if comb_used else "yok",
        f"{expected_column_count}@{fixed_k_score:.2f}" if fixed_k_used else "yok",
        column_count, n_interior, n_head, left_source, n_tail, right_source, n_sinif_gecis,
    )

    grid = {
        "grid_status": "ok",
        "column_count": int(column_count),
        "pitch": float(pitch),
        "rect_width": None, "rect_height": float(max(t_max - t_min, 1.0)),
        "centers": list(centers),
        "boundaries": list(boundaries),
        "corners": None,
        "H": None, "H_inv": None,
        "grid_source": "zincir",
        "shelf_number_from_bar": None,
        "confident": True,
        "axis_origin": [origin[0], origin[1]],
        "axis_v": [v[0], v[1]], "axis_u": [u[0], u[1]],
        "t_min": float(t_min), "t_max": float(t_max),
        "lane_lines": lane_lines,
        "boundary_lines": boundary_lines,
        "edge_left": left_edge, "edge_right": right_edge,
        "lane_meta": lane_meta,
        "numbering_reversed": numbering_reversed,
    }
    return grid, lane_meta


def estimate_grid_lane_based(
    image: np.ndarray | None,
    column_count: int,
    detections: list[dict[str, Any]],
    *,
    corners_override: Any | None = None,
    empty_class_name: str = "empty_slot",
    expected_column_count: int | None = None,
    manual_extend_direction: str | None = None,
    manual_extend_count: int = 0,
) -> dict[str, Any]:
    """Slot ızgarasını kimlikten bağımsız faz-hizalı LATİS ile üret (PRIMARY yol).
    Operatörün elle işaretlediği köşeler (``corners_override``) tek istisna: tam
    homografiye güvenilir (operatörün kararı otomatik geometriden üstündür).

    ``expected_column_count``: QR/tepsi-barından okunan KESİN kanal sayısı K (ör. rafın
    kendi barındaki "11...17" aralığı → K=7). Verilirse latis K-kısıtlı comb ile kurulur
    (bkz. ``estimate_pitch_comb_fixed_k``) — kanal SAYISI artık tahmin değil, veridir;
    comb yalnız hizayı bulur. ``None`` (varsayılan) → mevcut serbest comb davranışı
    DEĞİŞMEZ. ``column_count`` (statik config değeri) ile KARIŞTIRILMAMALI: o yalnızca
    ``grid_status="belirsiz"`` düşüşünde fallback sütun sayısıdır.

    ``manual_extend_direction``/``manual_extend_count`` (GÖREV 2): operatörün GÖRDÜĞÜ
    ama latis'in kaçırdığı boş kanal sayısı ve GERÇEK EKRAN yönü (``SCREEN_LEFT`` /
    ``SCREEN_RIGHT`` / ``SCREEN_UP`` / ``SCREEN_DOWN`` / ``BOTH``). Ekran yönünden
    s-uzayına çeviri ``resolve_screen_extend`` ile (u'nun GERÇEK işaretine bakarak,
    tek yerde) yapılır — mevcut pitch ile ekstrapolasyon, yeni CV/tahmin yok.
    ``direction=None`` (varsayılan) → davranış DEĞİŞMEZ.
    """
    if corners_override is not None:
        grid = estimate_homography_grid(image, column_count, corners_override=corners_override)
        grid["lane_meta"] = {"kaynak": "manuel", "n_chains": None}
        return grid

    grid, _meta = _lane_grid(detections, column_count, empty_class_name, image,
                              expected_column_count=expected_column_count,
                              manual_extend_direction=manual_extend_direction,
                              manual_extend_count=manual_extend_count)
    return grid
