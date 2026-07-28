"""GÖREV 2 kalıcı regresyon: SCREEN_LEFT/RIGHT/UP/DOWN gerçek ekran koordinatında
doğru yöne uzatmalı — u'nun işaretinden, numbering_reversed'dan ve düzenden (sütun/
satır) BAĞIMSIZ. İşaret mantığını değil SONUCU (gerçek görüntü dx/dy) test eder;
çeviri tablosu (``resolve_screen_extend``) ters yazılırsa bu test patlar.

5 foto: en az 2 satır-düzenli, en az 2 ``numbering_reversed=True`` (bkz.
pitch_lattice_slots.md — "s-uzayı ekran uzayıymış gibi kullanıldı" örüntüsü, 3. vaka).
Gerçek YOLO + deskew hattını çalıştırır (modeli bir kez yükler) — yavaş ama gerçek.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pytest

from src.analysis.product_anchored_grid import (
    SCREEN_BOTH, SCREEN_DOWN, SCREEN_LEFT, SCREEN_RIGHT, SCREEN_UP,
    estimate_grid_lane_based, resolve_screen_extend,
)
from src.analysis.skew_corrector import deskew
from src.models.predictor import load_model, predict_shelf

TRAIN_DIR = Path("/Users/gulfemsolak/Desktop/train")
TEMMUZ_DIR = Path("/Users/gulfemsolak/Desktop/10 Temmuz new data")

# (klasör, dosya, beklenen_düzen, beklenen_numbering_reversed) — keşifle doğrulanmış.
CASES = [
    (TRAIN_DIR, "20.jpeg", "column", True),
    (TRAIN_DIR, "37.jpeg", "column", True),
    (TEMMUZ_DIR, "WhatsApp Image 2026-07-09 at 20.52.21.jpeg", "row", False),
    (TEMMUZ_DIR, "WhatsApp Image 2026-07-09 at 20.52.06 (6).jpeg", "row", False),
    (TRAIN_DIR, "108.jpeg", "column", False),
]


@pytest.fixture(scope="module")
def model():
    return load_model("models/best.pt")


def _grid_for(model, folder: Path, fname: str, **kw):
    img = cv2.imread(str(folder / fname))
    assert img is not None, f"okunamadı: {folder / fname}"
    work = deskew(img)["image"]
    result = predict_shelf(image_path=work, model=model, conf_threshold=0.5,
                            config_path="config/config.yaml")
    dets = result["detections"]
    return estimate_grid_lane_based(work, 7, dets, **kw)


@pytest.mark.parametrize("folder,fname,expected_orientation,expected_reversed", CASES)
def test_screen_direction_matches_real_image_geometry(
    model, folder, fname, expected_orientation, expected_reversed,
):
    if not (folder / fname).exists():
        pytest.skip(f"kalibrasyon fotoğrafı bulunamadı: {folder / fname}")

    grid0 = _grid_for(model, folder, fname)
    assert grid0.get("grid_source") == "zincir"
    u = grid0["axis_u"]
    column_oriented = abs(u[0]) >= abs(u[1])
    orientation = "column" if column_oriented else "row"
    assert orientation == expected_orientation, (
        f"{fname}: beklenen düzen {expected_orientation}, ölçülen {orientation} (u={u}) — "
        "kalibrasyon fotoğrafı/beklentisi güncel değil, testi düzenle."
    )
    assert grid0.get("numbering_reversed") == expected_reversed, (
        f"{fname}: beklenen numbering_reversed={expected_reversed}, "
        f"ölçülen {grid0.get('numbering_reversed')}"
    )

    directions = (SCREEN_LEFT, SCREEN_RIGHT) if column_oriented else (SCREEN_UP, SCREEN_DOWN)
    for direction in directions:
        grid1 = _grid_for(model, folder, fname,
                           manual_extend_direction=direction, manual_extend_count=2)
        origin, u1 = grid1["axis_origin"], grid1["axis_u"]
        old_centers, new_centers = grid0["centers"], grid1["centers"]

        grew_low = new_centers[0] < old_centers[0] - 1e-6
        new_s = new_centers[0] if grew_low else new_centers[-1]
        old_s = old_centers[0] if grew_low else old_centers[-1]
        dx = (origin[0] + new_s * u1[0]) - (origin[0] + old_s * u1[0])
        dy = (origin[1] + new_s * u1[1]) - (origin[1] + old_s * u1[1])

        if direction == SCREEN_LEFT:
            assert dx < 0, f"{fname} SCREEN_LEFT: dx={dx:.1f} (< 0 olmalı)"
        elif direction == SCREEN_RIGHT:
            assert dx > 0, f"{fname} SCREEN_RIGHT: dx={dx:.1f} (> 0 olmalı)"
        elif direction == SCREEN_UP:
            assert dy < 0, f"{fname} SCREEN_UP: dy={dy:.1f} (< 0 olmalı)"
        elif direction == SCREEN_DOWN:
            assert dy > 0, f"{fname} SCREEN_DOWN: dy={dy:.1f} (> 0 olmalı)"


def test_mismatched_direction_raises_on_row_oriented_photo(model):
    """satır-düzenli fotoda (``|u[0]|`` gürültü düzeyinde) ``SCREEN_LEFT`` istenirse
    ``resolve_screen_extend`` sessizce (0, 0) döndürmek yerine açıkça hata vermeli —
    UI bu kombinasyonu üretmez ama API'den doğrudan çağrılabilir, kapı burada da olmalı."""
    folder, fname = TEMMUZ_DIR, "WhatsApp Image 2026-07-09 at 20.52.21.jpeg"
    if not (folder / fname).exists():
        pytest.skip(f"kalibrasyon fotoğrafı bulunamadı: {folder / fname}")

    grid = _grid_for(model, folder, fname)
    u = grid["axis_u"]
    assert abs(u[0]) < abs(u[1]), f"{fname}: satır-düzenli bekleniyordu, u={u}"

    with pytest.raises(ValueError):
        resolve_screen_extend(SCREEN_LEFT, 2, u)


@pytest.mark.parametrize(
    "folder,fname,expected_orientation",
    [
        (TRAIN_DIR, "20.jpeg", "column"),
        (TEMMUZ_DIR, "WhatsApp Image 2026-07-09 at 20.52.21.jpeg", "row"),
    ],
)
def test_screen_both_extends_both_screen_ends(model, folder, fname, expected_orientation):
    """``SCREEN_BOTH`` her iki ekran ucunu da genişletmeli — hem sütun-düzenli hem
    satır-düzenli fotoğrafta. Mevcut 5-fotoluk test yalnız LEFT/RIGHT/UP/DOWN'ı
    doğruluyor, BOTH hiç test edilmemişti."""
    if not (folder / fname).exists():
        pytest.skip(f"kalibrasyon fotoğrafı bulunamadı: {folder / fname}")

    grid0 = _grid_for(model, folder, fname)
    u = grid0["axis_u"]
    column_oriented = abs(u[0]) >= abs(u[1])
    orientation = "column" if column_oriented else "row"
    assert orientation == expected_orientation, (
        f"{fname}: beklenen düzen {expected_orientation}, ölçülen {orientation} (u={u})"
    )

    grid1 = _grid_for(model, folder, fname, manual_extend_direction=SCREEN_BOTH, manual_extend_count=2)
    old_centers, new_centers = grid0["centers"], grid1["centers"]

    assert new_centers[0] < old_centers[0] - 1e-6, (
        f"{fname} BOTH: düşük-s ucu büyümedi (old={old_centers[0]}, new={new_centers[0]})"
    )
    assert new_centers[-1] > old_centers[-1] + 1e-6, (
        f"{fname} BOTH: yüksek-s ucu büyümedi (old={old_centers[-1]}, new={new_centers[-1]})"
    )
