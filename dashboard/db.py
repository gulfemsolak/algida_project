"""SQLite-backed persistent analysis history for the IceVision dashboard.

Source of truth for the Dashboard's Recent Analyses list and top KPIs —
survives page reloads and process restarts, unlike st.session_state.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

DB_PATH = Path("data/analysis_history.db")
THUMBNAIL_DIR = Path("data/thumbnails")

# Fill rate below this counts as a restock alert — matches theme.fill_rate_color's
# "critical" cutoff, so the Dashboard's ALERTS count agrees with the pink/critical
# color cues used elsewhere (e.g. the fill-rate ring on the Shelf Analysis page).
ALERT_FILL_THRESHOLD = 0.40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_products INTEGER,
    empty_slots INTEGER,
    fill_rate REAL,
    avg_confidence REAL,
    shelf_rows INTEGER,
    class_counts TEXT,
    thumbnail_path TEXT,
    qr_found BOOLEAN DEFAULT 0,
    qr_values TEXT DEFAULT '',
    qr_verified BOOLEAN DEFAULT NULL,
    qr_message TEXT DEFAULT ''
);
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp DESC);"

# GÖREV 2: operatörün "kanal sayısı doğru mu?" / yön-uzatma müdahalelerinin kaydı —
# CV boş-kanal tespitini güvenilir yapamadığı için (bkz. product_anchored_grid.py
# "BİLİNEN SINIRLAMA") operatör ne sıklıkla düzeltiyor, ölçülebilsin diye ayrı tablo.
# add_left/add_right ÇÖZÜLMÜŞ s-uzayı değerleridir (``resolve_screen_extend`` çıktısı),
# ekran yönü DEĞİL — ``direction_label`` operatörün GÖRDÜĞÜ/seçtiği etiketi (ör.
# "Solda", "Yukarıda") taşır; ikisi ayrışırsa (çeviri hatası) log'da fark edilebilir.
_MANUAL_EXTENSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_extensions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    direction_label TEXT DEFAULT '',
    add_left INTEGER DEFAULT 0,
    add_right INTEGER DEFAULT 0,
    k_before INTEGER,
    k_after INTEGER
);
"""
_MANUAL_EXTENSIONS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_manual_extensions_timestamp "
    "ON manual_extensions(timestamp DESC);"
)

# Columns added after the initial release — applied to pre-existing DB files
# via ALTER TABLE, since CREATE TABLE IF NOT EXISTS is a no-op on them.
_MIGRATION_COLUMNS: dict[str, str] = {
    "qr_found": "BOOLEAN DEFAULT 0",
    "qr_values": "TEXT DEFAULT ''",
    "qr_verified": "BOOLEAN DEFAULT NULL",
    "qr_message": "TEXT DEFAULT ''",
}
_MANUAL_EXTENSIONS_MIGRATION_COLUMNS: dict[str, str] = {
    "direction_label": "TEXT DEFAULT ''",
}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)")}
    for col, decl in _MIGRATION_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {decl}")

    existing_ext = {row["name"] for row in conn.execute("PRAGMA table_info(manual_extensions)")}
    for col, decl in _MANUAL_EXTENSIONS_MIGRATION_COLUMNS.items():
        if col not in existing_ext:
            conn.execute(f"ALTER TABLE manual_extensions ADD COLUMN {col} {decl}")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the DB file, table, and thumbnail directory if they don't exist yet.
    Safe to call on every page load — all operations are idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(_SCHEMA)
        conn.execute(_INDEX)
        conn.execute(_MANUAL_EXTENSIONS_SCHEMA)
        conn.execute(_MANUAL_EXTENSIONS_INDEX)
        _migrate_schema(conn)
        conn.execute(_MANUAL_EXTENSIONS_INDEX)


def save_thumbnail(image_bgr: np.ndarray | None, max_width: int = 120, quality: int = 60) -> str | None:
    """Downscale a BGR frame to ~120px wide and save it as a small JPEG file.
    Returns the file path (str) to store in the analyses.thumbnail_path column."""
    if image_bgr is None:
        return None
    try:
        h, w = image_bgr.shape[:2]
        scale = max_width / w
        new_w, new_h = max_width, max(1, round(h * scale))
        thumb = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        path = THUMBNAIL_DIR / fname
        cv2.imwrite(str(path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return str(path)
    except Exception:
        return None


def save_analysis(data: dict[str, Any]) -> int:
    """Insert one analysis record. Recognized keys: filename, total_products,
    empty_slots, fill_rate, avg_confidence, shelf_rows, class_counts (dict or
    JSON string), thumbnail_path, qr_info (dict from qr_reader.verify_shelf_qr).
    Returns the new row's id."""
    class_counts = data.get("class_counts", {})
    if isinstance(class_counts, dict):
        class_counts = json.dumps(class_counts)

    qr_info = data.get("qr_info") or {}
    qr_found = bool(qr_info.get("qr_found", False))
    qr_values = json.dumps(qr_info.get("qr_values", []))
    qr_verified = qr_info.get("verified")
    qr_message = qr_info.get("message", "")

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses
                (filename, timestamp, total_products, empty_slots, fill_rate,
                 avg_confidence, shelf_rows, class_counts, thumbnail_path,
                 qr_found, qr_values, qr_verified, qr_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("filename"),
                _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("total_products", 0),
                data.get("empty_slots", 0),
                data.get("fill_rate", 0.0),
                data.get("avg_confidence", 0.0),
                data.get("shelf_rows", 0),
                class_counts,
                data.get("thumbnail_path"),
                qr_found,
                qr_values,
                qr_verified,
                qr_message,
            ),
        )
        return cur.lastrowid


def save_manual_extension(
    filename: str | None, add_left: int, add_right: int, k_before: int, k_after: int,
    direction_label: str = "",
) -> int:
    """GÖREV 2: operatörün yön-uzatma müdahalesini kaydet (hangi foto, hangi yön,
    kaç kanal, öncesi/sonrası K) — 'operatör ne sıklıkla düzeltiyor' sonradan
    ölçülebilsin diye. ``direction_label`` operatörün GÖRDÜĞÜ etiket (ör. "Solda"),
    ``add_left``/``add_right`` ÇÖZÜLMÜŞ s-uzayı değerleridir (``resolve_screen_extend``
    çıktısı) — ikisi ayrışırsa (çeviri hatası) log'da fark edilebilir. Returns the
    new row's id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO manual_extensions
                (filename, timestamp, direction_label, add_left, add_right, k_before, k_after)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                direction_label,
                add_left, add_right, k_before, k_after,
            ),
        )
        return cur.lastrowid


def get_manual_extension_stats() -> dict[str, Any]:
    """Kaç fotoda operatör müdahale etti, toplam kaç kanal eklendi — özet istatistik."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt, SUM(add_left) AS total_left, SUM(add_right) AS total_right
            FROM manual_extensions
            """
        ).fetchone()
    return {
        "count": row["cnt"] or 0,
        "total_left": row["total_left"] or 0,
        "total_right": row["total_right"] or 0,
    }


def _filter_clause(search_query: str | None, date_from: Any, date_to: Any) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if search_query:
        clauses.append("filename LIKE ?")
        params.append(f"%{search_query}%")
    if date_from:
        clauses.append("date(timestamp) >= date(?)")
        params.append(str(date_from))
    if date_to:
        clauses.append("date(timestamp) <= date(?)")
        params.append(str(date_to))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def get_analyses(
    page: int = 1,
    per_page: int = 10,
    search_query: str | None = None,
    date_from: Any = None,
    date_to: Any = None,
) -> list[dict[str, Any]]:
    """One page of analyses, newest first, with class_counts decoded back to a dict."""
    where, params = _filter_clause(search_query, date_from, date_to)
    offset = max(0, (page - 1) * per_page)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM analyses {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (*params, per_page, offset),
        ).fetchall()

    results = []
    for row in rows:
        entry = dict(row)
        try:
            entry["class_counts"] = json.loads(entry["class_counts"]) if entry["class_counts"] else {}
        except (TypeError, json.JSONDecodeError):
            entry["class_counts"] = {}
        try:
            entry["qr_values"] = json.loads(entry["qr_values"]) if entry.get("qr_values") else []
        except (TypeError, json.JSONDecodeError):
            entry["qr_values"] = []
        results.append(entry)
    return results


def get_total_count(search_query: str | None = None, date_from: Any = None, date_to: Any = None) -> int:
    where, params = _filter_clause(search_query, date_from, date_to)
    with _connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM analyses {where}", params).fetchone()
    return row[0] if row else 0


def get_aggregate_stats() -> dict[str, Any]:
    """Aggregate KPIs for the Dashboard's top metric cards, across all history."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt,
                   AVG(fill_rate) AS avg_fill,
                   SUM(total_products) AS products,
                   SUM(CASE WHEN fill_rate < ? THEN 1 ELSE 0 END) AS alerts
            FROM analyses
            """,
            (ALERT_FILL_THRESHOLD,),
        ).fetchone()

    count = row["cnt"] or 0
    return {
        "count": count,
        "avg_fill_rate": row["avg_fill"] if count else None,
        "products_detected": row["products"] or 0,
        "alerts": row["alerts"] or 0,
    }


def delete_analysis(analysis_id: int) -> None:
    """Remove one record and its thumbnail file, if any."""
    with _connect() as conn:
        row = conn.execute("SELECT thumbnail_path FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    if row and row["thumbnail_path"]:
        Path(row["thumbnail_path"]).unlink(missing_ok=True)


def clear_all() -> None:
    """Wipe every record and all thumbnail files."""
    with _connect() as conn:
        rows = conn.execute("SELECT thumbnail_path FROM analyses").fetchall()
        conn.execute("DELETE FROM analyses")
    for row in rows:
        if row["thumbnail_path"]:
            Path(row["thumbnail_path"]).unlink(missing_ok=True)
