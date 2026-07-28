"""IceVision theme: single source of truth for colors, CSS, and chart styling.

"Soğuk Zincir" visual identity — a freezer cabinet at night. Deep frost-navy
surfaces, glacier-blue data accents, and Algida red as the single brand pulse
(wordmark heart, page-header heat line, active nav, primary actions, critical
status). Display type is Sora; data is JetBrains Mono.

Call ``apply_theme()`` once per page, then use ``page_header()`` /
``section_header()`` for consistent structure.
"""
from __future__ import annotations

import html as _html

import streamlit as st

# ── Base surfaces ─────────────────────────────────────────────────────────────
BG = "#070B14"
SURFACE_1 = "#0C121E"
SURFACE_2 = "#101828"
SURFACE_3 = "#16202F"

# ── Accents ───────────────────────────────────────────────────────────────────
ICE = "#5CB8FF"            # glacier blue — data, interaction, links
MINT = "#2FE6A8"           # OK / healthy stock
AMBER = "#FFB547"          # warning / low stock
ALGIDA_RED = "#FF3355"     # brand pulse on dark + critical status
ALGIDA_RED_DEEP = "#E4002B"  # true brand red — primary button fills
QR_WARN = "#FFB800"        # QR "not found" badge — spec color, distinct from AMBER

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT_PRIMARY = "#EDF3FB"
TEXT_SECONDARY = "#8195B5"
TEXT_TERTIARY = "#46536E"
TEXT_ACCENT = ICE

# ── Borders ───────────────────────────────────────────────────────────────────
BORDER = "#1B2536"
BORDER_ACTIVE = "#5CB8FF33"
DIVIDER = "#131B2A"

DISPLAY_FONT = "'Sora', system-ui, -apple-system, sans-serif"
MONO_FONT = "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace"
SANS_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# ── Product identity (exact hexes) ───────────────────────────────────────────
# Matches the real trained model's 35 classes (data/real_dataset/data.yaml).
# Hue picked per flavor (chocolate → brown, fruit → its own color, etc.), grouped
# loosely by brand so a shelf photo with several variants of one brand still reads
# as distinct dots/boxes.
PRODUCT_COLORS: dict[str, str] = {
    "algida_boom_gold": "#FFC94A",
    "algida_klasik": "#5AD1E6",
    "algida_klasik_cikolata": "#7A4A2A",
    "algida_plombir": "#E8D5A0",
    "algida_volcano": "#E4572E",
    "cornetto_beyaz_disk": "#DDE3EC",
    "cornetto_cilek": "#FF4F79",
    "cornetto_findik_krema": "#C68958",
    "cornetto_fistik": "#8BC34A",
    "cornetto_karamel": "#D89B3C",
    "cornetto_kaymak": "#F2E2C4",
    "cornetto_mango": "#FFA630",
    "cornetto_milka": "#C9A0DC",
    "cornetto_oreo": "#6B6874",
    "empty_slot": "#2E3A55",
    "frigola_bar": "#3DDC97",
    "frigola_klasik": "#00B4A0",
    "magnum_ahududu": "#E0355B",
    "magnum_badem": "#CD9B6A",
    "magnum_beyaz": "#F2EFE6",
    "magnum_cookie": "#9C6B44",
    "magnum_double_cikolata": "#5A3620",
    "magnum_dubai": "#9163C9",
    "magnum_karadut": "#7A3A8E",
    "magnum_karamel": "#B87333",
    "magnum_klasik": "#8B4513",
    "magnum_sandwich_badem": "#A9744F",
    "magnum_signature_fistik": "#6FA84B",
    "maras_usulu_kutu": "#B7C98A",
    "nogger_karamel": "#C79A5D",
    "nogger_vanilya": "#EBD9B4",
    "twister_kavun": "#B4D94C",
    "twister_ocean": "#2E86C1",
    "twister_orman": "#A02B63",
    "viennetta": "#E8C77E",
}

# ── Brand grouping (Dashboard "Supported Products" catalog) ───────────────────
_BRAND_PREFIXES: list[tuple[str, str]] = [
    ("algida_", "Algida"),
    ("cornetto_", "Cornetto"),
    ("frigola_", "Frigola"),
    ("magnum_", "Magnum"),
    ("maras_usulu", "Maraş Usulü"),
    ("nogger_", "Nogger"),
    ("twister_", "Twister"),
    ("viennetta", "Viennetta"),
]


def infer_brand(category: str) -> str:
    """Best-effort brand label from a class name, for grouping the product catalog."""
    if category == "empty_slot":
        return "Other"
    for prefix, brand in _BRAND_PREFIXES:
        if category.startswith(prefix):
            return brand
    return "Other"

COLORWAY = [ICE, MINT, AMBER, ALGIDA_RED, "#9163C9", "#00B4A0", "#FF8A65", "#81C784"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE_1,
    font=dict(color=TEXT_SECONDARY, family=SANS_FONT, size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_TERTIARY)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_TERTIARY)),
    colorway=COLORWAY,
    margin=dict(l=50, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT_PRIMARY)),
)


def format_class_name(name: str) -> str:
    """'magnum_classic' -> 'Magnum Classic'."""
    return name.replace("_", " ").strip().title()


def product_color(category: str) -> str:
    return PRODUCT_COLORS.get(category, TEXT_SECONDARY)


def fill_rate_color(fill_rate: float) -> str:
    if fill_rate > 0.70:
        return MINT
    if fill_rate >= 0.40:
        return AMBER
    return ALGIDA_RED


# ── Global CSS ────────────────────────────────────────────────────────────────
def apply_theme() -> None:
    """Inject the IceVision 'Soğuk Zincir' theme. Call once at the top of every page."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
        }

        /* === MASTER THEME === */
        html, body {
            width: 100%;
            min-height: 100vh;
            background-color: #070B14;
        }
        .stApp,
        [data-testid="stAppViewContainer"] {
            color: #EDF3FB;
            width: 100%;
            min-height: 100vh;
            /* Frost fog: a cold light leak at the top of the cabinet */
            background:
                radial-gradient(90rem 26rem at 18% -12%, rgba(92, 184, 255, 0.07), transparent 60%),
                radial-gradient(60rem 20rem at 90% -14%, rgba(255, 51, 85, 0.045), transparent 55%),
                #070B14;
        }

        /* Remove Streamlit's default header bar spacing */
        [data-testid="stHeader"] {
            background: transparent;
            height: 0;
            min-height: 0;
        }

        [data-testid="stMain"] {
            width: 100%;
            max-width: 100%;
            background: transparent;
        }

        .block-container,
        .stMainBlockContainer {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 100% !important;
            width: 100%;
        }

        /* We render our own text-only nav; hide Streamlit's auto page list */
        [data-testid="stSidebarNav"] { display: none; }

        /* Native "Connecting"/"Running" toast — replaced by our own status_line() */
        [data-testid="stStatusWidget"] { display: none !important; }

        /* Streamlit chrome (Deploy button, toolbar, hamburger) — not part of the product */
        [data-testid="stToolbar"],
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"] { display: none !important; }

        /* Keyboard focus stays visible on the dark ground */
        :focus-visible {
            outline: 2px solid #5CB8FF66 !important;
            outline-offset: 2px;
        }

        /* === SIDEBAR ===
           Only background/border/height are themed here. Width and the expand/collapse
           transform are left to Streamlit's own stylesheet — locking them with a fixed
           min-width/max-width blocks the native collapse animation. */
        [data-testid="stSidebar"] {
            background: #0C121E;
            border-right: 1px solid #1B2536;
            height: 100vh;
        }
        [data-testid="stSidebar"] > div {
            background: #0C121E;
        }
        [data-testid="stSidebarContent"] {
            background: #0C121E;
            height: 100vh;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdown"] { color: #8195B5; }

        .brand-mark {
            display: flex;
            align-items: center;
            gap: 9px;
            font-family: 'Sora', system-ui, sans-serif;
            font-size: 1.05rem;
            font-weight: 300;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: #EDF3FB;
            padding: 6px 4px 16px 4px;
        }
        .brand-heart {
            flex-shrink: 0;
            filter: drop-shadow(0 0 6px #FF335560);
        }
        .brand-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.3em;
            color: #46536E;
            text-transform: uppercase;
            padding: 0 4px 14px 4px;
            margin-top: -12px;
        }

        [data-testid="stSidebarUserContent"] [data-testid="stPageLink-NavLink"] {
            color: #8195B5 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            border-radius: 0 4px 4px 0;
            border-left: 2px solid transparent;
            padding: 8px 12px !important;
            margin-bottom: 1px;
            transition: color 0.2s ease, background 0.2s ease, border-color 0.2s ease;
        }
        [data-testid="stSidebarUserContent"] [data-testid="stPageLink-NavLink"] p {
            color: inherit !important;
            font-weight: inherit !important;
        }
        [data-testid="stSidebarUserContent"] [data-testid="stPageLink-NavLink"]:hover {
            color: #EDF3FB !important;
            background: #5CB8FF10 !important;
            border-left: 2px solid #5CB8FF55 !important;
        }
        [data-testid="stSidebarUserContent"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
            color: #EDF3FB !important;
            background: #FF335514 !important;
            border-left: 2px solid #FF3355 !important;
            border-radius: 0 4px 4px 0 !important;
        }

        .sidebar-section-label {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #46536E;
            font-weight: 500;
            margin: 18px 0 6px 2px;
        }

        .sidebar-footer {
            color: #46536E;
            font-size: 0.7rem;
            letter-spacing: 0.03em;
            padding-top: 14px;
        }

        /* === PAGE HEADER — the "heat line" signature ===
           Red eyebrow above a Sora-300 title; below, a 2px rule that starts
           Algida-red and freezes into glacier blue, then vanishes. */
        .page-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.34em;
            color: #FF3355;
            margin-bottom: 6px;
        }
        .page-title {
            font-family: 'Sora', system-ui, sans-serif;
            font-weight: 300;
            font-size: 1.9rem;
            color: #EDF3FB;
            letter-spacing: -0.01em;
            line-height: 1.15;
        }
        .ice-rule {
            height: 2px;
            margin: 14px 0 22px 0;
            background: linear-gradient(90deg,
                #FF3355 0px, #FF3355 52px,
                #5CB8FF66 140px,
                #5CB8FF22 320px,
                transparent 60%);
        }

        .section-header {
            font-weight: 500;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #8195B5;
            margin: 8px 0 12px 0;
        }

        .glow-divider {
            height: 1px;
            border: none;
            background: linear-gradient(90deg, transparent, #5CB8FF2A, transparent);
            margin: 1.5rem 0;
        }

        /* === METRIC CARDS === */
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0) 38%), #101828;
            border: 1px solid #1B2536;
            border-radius: 10px;
            padding: 1.25rem;
            transition: border-color 0.3s ease;
        }
        [data-testid="stMetric"]:hover { border-color: #5CB8FF33; }
        [data-testid="stMetricValue"] {
            font-family: 'Sora', system-ui, sans-serif !important;
            font-size: 2rem !important;
            font-weight: 300 !important;
            color: #EDF3FB !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8195B5 !important;
            font-weight: 400 !important;
        }
        [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; }

        /* === BORDERED CONTAINERS AS CARDS === */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0) 42%), #101828;
            border: 1px solid #1B2536 !important;
            border-radius: 10px;
            transition: border-color 0.3s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #5CB8FF33; }

        /* === FILE UPLOADER === */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"] {
            background: #101828 !important;
            border: 1px dashed #24314A !important;
            border-radius: 10px !important;
            transition: border-color 0.3s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover,
        [data-testid="stFileUploadDropzone"]:hover { border-color: #5CB8FF55 !important; }

        /* === BUTTONS ===
           Primary = Algida red, the only loud action on the page.
           Everything else is a quiet glacier ghost. */
        .stButton > button {
            background: transparent;
            color: #5CB8FF !important;
            border: 1px solid #5CB8FF3D;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
            transition: all 0.25s ease;
            padding: 0.5rem 1.5rem;
        }
        .stButton > button p { color: inherit !important; }
        .stButton > button:hover {
            background: #5CB8FF12;
            border-color: #5CB8FF77;
        }
        .stButton > button[kind="primary"] {
            background: #E4002B;
            color: #FFFFFF !important;
            border: none;
            font-weight: 600;
            box-shadow: 0 2px 14px #E4002B40;
        }
        .stButton > button[kind="primary"]:hover {
            background: #FF1744;
            box-shadow: 0 2px 20px #E4002B66;
        }

        .stDownloadButton > button {
            background: transparent;
            color: #5CB8FF !important;
            border: 1px solid #5CB8FF3D;
            border-radius: 6px;
        }
        .stDownloadButton > button:hover {
            background: #5CB8FF12;
            border-color: #5CB8FF77;
        }

        /* === SELECTBOX / INPUTS === */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInput"] > div > div {
            background: #101828;
            border: 1px solid #1B2536;
            border-radius: 6px;
            color: #EDF3FB;
        }

        /* === TABS — red underline marks the active view === */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: transparent;
            border-bottom: 1px solid #1B2536;
            padding: 0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #8195B5;
            border: none;
            border-radius: 0;
            font-size: 0.82rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            padding: 8px 18px;
        }
        .stTabs [aria-selected="true"] {
            background: transparent !important;
            color: #EDF3FB !important;
            border: none !important;
            box-shadow: inset 0 -2px 0 #FF3355;
        }

        /* === EXPANDER === */
        [data-testid="stExpander"] {
            background: #101828;
            border: 1px solid #1B2536;
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary { color: #EDF3FB !important; }

        /* === DATAFRAME / TABLE === */
        [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #1B2536; }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #070B14; }
        ::-webkit-scrollbar-thumb { background: #1B2536; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #2E3A55; }

        /* === PLOTLY === */
        .js-plotly-plot { border-radius: 10px; overflow: hidden; }

        /* === ALERT / INFO BOXES === */
        [data-testid="stAlert"] {
            background: #101828;
            border: 1px solid #1B2536;
            border-radius: 10px;
            color: #8195B5;
        }

        h1, h2, h3, h4 {
            color: #EDF3FB !important;
            font-family: 'Sora', system-ui, sans-serif;
            font-weight: 300;
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #1B2536, transparent);
            margin: 1.5rem 0;
        }

        /* === CUSTOM COMPONENTS === */
        .quick-action-title {
            color: #EDF3FB;
            font-family: 'Sora', system-ui, sans-serif;
            font-size: 0.98rem;
            font-weight: 400;
            margin-bottom: 4px;
        }
        .quick-action-desc { color: #8195B5; font-size: 0.82rem; margin-bottom: 10px; }
        .quick-action-link { color: #5CB8FF; font-size: 0.82rem; }

        /* === PRODUCT CATALOG (Supported Products, brand-grouped chips) === */
        .product-brand-row {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin: 16px 0 8px 0;
        }
        .product-brand-row:first-child { margin-top: 0; }
        .product-brand-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8195B5;
            font-weight: 500;
        }
        .product-brand-count {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: #46536E;
        }
        .product-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 4px;
        }
        .product-chip {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: #101828;
            border: 1px solid #1B2536;
            border-radius: 20px;
            padding: 6px 14px 6px 10px;
            font-size: 0.78rem;
            font-weight: 400;
            color: #C9D6EA;
            white-space: nowrap;
            transition: border-color 0.25s ease, color 0.25s ease;
        }
        .product-chip:hover {
            border-color: #5CB8FF44;
            color: #EDF3FB;
        }
        .product-chip-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .product-chip-empty {
            border-style: dashed;
            color: #8195B5;
        }

        .alert-bar {
            background: #FF335514;
            border-left: 3px solid #FF3355;
            border-radius: 4px;
            padding: 12px 16px;
            color: #EDF3FB;
            font-size: 0.88rem;
            margin: 10px 0 18px 0;
        }
        .alert-bar-ok {
            background: #2FE6A814;
            border-left: 3px solid #2FE6A8;
            border-radius: 4px;
            padding: 12px 16px;
            color: #EDF3FB;
            font-size: 0.88rem;
            margin: 10px 0 18px 0;
        }

        .product-row { padding: 10px 2px; border-bottom: 1px solid #131B2A; }
        .product-row-top { display: flex; align-items: center; justify-content: space-between; }
        .product-row-left { display: flex; align-items: center; gap: 10px; }
        .product-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .product-name { color: #EDF3FB; font-size: 0.9rem; }
        .product-row-right { display: flex; align-items: center; gap: 10px; }
        .product-count {
            font-family: 'Sora', system-ui, sans-serif;
            font-weight: 300;
            font-size: 1.1rem;
            color: #EDF3FB;
        }
        .conf-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #8195B5;
            background: #16202F;
            border-radius: 20px;
            padding: 2px 9px;
        }
        .mini-bar-track { background: #1B2536; border-radius: 2px; height: 2px; width: 100%; margin-top: 8px; overflow: hidden; }
        .mini-bar-fill { height: 100%; border-radius: 2px; }

        .pill {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 3px 10px;
            border-radius: 20px;
            border: 1px solid;
            white-space: nowrap;
        }

        /* === RECENT ANALYSES (Dashboard, dense paginated rows) === */
        .analyses-summary-line {
            color: #8195B5;
            font-size: 0.78rem;
            margin: 2px 0 10px 0;
        }
        .analyses-page-label {
            text-align: center;
            color: #5CB8FF;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding-top: 8px;
        }
        /* Popover trigger buttons (row delete, clear-all) aren't wrapped in .stButton,
           so they miss the theme entirely by default — style them as ghost buttons. */
        [data-testid="stPopover"] > div > button {
            background: transparent;
            color: #8195B5;
            border: 1px solid #1B2536;
            border-radius: 6px;
            transition: border-color 0.25s ease, color 0.25s ease;
        }
        [data-testid="stPopover"] > div > button:hover {
            border-color: #FF335544;
            color: #FF3355;
        }
        /* The per-row "✕" delete trigger: small square icon button, no dropdown caret,
           vertically centered against its 40px-tall row instead of stretching/stacking. */
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2):has([data-testid="stPopover"]) {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="stPopover"] > div > button {
            width: 34px;
            height: 34px;
            min-width: 34px;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="stPopover"] > div > button svg {
            display: none;
        }
        .analysis-row {
            display: flex;
            align-items: stretch;
            background: #0C121E;
            border: 1px solid #1B2536;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 6px;
            transition: border-color 0.3s ease;
        }
        .analysis-row:hover {
            border-color: #5CB8FF33;
        }
        .analysis-row-stripe {
            width: 3px;
            flex-shrink: 0;
        }
        .analysis-row-body {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 14px;
            flex: 1;
            min-width: 0;
        }
        .analysis-row-thumb {
            width: 40px;
            height: 40px;
            border-radius: 5px;
            object-fit: cover;
            flex-shrink: 0;
            border: 1px solid #1B2536;
        }
        .analysis-row-thumb-empty {
            background: #101828;
            border-style: dashed;
        }
        .analysis-row-meta {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 160px;
            flex-shrink: 0;
        }
        .analysis-row-filename {
            color: #EDF3FB;
            font-size: 0.84rem;
            font-weight: 400;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 220px;
        }
        .analysis-row-time {
            font-family: 'JetBrains Mono', monospace;
            color: #8195B5;
            font-size: 0.68rem;
            letter-spacing: 0.03em;
        }
        .analysis-row-metrics {
            display: flex;
            gap: 22px;
            margin-left: auto;
            flex-wrap: wrap;
        }
        .analysis-row-metric {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            min-width: 52px;
        }
        .analysis-row-metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            font-weight: 300;
            color: #EDF3FB;
        }
        .analysis-row-metric-label {
            font-size: 0.58rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8195B5;
            margin-top: 1px;
        }

        .shelf-cell {
            width: 34px;
            height: 24px;
            border-radius: 3px;
            flex-shrink: 0;
        }
        .shelf-cell-empty {
            background: #16202F;
            border: 1px dashed #1B2536;
        }
        .shelf-row-label {
            width: 56px;
            font-size: 0.7rem;
            color: #8195B5;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            flex-shrink: 0;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            color: #8195B5;
            margin-right: 16px;
            margin-bottom: 6px;
        }

        .detection-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #8195B5;
            margin-top: 10px;
        }

        .units-needed-value {
            font-family: 'Sora', system-ui, sans-serif;
            font-size: 2.5rem;
            font-weight: 300;
            color: #EDF3FB;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Ensure the sidebar open/collapse button reads on the dark ground */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[kind="header"] {
            color: #5CB8FF !important;
            transition: color 0.3s ease;
        }
        [data-testid="collapsedControl"]:hover,
        [data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="stSidebarCollapseButton"]:hover,
        button[kind="header"]:hover {
            color: #EDF3FB !important;
        }
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg,
        button[kind="header"] svg {
            fill: currentColor !important;
            color: currentColor !important;
        }

        @media (prefers-reduced-motion: reduce) {
            * { transition: none !important; animation: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_EYEBROW_DEFAULT = "Algida Otomat Analitiği"


def page_header(title: str, eyebrow: str = _EYEBROW_DEFAULT) -> None:
    """Red mono eyebrow, Sora display title, and the heat-line rule beneath."""
    st.markdown(
        f'<div class="page-eyebrow">{_html.escape(eyebrow)}</div>'
        f'<div class="page-title">{_html.escape(title)}</div>'
        f'<div class="ice-rule"></div>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{_html.escape(text)}</div>', unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    """Inline HTML for a colored outline pill (e.g. KRİTİK / AZALDI / YETERLİ)."""
    return (
        f'<span class="pill" style="color:{color};border-color:{color}40;'
        f'background:{color}20;">{_html.escape(text)}</span>'
    )


def alert_bar(message: str, ok: bool = False) -> None:
    cls = "alert-bar-ok" if ok else "alert-bar"
    st.markdown(f'<div class="{cls}">{_html.escape(message)}</div>', unsafe_allow_html=True)


def status_line(label: str, ok: bool = True) -> None:
    """Small monospace status readout (e.g. 'HAZIR · MODEL: V5'), in place of
    Streamlit's native connection toast, which is hidden globally in apply_theme()."""
    color = MINT if ok else ALGIDA_RED
    st.markdown(
        f'<div class="detection-meta">'
        f'<span style="color:{color};">&#9679;</span> {_html.escape(label)}'
        f"</div>",
        unsafe_allow_html=True,
    )


def fill_rate_ring(fill_rate: float, size: int = 180) -> str:
    """Hand-drawn SVG ring gauge for shelf fill rate — the one glowing element
    on the page. Gradient arc in the status color over a frost track, with
    10% tick marks so the gauge reads as an instrument, not a decoration."""
    pct = max(0.0, min(1.0, fill_rate))
    color = fill_rate_color(pct)
    r = 78
    stroke_w = 10
    cx = cy = 90
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - pct)
    pct_label = round(pct * 100)

    # 10% tick marks on a slightly larger radius than the arc
    ticks = []
    for i in range(10):
        ticks.append(
            f'<line x1="{cx}" y1="4" x2="{cx}" y2="9" stroke="#1B2536" stroke-width="2" '
            f'transform="rotate({i * 36} {cx} {cy})"/>'
        )

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:12px 0;">
        <svg width="{size}" height="{size}" viewBox="0 0 180 180"
             style="filter: drop-shadow(0 0 10px {color}40);">
            <defs>
                <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="{color}"/>
                    <stop offset="100%" stop-color="{color}88"/>
                </linearGradient>
            </defs>
            {''.join(ticks)}
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1B2536" stroke-width="{stroke_w}"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="url(#ringGrad)" stroke-width="{stroke_w}"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{offset:.2f}"
                    transform="rotate(-90 {cx} {cy})"/>
            <text x="{cx}" y="86" text-anchor="middle" font-family="'Sora',system-ui,sans-serif"
                  font-size="34" font-weight="300" fill="#EDF3FB">{pct_label}%</text>
            <text x="{cx}" y="108" text-anchor="middle" font-family="system-ui,sans-serif"
                  font-size="10" letter-spacing="2" fill="#8195B5">DOLULUK</text>
        </svg>
    </div>
    """
