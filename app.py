"""
Brain Tumor MRI Classification - Streamlit Web Application (v2)
================================================================
Upload a brain MRI scan and get the predicted tumour type, the model confidence,
the probability of every class and a Grad-CAM heatmap that shows which part of
the scan influenced the decision.

Run with:   streamlit run app.py

Files expected in the `models/` folder (created by the training notebook):
    best_model.h5       -> model used for predictions
    class_names.json    -> class order of the model outputs
    model_comparison.csv (optional) -> comparison table shown in "Model Insights"

NOTE: This tool is for learning / demonstration only. It is NOT a medical device
and must never replace the judgement of a qualified radiologist.
"""

from __future__ import annotations

# ---------- Standard library ----------
import glob
import html
import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

# ---------- Third-party ----------
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# =============================================================================
# 1. PAGE CONFIGURATION (must be the first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. CONSTANTS
# =============================================================================
APP_VERSION = "2.0"
REPO_URL = "https://github.com/13msrajput/Brain-Tumor-MRI-Classification"

# --- files and folders ---
MODELS_DIR = "models"
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.h5")
CLASS_INFO_FILE = os.path.join(MODELS_DIR, "class_names.json")
COMPARISON_FILE = os.path.join(MODELS_DIR, "model_comparison.csv")

# --- model input (must match the training notebook) ---
IMG_SIZE = (224, 224)
DEFAULT_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# --- upload limits ---
MAX_UPLOAD_MB = 10
MAX_BATCH_FILES = 25
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MIN_SIDE_PX = 64

# --- decision thresholds ---
HIGH_CONFIDENCE = 0.85  # at or above -> "High confidence"
DEFAULT_LOW_CONFIDENCE = 0.60  # below the sidebar threshold -> "Low confidence"
CLOSE_CALL_MARGIN = 0.15  # top-2 gap smaller than this -> "close call"
COLOR_WARNING_LEVEL = 0.04  # average colour difference above this -> not grayscale

# --- display information for each class ---
PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary",
}
CLASS_COLORS = {
    "glioma": "#F87171",
    "meningioma": "#60A5FA",
    "no_tumor": "#34D399",
    "pituitary": "#FBBF24",
}
CLASS_DESCRIPTIONS = {
    "glioma": "Gliomas start in the glial cells that support nerve cells in the brain or spinal cord.",
    "meningioma": "Meningiomas grow from the meninges, the membranes that cover the brain and spinal cord. Many grow slowly.",
    "no_tumor": "No tumour pattern was detected in this scan by the model.",
    "pituitary": "Pituitary tumours form in the pituitary gland at the base of the brain and can affect hormone levels.",
}
PRETTY_MODEL_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenetv2": "MobileNetV2",
    "resnet50v2": "ResNet50V2",
    "inceptionv3": "InceptionV3",
    "efficientnetb0": "EfficientNetB0",
}

# --- results of the training run (used when models/model_comparison.csv is missing) ---
FALLBACK_RESULTS = pd.DataFrame(
    [
        ["ResNet50V2", "ImageNet", 0.8984, 0.8959, 0.8618, 0.8696, 0.8576, 0.8546, 0.9898, 24.10, 206.4, 3.04, 10.2],
        ["InceptionV3", "ImageNet", 0.8825, 0.8775, 0.8415, 0.8490, 0.8418, 0.8358, 0.9848, 22.34, 129.2, 2.85, 10.6],
        ["MobileNetV2", "ImageNet", 0.8466, 0.8404, 0.8252, 0.8391, 0.8259, 0.8232, 0.9949, 2.59, 24.3, 1.21, 6.4],
        ["Custom CNN", "Random", 0.7948, 0.7618, 0.7846, 0.8101, 0.7844, 0.7530, 0.9188, 1.23, 14.2, 2.16, 11.1],
        ["EfficientNetB0", "ImageNet", 0.6773, 0.6608, 0.6992, 0.7972, 0.6786, 0.6874, 0.9949, 4.38, 31.0, 2.60, 7.9],
    ],
    columns=[
        "Model", "Weights", "Val Accuracy", "Val F1 (Macro)", "Test Accuracy",
        "Test Precision (Macro)", "Test Recall (Macro)", "Test F1 (Macro)",
        "Tumour Sensitivity", "Parameters (M)", "Size (MB)",
        "Inference (ms/image)", "Training Time (min)",
    ],
)  # fmt: skip

# =============================================================================
# 3. THEME (dark / light) AND CSS
# =============================================================================
THEMES: dict[str, dict[str, str]] = {
    "Dark": {
        "bg": "#080B16",
        "sidebar": "#0B1020",
        "surface": "rgba(255,255,255,0.045)",
        "surface-2": "rgba(255,255,255,0.075)",
        "border": "rgba(255,255,255,0.10)",
        "text": "#E8ECF6",
        "muted": "#9AA4BD",
        "accent": "#818CF8",
        "accent-2": "#22D3EE",
        "glow": "rgba(99,102,241,0.22)",
        "glow-2": "rgba(34,211,238,0.12)",
        "shadow": "0 12px 40px rgba(0,0,0,0.45)",
        "track": "rgba(255,255,255,0.10)",
        "input": "rgba(255,255,255,0.06)",
    },
    "Light": {
        "bg": "#F3F5FB",
        "sidebar": "#FFFFFF",
        "surface": "#FFFFFF",
        "surface-2": "#F1F4FB",
        "border": "#E1E6F0",
        "text": "#0F172A",
        "muted": "#586178",
        "accent": "#4F46E5",
        "accent-2": "#0891B2",
        "glow": "rgba(79,70,229,0.10)",
        "glow-2": "rgba(8,145,178,0.08)",
        "shadow": "0 10px 30px rgba(15,23,42,0.08)",
        "track": "#E3E8F2",
        "input": "#F1F4FB",
    },
}

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, button, input, textarea { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, var(--glow), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, var(--glow-2), transparent 55%),
    var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, #MainMenu, footer { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
.element-container:has(> style) { display: none; }

/* ---------- native text colours (so both themes stay readable) ---------- */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stRadio"] label p, [data-testid="stToggle"] label p,
[data-testid="stCheckbox"] label p, [data-testid="stSlider"] label p,
[data-testid="stExpander"] summary p, [data-testid="stFileUploader"] label p,
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSliderThumbValue"] { color: var(--text) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted) !important; }
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] hr { border-color: var(--border); }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] { color: var(--muted); }

/* ---------- tabs ---------- */
[role="tablist"], [data-baseweb="tab-list"] { gap: 6px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 5px; }
[data-testid="stTab"], button[data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; height: auto; background: transparent; border: none; }
[data-testid="stTab"] p, button[data-baseweb="tab"] p { color: var(--muted) !important; font-weight: 600; font-size: 0.92rem; }
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
[data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p { color: #fff !important; }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [role="tablist"] + div, [data-testid="stTabs"] > div > div:first-child { border-bottom: none; }

/* ---------- widgets ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface); border: 1.5px dashed var(--border); border-radius: 16px; padding: 1.4rem;
  transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: var(--surface-2); }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderFile"] { border-radius: 10px; }
[data-testid="stFileUploaderFileName"] { color: var(--text) !important; }
[data-testid="stFileChip"] { background: var(--surface-2) !important; border: 1px solid var(--border); border-radius: 12px; color: var(--text) !important; }
[data-testid="stFileChip"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] > div:last-child:not(:first-child) { background: transparent; }
[data-testid="stRadioOption"] > div > div > div:first-child { background: var(--input) !important; border: 1.5px solid var(--muted) !important; }
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child { background: var(--accent) !important; border-color: var(--accent) !important; }
[data-testid="stToggle"] label > div:first-child { border-color: var(--border); }
[data-testid="stRadioOption"] > div > div > div:first-child > div { background: transparent !important; }
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div { background: #fff !important; }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
[data-testid="stExpander"] summary:hover p { color: var(--accent) !important; }
[data-testid="stDownloadButton"] button, [data-testid="stButton"] button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 12px; font-weight: 600; transition: all .2s;
}
[data-testid="stDownloadButton"] button p, [data-testid="stButton"] button p { color: var(--text) !important; }
[data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover { border-color: var(--accent); transform: translateY(-1px); }
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid var(--border); }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stSpinner"] p { color: var(--muted) !important; }

/* ---------- custom components ---------- */
.hero { position: relative; overflow: hidden; border: 1px solid var(--border); border-radius: 24px; padding: 2.2rem 2.4rem;
  background: linear-gradient(135deg, var(--surface-2), var(--surface)); box-shadow: var(--shadow); margin-bottom: 1.1rem; }
.hero::after { content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px; border-radius: 50%;
  background: radial-gradient(circle, var(--glow), transparent 70%); pointer-events: none; }
.eyebrow { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
  background: var(--surface); border: 1px solid var(--border); padding: 5px 12px; border-radius: 999px; }
.hero h1 { font-size: 2.5rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.1; margin: .9rem 0 .6rem 0; color: var(--text); }
.grad { background: linear-gradient(90deg, var(--accent), var(--accent-2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 720px; margin: 0 0 1.1rem 0; line-height: 1.6; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: .8rem; font-weight: 600; color: var(--text); background: var(--surface); border: 1px solid var(--border); padding: 6px 12px; border-radius: 999px; }
.chip b { color: var(--accent); font-weight: 700; }

.notice { display: flex; gap: 12px; align-items: flex-start; border-radius: 14px; padding: 12px 16px; margin: .5rem 0 1.2rem 0;
  font-size: .88rem; line-height: 1.5; color: var(--text); border: 1px solid var(--border); background: var(--surface); }
.notice.warn { border-color: rgba(251,191,36,.45); background: rgba(251,191,36,.09); }
.notice.info { border-color: rgba(96,165,250,.40); background: rgba(96,165,250,.08); }
.notice.good { border-color: rgba(52,211,153,.40); background: rgba(52,211,153,.08); }
.notice .ico { font-size: 1.1rem; line-height: 1.4; }
.notice b { font-weight: 700; }

.section-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 1.4rem 0 .7rem 0; letter-spacing: -.01em; }
.section-title small { color: var(--muted); font-weight: 500; margin-left: 8px; font-size: .82rem; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 1.2rem 1.3rem; box-shadow: var(--shadow); }
.result { position: relative; background: linear-gradient(160deg, var(--surface-2), var(--surface)); border: 1px solid var(--border);
  border-left: 4px solid var(--c); border-radius: 18px; padding: 1.3rem 1.5rem; box-shadow: var(--shadow); }
.result .label { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.result .status { display: inline-block; margin-top: .5rem; font-size: .8rem; font-weight: 700; color: var(--c);
  background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);
  padding: 4px 12px; border-radius: 999px; }
.result .cls { font-size: 2.3rem; font-weight: 800; letter-spacing: -.03em; color: var(--text); margin: .5rem 0 .2rem 0; line-height: 1.1; }
.result .conf-row { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.result .conf { font-size: 1.5rem; font-weight: 700; color: var(--c); }
.meter { height: 9px; background: var(--track); border-radius: 999px; overflow: hidden; margin: .8rem 0 .2rem 0; }
.meter > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--c), color-mix(in srgb, var(--c) 55%, white)); }

.lvl { font-size: .76rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; border: 1px solid transparent; }
.lvl.high { color: #34D399; background: rgba(52,211,153,.12); border-color: rgba(52,211,153,.35); }
.lvl.medium { color: #FBBF24; background: rgba(251,191,36,.12); border-color: rgba(251,191,36,.35); }
.lvl.low { color: #F87171; background: rgba(248,113,113,.12); border-color: rgba(248,113,113,.35); }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 12px 0; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: .85rem 1rem; }
.kpi .l { font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.kpi .v { font-size: 1.35rem; font-weight: 700; color: var(--text); margin-top: 4px; letter-spacing: -.02em; }
.kpi .s { font-size: .76rem; color: var(--muted); margin-top: 2px; }

.prow { display: grid; grid-template-columns: 130px 1fr 64px; align-items: center; gap: 12px; padding: 7px 0; }
.prow .name { display: flex; align-items: center; gap: 8px; font-size: .9rem; font-weight: 500; color: var(--muted); }
.prow.top .name { color: var(--text); font-weight: 700; }
.prow .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.prow .track { height: 10px; background: var(--track); border-radius: 999px; overflow: hidden; }
.prow .fill { height: 100%; border-radius: 999px; }
.prow .val { text-align: right; font-size: .9rem; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }
.prow.top .val { color: var(--text); }

.pill { display: inline-block; font-size: .78rem; font-weight: 700; color: var(--c); padding: 3px 11px; border-radius: 999px;
  background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent); white-space: nowrap; }
.tag { display: inline-block; font-size: .68rem; font-weight: 800; letter-spacing: .08em; color: #fff; padding: 2px 8px; border-radius: 6px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); margin-left: 8px; vertical-align: middle; }

.tbl-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 16px; background: var(--surface); }
table.tbl { width: 100%; border-collapse: collapse; font-size: .86rem; }
table.tbl th { text-align: left; padding: 11px 14px; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  background: var(--surface-2); border-bottom: 1px solid var(--border); white-space: nowrap; }
table.tbl td { padding: 10px 14px; color: var(--text); border-bottom: 1px solid var(--border); white-space: nowrap; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl td.num { font-variant-numeric: tabular-nums; text-align: right; }
table.tbl th.num { text-align: right; }
table.tbl td.best { color: var(--accent-2); font-weight: 800; }
table.tbl tr.active td { background: color-mix(in srgb, var(--accent) 10%, transparent); }

.mini { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
.mini .card h4 { margin: 0 0 6px 0; font-size: .98rem; font-weight: 700; color: var(--text); }
.mini .card p { margin: 0; font-size: .86rem; color: var(--muted); line-height: 1.55; }
.step-n { display: inline-flex; width: 26px; height: 26px; align-items: center; justify-content: center; border-radius: 8px; margin-bottom: 10px;
  font-size: .8rem; font-weight: 800; color: #fff; background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
.classcard { border-top: 3px solid var(--c); }
.footer { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border); }
.footer a { color: var(--accent); text-decoration: none; font-weight: 600; }
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: .4rem; }
.brand .logo { width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
.brand .t { font-weight: 800; font-size: 1.02rem; color: var(--text); line-height: 1.1; }
.brand .s { font-size: .74rem; color: var(--muted); }
.side-label { font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 1rem 0 .3rem 0; }
@media (max-width: 720px) { .hero { padding: 1.4rem; } .hero h1 { font-size: 1.8rem; } .prow { grid-template-columns: 100px 1fr 54px; } }
"""


def inject_css(theme: str) -> None:
    """Write the CSS variables of the chosen theme plus the shared styles into the page."""
    palette = THEMES[theme]
    variables = ";".join(f"--{key}:{value}" for key, value in palette.items())
    st.markdown(
        f"<style>:root{{{variables}}}{STATIC_CSS}</style>", unsafe_allow_html=True
    )


# =============================================================================
# 4. SMALL HTML HELPERS
# =============================================================================
def render(markup: str) -> None:
    """Render raw HTML. Lines are joined so Markdown never mistakes indented HTML for code."""
    compact = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def pretty(class_name: str) -> str:
    """Human-readable class name."""
    return PRETTY_NAMES.get(class_name, class_name.replace("_", " ").title())


def color_of(class_name: str) -> str:
    """Colour assigned to a class."""
    return CLASS_COLORS.get(class_name, "#818CF8")


def pill(class_name: str) -> str:
    """Small coloured badge with the class name."""
    return f'<span class="pill" style="--c:{color_of(class_name)}">{html.escape(pretty(class_name))}</span>'


def section(title: str, hint: str = "") -> None:
    """Section heading with an optional small grey hint."""
    small = f"<small>{html.escape(hint)}</small>" if hint else ""
    render(f'<div class="section-title">{html.escape(title)}{small}</div>')


def notice(text: str, kind: str = "info", icon: str = "ℹ️") -> None:
    """Coloured information box. `text` may contain simple HTML such as <b>."""
    render(
        f'<div class="notice {kind}"><span class="ico">{icon}</span><span>{text}</span></div>'
    )


def kpi_grid(items: list[tuple[str, str, str]]) -> None:
    """Row of KPI cards. Each item is (label, value, small_text)."""
    cards = "".join(
        f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>'
        + (f'<div class="s">{html.escape(sub)}</div>' if sub else "")
        + "</div>"
        for label, value, sub in items
    )
    render(f'<div class="kpis">{cards}</div>')


def html_table(
    frame: pd.DataFrame,
    raw_columns: tuple[str, ...] = (),
    number_format: dict[str, str] | None = None,
    best_columns: tuple[str, ...] = (),
    active_column: str | None = None,
    active_value: str | None = None,
) -> str:
    """Convert a DataFrame into a themed HTML table (fully controlled by our CSS)."""
    number_format = number_format or {}
    best_values = {col: frame[col].max() for col in best_columns if col in frame}
    head = "".join(
        f'<th class="{"num" if pd.api.types.is_numeric_dtype(frame[col]) else ""}">{html.escape(str(col))}</th>'
        for col in frame.columns
    )
    rows = []
    for _, row in frame.iterrows():
        is_active = (
            active_column is not None and str(row[active_column]) == active_value
        )
        cells = []
        for col in frame.columns:
            value = row[col]
            numeric = pd.api.types.is_numeric_dtype(frame[col])
            css = "num" if numeric else ""
            if col in best_values and value == best_values[col]:
                css += " best"
            if col in raw_columns:
                text = str(value)  # already safe HTML built by this app
            elif numeric and col in number_format:
                text = format(value, number_format[col])
            else:
                text = html.escape(str(value))
            if is_active and col == active_column:
                text += '<span class="tag">ACTIVE</span>'
            cells.append(f'<td class="{css.strip()}">{text}</td>')
        rows.append(
            f'<tr class="{"active" if is_active else ""}">{"".join(cells)}</tr>'
        )
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def style_chart(
    chart: alt.Chart | alt.LayerChart, theme: str
) -> alt.Chart | alt.LayerChart:
    """Apply the active theme to an Altair chart (transparent background, readable axes)."""
    palette = THEMES[theme]
    text = palette["text"]
    muted = palette["muted"]
    grid = "rgba(255,255,255,0.08)" if theme == "Dark" else "#E1E6F0"
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=muted,
            titleColor=muted,
            gridColor=grid,
            domainColor=grid,
            tickColor=grid,
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(labelColor=text, titleColor=muted)
    )


# =============================================================================
# 5. MODEL, PREPROCESSING, PREDICTION AND GRAD-CAM
# =============================================================================
def load_class_info() -> dict[str, Any]:
    """Read class_names.json (class order + extra info). Fall back to defaults if it is unusable."""
    try:
        with open(CLASS_INFO_FILE, "r", encoding="utf-8") as file:
            info = json.load(file)
        if isinstance(info.get("class_names"), list) and info["class_names"]:
            return info
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"class_names": DEFAULT_CLASSES, "_default": True}


@st.cache_resource(show_spinner="Loading the model ...")
def load_model_cached(model_path: str):
    """Load a Keras model once and keep it in memory so the app stays fast."""
    from tensorflow import (
        keras,
    )  # imported here so a missing TensorFlow gives a friendly error

    # compile=False because we only need predictions, not training
    return keras.models.load_model(model_path, compile=False)


def model_display_name(model, fallback: str = "Model") -> str:
    """Readable architecture name taken from the loaded model itself."""
    raw = str(getattr(model, "name", "") or fallback)
    return PRETTY_MODEL_NAMES.get(raw.lower(), raw)


def open_rgb_image(image_bytes: bytes) -> Image.Image:
    """Open uploaded bytes as an RGB image (MRI files are often grayscale)."""
    image = Image.open(io.BytesIO(image_bytes))
    image.load()  # force full decoding so broken files fail here
    return image.convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    """RGB image -> array (1, 224, 224, 3) with pixels in 0-1, identical to the training pipeline."""
    import tensorflow as tf

    array = np.asarray(image, dtype=np.uint8)
    # same resize as training: bilinear, then rounded to whole pixel values
    resized = tf.image.resize(array, IMG_SIZE)
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def grad_cam(model, batch: np.ndarray, class_index: int) -> np.ndarray | None:
    """
    Grad-CAM heatmap (values 0-1) for one class.

    The layers are run one by one; the feature maps just before Global Average
    Pooling are watched, and the gradient of the class score with respect to them
    tells us which regions pushed the prediction. Returns None if it cannot be computed.
    """
    import tensorflow as tf
    from tensorflow import keras

    try:
        x = tf.convert_to_tensor(batch)
        features = None
        with tf.GradientTape() as tape:
            for layer in model.layers:
                if isinstance(layer, keras.layers.InputLayer):
                    continue
                if (
                    isinstance(layer, keras.layers.GlobalAveragePooling2D)
                    and features is None
                ):
                    features = x
                    tape.watch(features)
                x = layer(x, training=False)
            score = x[:, class_index]
        if features is None:
            return None
        grads = tape.gradient(score, features)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0].numpy()
        peak = float(cam.max())
        return cam / peak if peak > 1e-8 else np.zeros_like(cam)
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def analyze_image(
    image_bytes: bytes, model_path: str, with_cam: bool
) -> dict[str, Any]:
    """Preprocess, predict (timed) and optionally compute Grad-CAM. Cached per image + model."""
    model = load_model_cached(model_path)
    image = open_rgb_image(image_bytes)
    batch = preprocess(image)

    start = time.perf_counter()
    probabilities = model.predict(batch, verbose=0)[0].astype(float)
    elapsed_ms = (time.perf_counter() - start) * 1000

    cam = grad_cam(model, batch, int(np.argmax(probabilities))) if with_cam else None
    return {"probabilities": probabilities.tolist(), "ms": elapsed_ms, "cam": cam}


def heatmap_images(
    image: Image.Image, cam: np.ndarray, opacity: float, size: int = 448
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Return (model view, heatmap, overlay) images built from a Grad-CAM map."""
    from matplotlib import colormaps

    base = image.resize((size, size))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    mask = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = (colormaps["inferno"](mask)[..., :3] * 255).astype(np.float32)
    base_arr = np.asarray(base, dtype=np.float32)
    strong = ((mask - 0.25) / 0.75).clip(0, 1)  # drop the weakest 25% of activations
    weight = (strong * opacity * 1.4).clip(0, 1)[
        ..., None
    ]  # strong regions get more colour
    overlay = base_arr * (1 - weight) + heat * weight
    return (
        base,
        Image.fromarray(heat.astype(np.uint8)),
        Image.fromarray(overlay.astype(np.uint8)),
    )


def image_checks(image: Image.Image, file_size: int) -> list[tuple[str, str, bool]]:
    """Simple quality checks. Each item is (title, detail, passed)."""
    small = np.asarray(image.resize((64, 64)), dtype=np.float32)
    colour_level = float(
        (
            np.abs(small[..., 0] - small[..., 1])
            + np.abs(small[..., 1] - small[..., 2])
        ).mean()
        / 2
        / 255
    )
    width, height = image.size
    return [
        ("Resolution", f"{width} x {height} px", min(width, height) >= MIN_SIDE_PX),
        (
            "Grayscale scan",
            "yes" if colour_level <= COLOR_WARNING_LEVEL else "colour image",
            colour_level <= COLOR_WARNING_LEVEL,
        ),
        (
            "File size",
            f"{file_size / 1024:.0f} KB",
            file_size <= MAX_UPLOAD_MB * 1024 * 1024,
        ),
    ]


def confidence_level(confidence: float, low_threshold: float) -> tuple[str, str]:
    """Return (css class, label) for a confidence value."""
    if confidence >= HIGH_CONFIDENCE:
        return "high", "High confidence"
    if confidence >= low_threshold:
        return "medium", "Medium confidence"
    return "low", "Low confidence"


def summarize(probabilities: list[float], class_names: list[str]) -> dict[str, Any]:
    """Top-1 / top-2 information used by the result card and the reports."""
    order = np.argsort(probabilities)[::-1]
    top, second = int(order[0]), int(order[1])
    return {
        "class": class_names[top],
        "confidence": float(probabilities[top]),
        "runner_up": class_names[second],
        "runner_up_conf": float(probabilities[second]),
        "margin": float(probabilities[top] - probabilities[second]),
    }


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes (for download buttons)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =============================================================================
# 6. SESSION STATE (history)
# =============================================================================
def log_history(
    file_name: str, summary: dict[str, Any], model_name: str, token: str
) -> None:
    """Add one analysis to the session history (only once per image + model)."""
    history = st.session_state.setdefault("history", [])
    if st.session_state.get("last_token") == token:
        return
    st.session_state["last_token"] = token
    history.insert(
        0,
        {
            "Time (UTC)": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "File": file_name,
            "Prediction": summary["class"],
            "Confidence": round(summary["confidence"] * 100, 1),
            "Model": model_name,
        },
    )
    del history[20:]  # keep the latest 20 only


# =============================================================================
# 7. START OF THE PAGE - theme, sidebar, model loading
# =============================================================================
active_theme = st.session_state.get("theme", "Dark")
inject_css(active_theme)

model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5")))
if not model_files:
    st.error(
        "No trained model found in the `models/` folder. Run the notebook first to train and save the models."
    )
    st.stop()
if BEST_MODEL_FILE in model_files:  # best model first
    model_files.remove(BEST_MODEL_FILE)
    model_files.insert(0, BEST_MODEL_FILE)

with st.sidebar:
    render(
        '<div class="brand"><div class="logo">🧠</div><div><div class="t">NeuroScan AI</div>'
        f'<div class="s">MRI Classifier · v{APP_VERSION}</div></div></div>'
    )

    render('<div class="side-label">Appearance</div>')
    st.radio(
        "Theme",
        list(THEMES),
        key="theme",
        horizontal=True,
        label_visibility="collapsed",
    )

    render('<div class="side-label">Model</div>')
    if len(model_files) > 1:
        selected_model_path = st.radio(
            "Active model",
            model_files,
            format_func=lambda path: (
                "⭐ Best model (recommended)"
                if path == BEST_MODEL_FILE
                else os.path.basename(path).replace(".h5", "")
            ),
            label_visibility="collapsed",
        )
    else:
        selected_model_path = model_files[0]
        st.caption("Using the deployed model.")

    render('<div class="side-label">Analysis settings</div>')
    low_threshold = st.slider(
        "Low-confidence warning below", 0.30, 0.95, DEFAULT_LOW_CONFIDENCE, 0.05
    )
    show_cam = st.toggle(
        "Explain with Grad-CAM heatmap",
        value=True,
        help="Highlights the regions that influenced the prediction.",
    )
    overlay_opacity = st.slider(
        "Heatmap strength", 0.2, 1.0, 0.6, 0.05, disabled=not show_cam
    )

    st.divider()
    st.caption("For education and demonstration only. Not a medical device.")

# ---------- load class names and the model (with friendly errors) ----------
class_info = load_class_info()
class_names: list[str] = class_info["class_names"]

try:
    model = load_model_cached(selected_model_path)
except ImportError:
    st.error(
        "TensorFlow is not installed. Run `pip install tensorflow` and restart the app."
    )
    st.stop()
except Exception as error:  # a broken file must not crash the whole app
    st.error(f"The model file could not be loaded: {error}")
    st.stop()

if model.output_shape[-1] != len(class_names):
    st.error(
        f"The model has {model.output_shape[-1]} outputs but {len(class_names)} class names were found. "
        "Make sure `class_names.json` comes from the same training run."
    )
    st.stop()

model_name = model_display_name(model)
n_params = model.count_params()

# =============================================================================
# 8. HERO + DISCLAIMER
# =============================================================================
render(f"""
    <div class="hero">
      <span class="eyebrow">AI · Medical Imaging</span>
      <h1>Brain Tumor <span class="grad">MRI Classifier</span></h1>
      <p>Upload a brain MRI scan and get an instant classification into Glioma, Meningioma, Pituitary tumour or No Tumor,
      with per-class probabilities and a Grad-CAM heatmap that shows where the model looked.</p>
      <div class="chips">
        <span class="chip">Model <b>{html.escape(model_name)}</b></span>
        <span class="chip">Classes <b>{len(class_names)}</b></span>
        <span class="chip">Input <b>{IMG_SIZE[0]}×{IMG_SIZE[1]}</b></span>
        <span class="chip">Parameters <b>{n_params / 1e6:.1f}M</b></span>
      </div>
    </div>
    """)
notice(
    "<b>For education and demonstration only.</b> This is not a medical device. Never use it for diagnosis or "
    "treatment decisions. Always consult a qualified radiologist or doctor.",
    kind="warn",
    icon="⚠️",
)

tab_single, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔬  Analyze", "🗂️  Batch", "📊  Model Insights", "ℹ️  About"]
)

# =============================================================================
# 9. TAB 1 - SINGLE IMAGE ANALYSIS
# =============================================================================
with tab_single:
    uploaded = st.file_uploader(
        "Upload a brain MRI image (JPG, JPEG or PNG)",
        type=ALLOWED_TYPES,
        key="single_upload",
    )

    if uploaded is None:
        notice(
            "Upload an MRI image to get a prediction. Your image is processed in memory and is not stored.",
            kind="info",
            icon="👆",
        )
    elif uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image."
        )
    else:
        image_bytes = uploaded.getvalue()
        try:
            image = open_rgb_image(image_bytes)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error(
                "This file could not be read as an image. Please upload a valid JPG or PNG file."
            )
            st.stop()

        try:
            with st.spinner("Analysing the scan ..."):
                result = analyze_image(image_bytes, selected_model_path, show_cam)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        probabilities: list[float] = result["probabilities"]
        summary = summarize(probabilities, class_names)
        top_class, confidence = summary["class"], summary["confidence"]
        level_css, level_label = confidence_level(confidence, low_threshold)
        has_tumour_pattern = top_class != "no_tumor"
        log_history(
            uploaded.name,
            summary,
            model_name,
            f"{uploaded.name}-{uploaded.size}-{selected_model_path}",
        )

        left, right = st.columns([1, 1.15], gap="large")

        # ----- left column: image viewer -----
        with left:
            section("Scan", uploaded.name)
            cam = result.get("cam")
            if show_cam and cam is not None:
                view = st.radio(
                    "View",
                    ["Original", "Heatmap", "Overlay"],
                    index=2,
                    horizontal=True,
                    label_visibility="collapsed",
                )
                model_view, heat_img, overlay_img = heatmap_images(
                    image, cam, overlay_opacity
                )
                st.image(
                    {"Original": image, "Heatmap": heat_img, "Overlay": overlay_img}[
                        view
                    ],
                    width="stretch",
                )
                st.caption(
                    "Grad-CAM highlights regions that influenced the prediction. It is not a tumour outline."
                )
            else:
                st.image(image, width="stretch")
                if show_cam:
                    st.caption("A heatmap could not be created for this model.")

        # ----- right column: result card -----
        with right:
            section("Prediction")
            colour = color_of(top_class)
            status = (
                "Tumour pattern detected"
                if has_tumour_pattern
                else "No tumour pattern detected"
            )
            render(f"""
                <div class="result" style="--c:{colour}">
                  <div class="label">Predicted class</div>
                  <span class="status">{status}</span>
                  <div class="cls">{html.escape(pretty(top_class))}</div>
                  <div class="conf-row"><span class="conf">{confidence * 100:.1f}%</span>
                  <span class="lvl {level_css}">{level_label}</span></div>
                  <div class="meter"><div style="width:{confidence * 100:.1f}%"></div></div>
                </div>
                """)
            kpi_grid(
                [
                    (
                        "Runner-up",
                        pretty(summary["runner_up"]),
                        f"{summary['runner_up_conf'] * 100:.1f}%",
                    ),
                    (
                        "Top-2 margin",
                        f"{summary['margin'] * 100:.1f} pts",
                        "gap to runner-up",
                    ),
                    ("Inference", f"{result['ms']:.0f} ms", model_name),
                ]
            )

            if confidence < low_threshold:
                notice(
                    f"The model is not very confident (below {low_threshold * 100:.0f}%). Treat this prediction with extra caution.",
                    kind="warn",
                    icon="⚠️",
                )
            elif summary["margin"] < CLOSE_CALL_MARGIN:
                notice(
                    f"Close call: <b>{html.escape(pretty(summary['runner_up']))}</b> is also likely "
                    f"({summary['runner_up_conf'] * 100:.1f}%). Review this scan carefully.",
                    kind="info",
                    icon="🔎",
                )

        # ----- quality checks -----
        checks = image_checks(image, uploaded.size)
        chips = "".join(
            f'<span class="chip">{"✅" if ok else "⚠️"} {html.escape(title)}: <b>{html.escape(detail)}</b></span>'
            for title, detail, ok in checks
        )
        render(f'<div class="chips" style="margin-top:6px">{chips}</div>')
        if not checks[1][2]:
            notice(
                "This looks like a colour image. The model was trained on grayscale MRI scans, so the result may be unreliable.",
                kind="warn",
                icon="🎨",
            )
        if not checks[0][2]:
            notice(
                f"The image is very small (under {MIN_SIDE_PX}px). Details are lost when it is enlarged, so the result may be unreliable.",
                kind="warn",
                icon="📏",
            )

        # ----- probabilities -----
        section("Probability of every class", "sums to 100%")
        order = np.argsort(probabilities)[::-1]
        rows = ""
        for rank, index in enumerate(order):
            name = class_names[index]
            pct = probabilities[index] * 100
            c = color_of(name)
            rows += (
                f'<div class="prow {"top" if rank == 0 else ""}"><div class="name"><span class="dot" style="background:{c}"></span>'
                f'{html.escape(pretty(name))}</div><div class="track"><div class="fill" style="width:{pct:.2f}%;'
                f'background:linear-gradient(90deg,{c},{c}AA)"></div></div><div class="val">{pct:.1f}%</div></div>'
            )
        render(f'<div class="card">{rows}</div>')

        with st.expander(f"About {pretty(top_class)}"):
            st.write(CLASS_DESCRIPTIONS.get(top_class, ""))
            st.caption("General educational information only, not medical advice.")

        # ----- downloads -----
        section("Export")
        report = {
            "file": uploaded.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model_name,
            "predicted_class": top_class,
            "confidence": round(confidence, 4),
            "confidence_level": level_label,
            "probabilities": {
                name: round(p, 4) for name, p in zip(class_names, probabilities)
            },
            "inference_ms": round(result["ms"], 1),
            "disclaimer": "Educational demonstration only. Not a medical device.",
        }
        col_a, col_b, _ = st.columns([1, 1, 2])
        col_a.download_button(
            "⬇️ Report (JSON)",
            json.dumps(report, indent=2),
            f"{os.path.splitext(uploaded.name)[0]}_report.json",
            "application/json",
            on_click="ignore",
        )
        if show_cam and cam is not None:
            col_b.download_button(
                "⬇️ Heatmap (PNG)",
                image_to_png_bytes(overlay_img),
                f"{os.path.splitext(uploaded.name)[0]}_gradcam.png",
                "image/png",
                on_click="ignore",
            )

    # ----- session history -----
    history = st.session_state.get("history", [])
    if history:
        section("Session history", f"last {len(history)} analyses")
        frame = pd.DataFrame(history)
        frame["Prediction"] = frame["Prediction"].map(pill)
        frame["Confidence"] = frame["Confidence"].map(lambda v: f"{v:.1f}%")
        render(html_table(frame, raw_columns=("Prediction",)))
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.session_state.pop("last_token", None)
            st.rerun()

# =============================================================================
# 10. TAB 2 - BATCH ANALYSIS
# =============================================================================
with tab_batch:
    section("Batch analysis", f"up to {MAX_BATCH_FILES} images")
    files = st.file_uploader(
        "Upload several MRI images",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="batch_upload",
    )

    if not files:
        notice(
            "Upload multiple scans to classify them all at once and download the results as a CSV file.",
            kind="info",
            icon="🗂️",
        )
    else:
        if len(files) > MAX_BATCH_FILES:
            st.warning(f"Only the first {MAX_BATCH_FILES} images are processed.")
            files = files[:MAX_BATCH_FILES]

        records, skipped = [], []
        progress = st.progress(0.0, text="Analysing images ...")
        for position, file in enumerate(files, start=1):
            try:
                if file.size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"larger than {MAX_UPLOAD_MB} MB")
                outcome = analyze_image(file.getvalue(), selected_model_path, False)
                info = summarize(outcome["probabilities"], class_names)
                row = {
                    "File": file.name,
                    "Prediction": info["class"],
                    "Confidence": info["confidence"] * 100,
                    "Runner-up": info["runner_up"],
                    "Margin": info["margin"] * 100,
                    "Flag": (
                        "Low confidence"
                        if info["confidence"] < low_threshold
                        else (
                            "Close call" if info["margin"] < CLOSE_CALL_MARGIN else "OK"
                        )
                    ),
                }
                for name, p in zip(class_names, outcome["probabilities"]):
                    row[f"P({pretty(name)}) %"] = round(p * 100, 2)
                records.append(row)
            except Exception as error:
                skipped.append((file.name, str(error)))
            progress.progress(
                position / len(files),
                text=f"Analysing images ... {position}/{len(files)}",
            )
        progress.empty()

        if records:
            results_df = pd.DataFrame(records)
            tumour_count = int((results_df["Prediction"] != "no_tumor").sum())
            kpi_grid(
                [
                    (
                        "Images analysed",
                        str(len(results_df)),
                        f"{len(skipped)} skipped" if skipped else "all readable",
                    ),
                    (
                        "Tumour pattern",
                        str(tumour_count),
                        f"{tumour_count / len(results_df) * 100:.0f}% of images",
                    ),
                    ("No tumour pattern", str(len(results_df) - tumour_count), ""),
                    (
                        "Average confidence",
                        f"{results_df['Confidence'].mean():.1f}%",
                        "",
                    ),
                    (
                        "Needs review",
                        str(int((results_df["Flag"] != "OK").sum())),
                        "low confidence or close call",
                    ),
                ]
            )

            counts = (
                results_df["Prediction"]
                .value_counts()
                .reindex(class_names, fill_value=0)
                .reset_index()
            )
            counts.columns = ["class", "count"]
            counts["Class"] = counts["class"].map(pretty)
            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=6, size=26)
                .encode(
                    y=alt.Y(
                        "Class:N", sort=[pretty(c) for c in class_names], title=None
                    ),
                    x=alt.X("count:Q", title="Images", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color(
                        "Class:N",
                        scale=alt.Scale(
                            domain=[pretty(c) for c in class_names],
                            range=[color_of(c) for c in class_names],
                        ),
                        legend=None,
                    ),
                    tooltip=["Class", "count"],
                )
                .properties(height=170)
            )
            section("Predicted class distribution")
            st.altair_chart(
                style_chart(chart, active_theme), width="stretch", theme=None
            )

            section("Results")
            display = results_df[
                ["File", "Prediction", "Confidence", "Runner-up", "Margin", "Flag"]
            ].copy()
            display["Prediction"] = display["Prediction"].map(pill)
            display["Runner-up"] = display["Runner-up"].map(pill)
            display = display.rename(
                columns={"Confidence": "Confidence (%)", "Margin": "Margin (pts)"}
            )
            render(
                html_table(
                    display,
                    raw_columns=("Prediction", "Runner-up"),
                    number_format={"Confidence (%)": ".1f", "Margin (pts)": ".1f"},
                )
            )

            export = results_df.copy()
            export["Prediction"] = export["Prediction"].map(pretty)
            export["Runner-up"] = export["Runner-up"].map(pretty)
            st.download_button(
                "⬇️ Download results (CSV)",
                export.to_csv(index=False),
                "batch_predictions.csv",
                "text/csv",
                on_click="ignore",
            )

        if skipped:
            with st.expander(f"{len(skipped)} file(s) could not be processed"):
                for name, reason in skipped:
                    st.write(f"**{name}** - {reason}")

# =============================================================================
# 11. TAB 3 - MODEL INSIGHTS
# =============================================================================
with tab_insights:
    section("Model card", "the model currently running in this app")
    model_size_mb = os.path.getsize(selected_model_path) / (1024 * 1024)
    kpi_grid(
        [
            ("Architecture", model_name, ""),
            ("Parameters", f"{n_params / 1e6:.2f} M", ""),
            (
                "File size",
                f"{model_size_mb:.1f} MB",
                os.path.basename(selected_model_path),
            ),
            ("Input", f"{IMG_SIZE[0]}×{IMG_SIZE[1]}×3", "pixels scaled to 0-1"),
        ]
    )

    # comparison table: use the CSV from the notebook if it exists, otherwise the built-in results
    results_source = "built-in results of the training run"
    comparison = FALLBACK_RESULTS.copy()
    if os.path.exists(COMPARISON_FILE):
        try:
            comparison = pd.read_csv(COMPARISON_FILE)
            results_source = "models/model_comparison.csv"
        except Exception:
            pass

    section("Model comparison", f"source: {results_source}")
    show_cols = [
        c
        for c in [
            "Model",
            "Weights",
            "Val Accuracy",
            "Test Accuracy",
            "Test Precision (Macro)",
            "Test Recall (Macro)",
            "Test F1 (Macro)",
            "Tumour Sensitivity",
            "Size (MB)",
            "Inference (ms/image)",
        ]
        if c in comparison
    ]
    table_df = comparison[show_cols].copy()
    formats = {
        c: ".4f"
        for c in show_cols
        if c not in ("Model", "Weights", "Size (MB)", "Inference (ms/image)")
    }
    formats.update({"Size (MB)": ".1f", "Inference (ms/image)": ".2f"})
    render(
        html_table(
            table_df,
            number_format=formats,
            best_columns=tuple(
                c
                for c in show_cols
                if c in formats and c not in ("Size (MB)", "Inference (ms/image)")
            ),
            active_column="Model",
            active_value=model_name,
        )
    )
    st.caption(
        "Highlighted values are the best in each column. Metrics are macro-averaged on the 246-image test set."
    )

    metric_cols = [
        c
        for c in ["Test Accuracy", "Test F1 (Macro)", "Tumour Sensitivity"]
        if c in comparison
    ]
    if "Model" in comparison and metric_cols:
        long = comparison.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score",
        )
        grouped = (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=list(comparison["Model"]),
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                ),
                xOffset="Metric:N",
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(range=["#818CF8", "#22D3EE", "#34D399"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
            )
            .properties(height=300)
        )
        section("Performance at a glance")
        st.altair_chart(style_chart(grouped, active_theme), width="stretch", theme=None)

    section("How to read the metrics")
    render("""
        <div class="mini">
          <div class="card"><h4>Accuracy</h4><p>Share of all scans classified correctly. Easy to read, but it can hide weak classes.</p></div>
          <div class="card"><h4>Precision &amp; Recall</h4><p>Precision: when the model names a class, how often it is right. Recall: how many real cases of a class it finds.</p></div>
          <div class="card"><h4>F1-score (macro)</h4><p>Balance of precision and recall, averaged over the four classes so small classes count equally.</p></div>
          <div class="card"><h4>Tumour sensitivity</h4><p>Of the scans with a real tumour, the share not labelled "No Tumor". Missing a tumour is the most serious mistake.</p></div>
        </div>
        """)
    section("Known limitations")
    notice(
        "About 19% of validation and 23% of test images share a source-scan ID with a training image, so scores are probably "
        "optimistic for brand-new patients. The test set is small (246 images), and class differences in brightness and scan style "
        "mean a model may partly learn the scan appearance. The model has not been validated on other hospitals or scanners.",
        kind="info",
        icon="🧪",
    )

# =============================================================================
# 12. TAB 4 - ABOUT
# =============================================================================
with tab_about:
    section("How it works")
    render("""
        <div class="mini">
          <div class="card"><div class="step-n">1</div><h4>Upload &amp; validate</h4><p>The file is checked for type, size and readability, and the scan is converted to RGB.</p></div>
          <div class="card"><div class="step-n">2</div><h4>Preprocess</h4><p>The image is resized to 224 × 224 and pixel values are scaled to 0-1, exactly as during training.</p></div>
          <div class="card"><div class="step-n">3</div><h4>Classify</h4><p>The deep learning model outputs one probability for each of the four classes.</p></div>
          <div class="card"><div class="step-n">4</div><h4>Explain</h4><p>Grad-CAM turns the model's internal feature maps into a heatmap of the most influential regions.</p></div>
        </div>
        """)

    section("The four classes")
    cards = "".join(
        f'<div class="card classcard" style="--c:{color_of(name)}"><h4>{html.escape(pretty(name))}</h4><p>{html.escape(CLASS_DESCRIPTIONS.get(name, ""))}</p></div>'
        for name in class_names
    )
    render(f'<div class="mini">{cards}</div>')

    section("Responsible use")
    notice(
        "This app is a student project built to demonstrate deep learning on medical images. Predictions can be wrong, "
        "especially for scans unlike the training data. It must not be used to make medical decisions.",
        kind="warn",
        icon="🩺",
    )

    section("Tech stack")
    render(
        '<div class="chips">'
        '<span class="chip">TensorFlow / Keras</span><span class="chip">Transfer learning</span><span class="chip">Grad-CAM</span>'
        '<span class="chip">Streamlit</span><span class="chip">Altair</span><span class="chip">Pillow · NumPy · pandas</span></div>'
    )

render(
    f'<div class="footer">Built by Mohit Singh Rajput · <a href="{REPO_URL}" target="_blank">View on GitHub</a>'
    f" · v{APP_VERSION} · Educational use only</div>"
)
