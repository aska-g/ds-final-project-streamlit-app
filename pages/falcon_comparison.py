"""Falcon Perception vs YOLO — comparison gallery + detection table."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import base64

import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(
    """
<style>
/* Full-bleed hero — breaks out of Streamlit's centered .block-container so the
   comparison render truly spans the viewport edge-to-edge. */
.falcon-hero {
  width: 100vw;
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  background: #FFFFFF;
  border-top: 1px solid #C4C6C0;
  border-bottom: 1px solid #C4C6C0;
}
.falcon-hero img { width: 100%; height: auto; display: block; }
.falcon-caption {
  text-align: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #71736D;
  padding: 8px 16px 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(vh.header_html("FALCON PERCEPTION"), unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent.parent
CANDIDATES = [APP_DIR / "falcon" / "outputs" / "comparison", APP_DIR / "falcon" / "output" / "comparison"]
COMP_DIR = next((p for p in CANDIDATES if p.is_dir()), CANDIDATES[0])
YOLO_DIR = APP_DIR / "falcon" / "outputs" / "yolov26spv"

# ---- Explainer ----
st.markdown(
    '<div class="hv-h1" style="font-size:28px;margin:6px 0 8px">What is Falcon Perception</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<div style="font-size:13.5px;color:#4A4B47;line-height:1.6;margin-bottom:14px">
<b style="color:#141414">Falcon Perception</b> is a 0.6B-parameter Small Language Model (SLM) that exists in the gap
between machine-learning models like YOLO and huge LLMs like Gemini, ChatGPT and Claude. It's somewhat more accurate
than YOLO, and slightly bigger than YOLO. You input an object name like
<span class="hv-mono" style="background:#FFFFFF;border:1px solid #C4C6C0;padding:1px 6px;font-size:12px">"construction helmet"</span>,
and the model will draw a bounding box on top of your image.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:8px;margin-bottom:14px">'
    '<img src="https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/GvD4A4LXMXkaWmYOlDCRj.png" '
    'alt="Falcon Perception" style="width:100%;height:auto;display:block"/>'
    "</div>",
    unsafe_allow_html=True,
)

# Pros / Cons vs YOLO
st.markdown('<div class="hv-h1" style="font-size:18px;margin:10px 0 10px">Falcon Perception vs YOLO</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:14px 16px">
  <div class="hv-mono" style="font-size:11px;letter-spacing:1.2px;color:#71736D;margin-bottom:6px">FALCON PERCEPTION</div>
  <div class="hv-h1" style="font-size:16px;margin-bottom:10px">Open-vocabulary, language-grounded</div>
  <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#5FBC73;margin-bottom:4px">PROS</div>
  <ul style="margin:0 0 10px 18px;padding:0;font-size:12.5px;color:#141414;line-height:1.6">
    <li>Uses language — could describe anything</li>
    <li>accurate</li>
    <li>doesn't need training</li>
  </ul>
  <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#EC5B4C;margin-bottom:4px">CONS</div>
  <ul style="margin:0 0 0 18px;padding:0;font-size:12.5px;color:#141414;line-height:1.6">
    <li>Language can be ambiguous, hard to detect negative classes</li>
    <li>Can only detect one class per run / pass</li>
    <li>Takes 3–10&nbsp;s to process per image</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div style="background:#141414;color:#FFFFFF;padding:14px 16px;border:1px solid #141414">
  <div class="hv-mono" style="font-size:11px;letter-spacing:1.2px;color:#9B9D97;margin-bottom:6px">YOLO (v8 / v26)</div>
  <div class="hv-h1" style="font-size:16px;margin-bottom:10px;color:#FFFFFF">Closed-set, single-shot detector</div>
  <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#EFE600;margin-bottom:4px">PROS</div>
  <ul style="margin:0 0 10px 18px;padding:0;font-size:12.5px;color:#E4E5E2;line-height:1.6">
    <li>Multi-class in one pass</li>
    <li>Box-only, fast and lean</li>
    <li>Strong on the trained vocabulary</li>
  </ul>
  <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#9B9D97;margin-bottom:4px">CONS</div>
  <ul style="margin:0 0 0 18px;padding:0;font-size:12.5px;color:#E4E5E2;line-height:1.6">
    <li>needs training and data<li>
    <li>only knows classes its trained on</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hv-mono" style="font-size:11px;color:#71736D;margin-top:8px">All 10 images below are side-by-side renders from <span style="color:#141414">falcon/outputs/comparison</span> (fallback <span style="color:#141414">falcon/output/comparison</span>). Detections for the table are read from <span style="color:#141414">falcon/outputs/yolov26spv/&lt;image&gt;/detections.json</span>.</div>',
    unsafe_allow_html=True,
)
st.markdown("<hr style='margin:18px 0'/>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def list_comparison_images():
    if not COMP_DIR.is_dir():
        return []
    # canonical + fallback
    files = sorted(COMP_DIR.glob("*.jpg"))
    if not files:
        files = sorted(COMP_DIR.glob("*.png"))
    # dedupe by name if both dirs existed (we only list canonical but keep fallback lookup)
    return [str(p) for p in files]


@st.cache_data(show_spinner=False)
def load_detections(stem: str):
    """stem = filename without suffix, e.g. 000037_jpg.rf...."""
    base = stem  # folder name matches stem (without .jpg)
    p = YOLO_DIR / base / "detections.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # normalize to list of dicts
            if isinstance(data, dict) and "detections" in data:
                data = data["detections"]
            return data if isinstance(data, list) else []
        except Exception:
            return []
    # fallback: predictions_yolo.txt (class cx cy w h)
    txt = YOLO_DIR / base / "predictions_yolo.txt"
    if txt.exists():
        rows = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    rows.append({"label": parts[0], "conf": None, "box": list(map(float, parts[1:5]))})
                except Exception:
                    continue
        return rows
    return []


image_paths = list_comparison_images()

if not image_paths:
    st.warning(f"No comparison images found in `{COMP_DIR}` (and fallback `{CANDIDATES[1]}`).")
    st.stop()

# ---- Gallery — single comparison view with left/right navigation ----
st.markdown(
    f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap">'
    f'<div class="hv-h1" style="font-size:20px">Gallery — Falcon vs YOLO</div>'
    f'<div class="hv-mono" style="font-size:11px;color:#71736D">{len(image_paths)} comparisons</div>'
    f"</div>",
    unsafe_allow_html=True,
)

if "falcon_idx" not in st.session_state:
    st.session_state.falcon_idx = 0
st.session_state.falcon_idx = max(0, min(st.session_state.falcon_idx, len(image_paths) - 1))
idx = st.session_state.falcon_idx
labels = [Path(p).name for p in image_paths]
selected = labels[idx]

# Navigation row — left / counter / right (replaces selectbox + ALL COMPARISONS grid)
nav_prev, nav_center, nav_next = st.columns([1, 3, 1], vertical_alignment="center")
with nav_prev:
    if st.button("‹  Prev", disabled=idx == 0, use_container_width=True, key="falcon_prev"):
        st.session_state.falcon_idx = idx - 1
        st.rerun()
with nav_center:
    st.markdown(
        f'<div class="hv-mono" style="text-align:center;font-size:11px;color:#4A4B47;'
        f'padding:6px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
        f'{idx + 1} / {len(image_paths)} &nbsp;·&nbsp; <span style="color:#141414">{selected}</span></div>',
        unsafe_allow_html=True,
    )
with nav_next:
    if st.button("Next  ›", disabled=idx == len(image_paths) - 1, use_container_width=True, key="falcon_next"):
        st.session_state.falcon_idx = idx + 1
        st.rerun()

# Big comparison image — full-bleed (edge-to-edge) via inline base64 + vw breakout
sel_path = Path(COMP_DIR / selected)
if not sel_path.exists():
    alt = CANDIDATES[1] / selected
    if alt.exists():
        sel_path = alt
try:
    b64 = base64.b64encode(sel_path.read_bytes()).decode("ascii")
    ext = sel_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "image/png" if ext == "png" else "image/jpeg"
    st.markdown(
        f'<div class="falcon-hero"><img src="data:{mime};base64,{b64}" alt="{selected}"/></div>'
        f'<div class="falcon-caption">{selected}</div>',
        unsafe_allow_html=True,
    )
except Exception:
    st.error(f"Failed to load `{sel_path}`.")

st.markdown("<hr style='margin:18px 0'/>", unsafe_allow_html=True)

# ---- Detection table ----
st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:4px">Detection results</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#71736D;margin-bottom:10px">YOLO detections for the selected image from <span class="hv-mono" style="font-size:11px;color:#141414">yolov26spv/&lt;image&gt;/detections.json</span> '
    "(conf 0–1, box x0 y0 x1 y1 normalized). Falcon side is visual — same image rendered side-by-side above.",
    unsafe_allow_html=True,
)

stem = Path(selected).stem  # 000037_jpg.rf.xxx without .jpg? Actually Path stem drops .jpg only, keeps _jpg.rf... -> matches folder
# The folder names are stem without trailing .jpg extension, which is exactly Path.stem
dets = load_detections(stem)

if not dets:
    st.info(f"No structured detections found for `{stem}` — gallery still shows the side-by-side render, but the table is empty for this image.")
else:
    df = pd.DataFrame(dets)
    # normalize columns for display
    # expected keys: label/canonical label, conf, box, canon_id, key
    if "label" not in df.columns and "key" in df.columns:
        df["label"] = df["key"]
    if "conf" not in df.columns:
        df["conf"] = None
    # box formatting
    if "box" in df.columns:
        df["box"] = df["box"].apply(lambda b: f"[{', '.join(f'{x:.3f}' for x in b)}]" if isinstance(b, (list, tuple)) else str(b))
    # counts per class
    counts = df["label"].value_counts().rename_axis("class").reset_index(name="count") if "label" in df.columns else pd.DataFrame()
    c_left, c_right = st.columns([2, 1])
    with c_left:
        show_cols = [c for c in ["label", "key", "canon_id", "conf", "box"] if c in df.columns]
        # round conf
        if "conf" in df.columns:
            df["conf"] = df["conf"].apply(lambda x: round(float(x), 3) if isinstance(x, (int, float)) else x)
        st.dataframe(df[show_cols] if show_cols else df, width="stretch", hide_index=True)
    with c_right:
        if not counts.empty:
            st.markdown('<div class="hv-mono" style="font-size:11px;letter-spacing:1px;color:#71736D;margin-bottom:6px">COUNT BY CLASS</div>', unsafe_allow_html=True)
            st.dataframe(counts, width="stretch", hide_index=True)
        st.markdown(
            '<div style="background:#EFE600;border:1px solid #141414;padding:8px 10px;margin-top:8px;font-size:11px;color:#141414">'
            '<span class="hv-mono" style="letter-spacing:.5px">NOTE</span> · Falcon visual is the left/right render above; numeric rows here are the YOLO side that remains machine-readable.'
            "</div>",
            unsafe_allow_html=True,
        )

st.caption("Gallery reads directly from falcon/outputs/comparison (fallback falcon/output/comparison); table reads yolov26spv/detections.json per image. Add new comparisons by dropping JPGs into that folder — they appear on next reload (clear cache if needed).")
