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

# ---- Top: two boxes — Falcon + YOLO, each with image as background ----
_FALCON_BG = "https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/GvD4A4LXMXkaWmYOlDCRj.png"
_YOLO_BG = "https://play-lh.googleusercontent.com/l6FTUmnhmusjPgNtJPSF1U1DvYCvwNgSt440oM64hqpc7ZvudhDAAg0ri3cF6IM6vSoR=s94-rw"
st.markdown(
    f"""
<style>
.falcon-intro-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 10px;
}}
@media (max-width: 860px) {{
  .falcon-intro-grid {{ grid-template-columns: 1fr; }}
}}
/* Left: Falcon — image as background, text on a readable overlay */
.falcon-box {{
  position: relative;
  overflow: hidden;
  border: 1px solid #141414;
  min-height: 360px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.88) 36%, rgba(255,255,255,0.96) 100%),
    url('{_FALCON_BG}') center / cover no-repeat;
}}
.falcon-box-inner {{
  position: relative;
  z-index: 1;
  padding: 16px 16px 14px;
}}
/* Right: YOLO — image as background with dark overlay for readability */
.yolo-box {{
  position: relative;
  overflow: hidden;
  border: 1px solid #141414;
  min-height: 360px;
  background:
    linear-gradient(180deg, rgba(20,20,20,0.82) 0%, rgba(20,20,20,0.88) 50%, rgba(20,20,20,0.92) 100%),
    url('{_YOLO_BG}') center / cover no-repeat;
  color: #FFFFFF;
}}
.yolo-box-inner {{
  position: relative;
  z-index: 1;
  padding: 16px;
}}
.yolo-box .yolo-body {{ font-size:12.8px; line-height:1.65; color:#E4E5E2; }}
.yolo-box .yolo-body b {{ color:#FFFFFF; }}
</style>
<div class="falcon-intro-grid">
  <!-- FALCON -->
  <div class="falcon-box">
    <div class="falcon-box-inner">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.2px;color:#71736D;margin-bottom:6px">FALCON PERCEPTION</div>
      <div class="hv-h1" style="font-size:18px;color:#141414;margin-bottom:8px">Falcon Perception</div>
      <div style="font-size:12.8px;color:#141414;line-height:1.6;margin-bottom:12px">
        A 0.6B-parameter Small Language Model (SLM) that lives in the gap between
        machine-learning models like YOLO and huge LLMs like Gemini, ChatGPT and Claude.
        It's somewhat more accurate than YOLO, and slightly bigger than YOLO. You input an
        object name like <span class="hv-mono" style="background:#FFFFFF;border:1px solid #C4C6C0;padding:1px 5px;font-size:11.5px">"construction helmet"</span>,
        and the model will draw a bounding box on top of your image.
      </div>
      <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#5FBC73;margin-bottom:4px">PROS</div>
      <ul style="margin:0 0 10px 18px;padding:0;font-size:12.5px;color:#141414;line-height:1.6">
        <li>Uses language — could describe anything</li>
        <li>Pretty accurate</li>
      </ul>
      <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#EC5B4C;margin-bottom:4px">CONS</div>
      <ul style="margin:0 0 0 18px;padding:0;font-size:12.5px;color:#141414;line-height:1.6">
        <li>Language can be ambiguous, hard to detect negative classes</li>
        <li>Can only detect one class per run / pass</li>
        <li>Takes 3–10&nbsp;s to process per image</li>
      </ul>
    </div>
  </div>
  <!-- YOLO -->
  <div class="yolo-box"><div class="yolo-box-inner">
    <div class="hv-mono" style="font-size:11px;letter-spacing:1.2px;color:#9B9D97;margin-bottom:6px">YOLO</div>
    <div class="hv-h1" style="font-size:18px;color:#FFFFFF;margin-bottom:8px">YOLO</div>
    <div class="yolo-body" style="margin-bottom:12px">
      YOLO is a small, fast object detector trained to find specific things it has seen before —
      like helmets, vests, or people. You show it a photo and it draws boxes around what it recognizes.
      It's quick and light, which makes it great for real-time use. The trade-off is it only knows
      what it was trained on: if you want it to find something new, you need to collect examples
      and train it again.
    </div>
    <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#EFE600;margin-bottom:4px">PROS</div>
    <ul style="margin:0 0 10px 18px;padding:0;font-size:12.5px;color:#E4E5E2;line-height:1.6">
      <li>Multi-class in one pass</li>
      <li>Fast and light</li>
      <li>Reliable on the things it was trained for</li>
    </ul>
    <div class="hv-mono" style="font-size:10.5px;letter-spacing:.8px;color:#9B9D97;margin-bottom:4px">CONS</div>
    <ul style="margin:0 0 0 18px;padding:0;font-size:12.5px;color:#E4E5E2;line-height:1.6">
      <li>Needs training data</li>
      <li>Only knows classes it was trained on</li>
    </ul>
  </div></div>
</div>
<div class="hv-mono" style="font-size:11px;color:#71736D;margin-bottom:6px">All 10 images below are side-by-side renders from <span style="color:#141414">falcon/outputs/comparison</span> (fallback <span style="color:#141414">falcon/output/comparison</span>). Detections for the table are read from <span style="color:#141414">falcon/outputs/yolov26spv/&lt;image&gt;/detections.json</span>.</div>
""",
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

# ---- Detection table — simple counts per class per model ----
st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:4px">Detections per class</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#71736D;margin-bottom:10px">How many boxes each model found in each class for the selected image. '
    "YOLO = <span class=\"hv-mono\" style=\"font-size:11px;color:#141414\">yolov26spv/*.json</span> · "
    "Falcon = <span class=\"hv-mono\" style=\"font-size:11px;color:#141414\">falcon/*.json</span> · "
    "Gemini = <span class=\"hv-mono\" style=\"font-size:11px;color:#141414\">gemini/*.json</span> (9-class: person/helmet/gloves/boots/vest + no hardhat/no safety vest/no gloves/no boots).</div>",
    unsafe_allow_html=True,
)

# 9-class order: 5 positives + 4 negatives (no hardhat/no safety vest/no gloves/no boots)
CLASS_ORDER = ["person", "helmet", "gloves", "boots", "vest", "no hardhat", "no safety vest", "no gloves", "no boots"]
# For legacy YOLO txt (0-4) and Gemini/Falcon json key mapping
CLASS_ID_TO_NAME = {0: "person", 1: "helmet", 2: "gloves", 3: "boots", 4: "vest", 5: "no hardhat", 6: "no safety vest", 7: "no gloves", 8: "no boots"}

# normalize label strings to the 9 canonical names (covers YOLO/Falcon/Gemini gemini keys)
_LABEL_MAP = {
    "person": "person",
    "helmet": "helmet",
    "hardhat": "helmet",
    "no-helmet": "no hardhat",
    "no hardhat": "no hardhat",
    "no-hardhat": "no hardhat",
    "vest": "vest",
    "safety vest": "vest",
    "safety_vest": "vest",
    "no-vest": "no safety vest",
    "no safety vest": "no safety vest",
    "no-safety vest": "no safety vest",
    "novest": "no safety vest",
    "gloves": "gloves",
    "no-gloves": "no gloves",
    "no gloves": "no gloves",
    "nogloves": "no gloves",
    "boots": "boots",
    "no-boots": "no boots",
    "no boots": "no boots",
    "noboots": "no boots",
    "no-mask": "no hardhat",  # gemini uses NO-Mask as proxy for missing PPE; map to no hardhat for display
    "nomask": "no hardhat",
}
_YOLO_LABEL_MAP = _LABEL_MAP
_FALCON_LABEL_MAP = _LABEL_MAP
_GEMINI_LABEL_MAP = _LABEL_MAP


@st.cache_data(show_spinner=False)
def _falcon_counts_for(stem: str) -> dict:
    # new 9-class: falcon/outputs/falcon/<stem>.json (flat, key/conf/box)
    jp = APP_DIR / "falcon" / "outputs" / "falcon" / f"{stem}.json"
    if jp.exists():
        try:
            data = json.loads(jp.read_text())
            if isinstance(data, dict) and "detections" in data:
                data = data["detections"]
            counts: dict[str, int] = {}
            for d in data if isinstance(data, list) else []:
                raw = str(d.get("key") or d.get("label") or "").strip().lower()
                name = _FALCON_LABEL_MAP.get(raw, raw)
                if name in CLASS_ORDER:
                    counts[name] = counts.get(name, 0) + 1
            return counts
        except Exception:
            pass
    # legacy: chosen-pics-5cls per-image txt (5cls) or flat
    for cand in [
        APP_DIR / "falcon" / "outputs" / "chosen-pics-5cls" / stem / "predictions_yolo.txt",
        APP_DIR / "falcon" / "outputs" / "chosen-pics-5cls" / f"{stem}.txt",
        APP_DIR / "falcon" / "outputs" / "falcon" / f"{stem}.txt",
    ]:
        if cand.exists():
            counts = {}
            for line in cand.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                name = CLASS_ID_TO_NAME.get(cid)
                if name:
                    counts[name] = counts.get(name, 0) + 1
            return counts
    return {}


@st.cache_data(show_spinner=False)
def _yolo_counts_for(stem: str) -> dict:
    # new flat: falcon/outputs/yolov26spv/<stem>.json
    for cand in [
        APP_DIR / "falcon" / "outputs" / "yolov26spv" / f"{stem}.json",
        APP_DIR / "falcon" / "outputs" / "yolov26spv" / stem / "detections.json",
    ]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                if isinstance(data, dict) and "detections" in data:
                    data = data["detections"]
                counts: dict[str, int] = {}
                for d in data if isinstance(data, list) else []:
                    raw = str(d.get("label") or d.get("key") or "").strip().lower()
                    name = _YOLO_LABEL_MAP.get(raw, raw)
                    if name in CLASS_ORDER:
                        counts[name] = counts.get(name, 0) + 1
                return counts
            except Exception:
                pass
    # fallback to predictions_yolo.txt (flat or per-folder)
    for cand in [
        APP_DIR / "falcon" / "outputs" / "yolov26spv" / f"{stem}.txt",
        APP_DIR / "falcon" / "outputs" / "yolov26spv" / stem / "predictions_yolo.txt",
    ]:
        if cand.exists():
            counts = {}
            for line in cand.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                name = CLASS_ID_TO_NAME.get(cid)
                if name:
                    counts[name] = counts.get(name, 0) + 1
            return counts
    return {}


@st.cache_data(show_spinner=False)
def _gt_counts_for(stem: str) -> dict | None:
    # GT for the chosen-pics comparison set is not shipped in this repo
    # (data/merged is git-ignored). Try a few plausible locations; return None if missing.
    candidates = [
        APP_DIR / "falcon" / "outputs" / "chosen-pics-5cls" / stem / "gt.txt",
        APP_DIR / "data" / "merged" / "train" / "labels" / f"{stem}.txt",
        APP_DIR / "falcon" / "outputs" / "gt" / f"{stem}.txt",
    ]
    for p in candidates:
        if p.exists():
            counts: dict[str, int] = {}
            for line in p.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                name = CLASS_ID_TO_NAME.get(cid)
                if name:
                    counts[name] = counts.get(name, 0) + 1
            return counts
    return None


def _gemini_counts_for(stem: str) -> dict:
    # gemini flat json; ppe_0837 uses _2 variant
    candidates = [APP_DIR / "falcon" / "outputs" / "gemini" / f"{stem}.json"]
    if stem == "ppe_0837_jpg.rf.qwzMohKlnO50iRnP6P8m":
        candidates.append(APP_DIR / "falcon" / "outputs" / "gemini" / "ppe_0837_jpg.rf.qwzMohKlnO50iRnP6P8m_2.json")
    for cand in candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                if isinstance(data, dict) and "detections" in data:
                    data = data["detections"]
                counts: dict[str, int] = {}
                for d in data if isinstance(data, list) else []:
                    raw = str(d.get("key") or d.get("label") or "").strip().lower()
                    name = _GEMINI_LABEL_MAP.get(raw, raw)
                    if name in CLASS_ORDER:
                        counts[name] = counts.get(name, 0) + 1
                return counts
            except Exception:
                pass
    return {}


stem = Path(selected).stem
yolo_c = _yolo_counts_for(stem)
falcon_c = _falcon_counts_for(stem)
gemini_c = _gemini_counts_for(stem)

rows = []
for cls in CLASS_ORDER:
    rows.append({"Class": cls, "YOLO": yolo_c.get(cls, 0), "Falcon": falcon_c.get(cls, 0), "Gemini": gemini_c.get(cls, 0)})

count_df = pd.DataFrame(rows)

# Totals row
tot_yolo = sum(yolo_c.get(k, 0) for k in CLASS_ORDER)
tot_falcon = sum(falcon_c.get(k, 0) for k in CLASS_ORDER)
tot_gemini = sum(gemini_c.get(k, 0) for k in CLASS_ORDER)
count_df.loc[len(count_df)] = ["Total", tot_yolo, tot_falcon, tot_gemini]

st.dataframe(count_df, width="stretch", hide_index=True)

st.caption("Counts are boxes per class for the selected image. YOLO from yolov26spv/*.json, Falcon from falcon/*.json, Gemini from gemini/*.json.")

st.caption("Gallery reads from falcon/outputs/comparison (fallback falcon/output/comparison). Add new comparisons by dropping JPGs into that folder — they appear on next reload (clear cache if needed).")
