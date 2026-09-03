"""HI-VIS — side-by-side comparison of the trained runs. Deliberately minimal
for now: no upload box of its own and no detail view (tiles aren't
clickable) — it runs the two models we're actively comparing over a fixed
set of demo photos (assets/images/), not whatever's loaded on the Demo
page, so this page always shows the same comparison regardless of what
anyone's uploaded elsewhere. Shows the three core stat tiles (Photos
Processed / PPE Exceptions Flagged / Image Compliance) per model, then the
same photo run through both models side by side, one row per photo, so the
two can be judged directly against each other on identical input. The
whole page body (name label, stat tiles, the PHOTOS divider, every photo
row) is ONE CSS grid rather than separate st.columns() calls — see the
comment below — so each model's name label can be position:sticky and stay
visible the whole way down, with exactly one copy of it.

Raw detections are read from assets/images/compare_cache.json, precomputed
by precompute_compare_cache.py — see that script's docstring. Re-run it
whenever a demo image is added/removed/replaced, or either model's weights
change. If the cache is missing an entry for a given model (script never
run, or a new image landed since), that one model falls back to running
live here instead of breaking the page — slower, but correct.

The other trained runs (v8, v26, css-m-150/300, merged, merged-m,
mergedpeople, altec) aren't dropped — they're just not shown on this page.
Their weights and training runs stay in the repo; see detector.py and the
Model Performance page for their metrics.
"""

import json
import re
from pathlib import Path

import streamlit as st

import detector
import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("MODEL COMPARISON"), unsafe_allow_html=True)

# Fixed demo set, not the Demo page's batch — so this page's comparison never
# depends on what (if anything) someone's uploaded elsewhere. Kept in sync
# with precompute_compare_cache.py, which caches raw detections for exactly
# this folder's images.
IMAGES_DIR = detector.REPO_ROOT / "assets" / "images"
CACHE_PATH = IMAGES_DIR / "compare_cache.json"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    (p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS),
    key=lambda p: p.name,
) if IMAGES_DIR.is_dir() else []

if not image_paths:
    st.info(f"No demo images found in `{IMAGES_DIR}`. Add some photos there to populate this page.")
    st.stop()

items = [{"key": p.name, "name": p.name, "image": vh.load_image(p.read_bytes())} for p in image_paths]

threshold = st.session_state.get("threshold", 0.35)
required = tuple(
    s for s, dflt in (
        ("hardhat", True), ("vest", True), ("mask", True), ("gloves", True), ("boots", True),
    )
    if st.session_state.get(f"require_{s}", dflt)
)

# Only the two runs we're actively comparing in this deck's story: current
# production model vs. the SuperVisor.v4 run (preprocessed dataset). Every
# other trained run stays out of this page — see the module docstring.
MODELS = [
    {"key": "merged-m-v2", "weights": detector.MERGED_M_V2_WEIGHTS, "label": detector.MERGED_M_V2_LABEL},
    {"key": "supervisorv4", "weights": detector.SUPERVISOR_V4_WEIGHTS, "label": detector.SUPERVISOR_V4_LABEL},
]


@st.cache_data
def _load_precomputed_cache(cache_path_str, cache_mtime):
    """Cached on (path, mtime) so a re-run of precompute_compare_cache.py is
    picked up without restarting the app — Streamlit's own file-watcher
    reruns the script, and a changed mtime busts this cache."""
    return json.loads(Path(cache_path_str).read_text())


_raw_cache = None
if CACHE_PATH.exists():
    try:
        _raw_cache = _load_precomputed_cache(str(CACHE_PATH), CACHE_PATH.stat().st_mtime)
    except (json.JSONDecodeError, OSError):
        _raw_cache = None

# Per-model raw detections: precomputed cache first (fast — no model load,
# no inference), live detect_raw() as a fallback only for whatever the cache
# doesn't cover, so a missing/partial/stale cache degrades gracefully
# instead of breaking the page.
st.session_state.setdefault("_compare_raw_live", {})

runs = []
for m in MODELS:
    entry = {"key": m["key"], "label": m["label"], "weights": m["weights"], "results": None, "warning": None}
    cached_model = ((_raw_cache or {}).get("models", {})).get(m["key"], {}).get("results", {})
    raw_by_image = {}
    missing = []
    for it in items:
        if it["key"] in cached_model:
            raw_by_image[it["key"]] = cached_model[it["key"]]
        else:
            missing.append(it)

    if missing:
        model = detector.load_model(m["weights"])
        if model is None:
            entry["warning"] = f"Weights not found at `{m['weights']}`."
        else:
            live_cache = st.session_state._compare_raw_live.setdefault(m["key"], {})
            for it in missing:
                if it["key"] not in live_cache:
                    live_cache[it["key"]] = detector.detect_raw(model, it["image"])
                raw_by_image[it["key"]] = live_cache[it["key"]]

    if len(raw_by_image) == len(items):
        entry["results"] = [detector.assess(raw_by_image[it["key"]], threshold, required=required) for it in items]
    runs.append(entry)

# Built as ONE CSS grid — model name, stat tiles, the "PHOTOS" divider and
# every photo cell all share the same parent grid container, rather than
# each model's name+tiles living in its own short-lived st.columns() call.
# position:sticky only stays "stuck" for as long as the viewport is
# scrolling through its own parent element's box, so a label sitting alone
# in a one-row st.columns() container un-sticks again almost immediately.
# Making everything below the header one shared grid gives the sticky name
# label the full scroll range of the whole page (tiles + every photo row)
# to stay pinned across, with exactly ONE copy of each model's name.
STICKY_TOP = "52px"  # nudge this if the header sits under/behind Streamlit's own toolbar


def _flat(html):
    """Collapse an HTML fragment to a single line. Concatenating many
    multi-line, indented fragments back to back (no blank line between
    them) makes Streamlit's markdown parser misread the join as an indented
    code block instead of one raw HTML block, so every </div> and <img> in
    it shows up as literal escaped text on the page instead of rendering.
    Collapsing whitespace sidesteps that — safe here since nothing in this
    grid depends on preserved whitespace."""
    return re.sub(r"\s+", " ", html).strip()


cells = []

# Row 1, one cell per model: the sticky name label — the only copy of it.
for run in runs:
    cells.append(_flat(f"""
        <div style="position:sticky;top:{STICKY_TOP};z-index:50;background:#E4E5E2;
                    padding:10px 0 8px;margin-bottom:10px;border-bottom:1px solid #C4C6C0">
          <div class="hv-h1" style="font-size:20px">{run['label']}</div>
        </div>
    """))

# Row 2, one cell per model: stat tiles, or a warning card if this model has
# no results at all (weights missing, etc.).
for run in runs:
    if run["results"] is None:
        cells.append(_flat(f"""
            <div style="background:#FFFFFF;border:2px dashed #9B9D97;padding:16px;
                        color:#4A4B47;font-size:13px;margin-bottom:10px">
              {run["warning"] or "No results for this model."}
            </div>
        """))
        continue

    results = run["results"]
    assessed = [r for r in results if r["verdict"] != "none"]
    non_items = [r for r in results if r["verdict"] == "non"]
    rate = round((len(assessed) - len(non_items)) / len(assessed) * 100) if assessed else 0
    exc_bg = "#EFE600" if non_items else "#FFFFFF"

    cells.append(_flat(f"""
        <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:10px">
          <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">PHOTOS PROCESSED</div>
            <div class="hv-h1" style="font-size:44px;line-height:1;color:#FFFFFF">{len(items)}</div>
            <div style="font-size:12px;color:#9B9D97">{len(assessed)} assessed · {len(items) - len(assessed)} no person detected</div>
          </div>
          <div style="background:{exc_bg};color:#141414;border:1px solid #141414;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#3A3B30">PPE EXCEPTIONS FLAGGED</div>
            <div class="hv-h1" style="font-size:44px;line-height:1">{len(non_items)}</div>
            <div style="font-size:12px;color:#3A3B30">{"photos with at least one finding" if non_items else "none at this threshold"}</div>
          </div>
          <div style="background:#FFFFFF;color:#141414;border:1px solid #C4C6C0;padding:16px 20px 14px">
            <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">IMAGE COMPLIANCE</div>
            <div class="hv-h1" style="font-size:44px;line-height:1">{rate}%</div>
            <div style="font-size:12px;color:#71736D">{len(assessed)} of {len(items)} photos assessed at this threshold</div>
          </div>
        </div>
    """))

# Row 3, spanning every column: the "PHOTOS" section divider.


# Remaining rows, one per photo: that photo's result from each model, side by side.
for i, it in enumerate(items):
    for run in runs:
        if run["results"] is None:
            cells.append("<div></div>")
            continue
        r = run["results"][i]
        thumb_b64 = vh.b64_image(vh.draw_overlay(it["image"], r["persons"], show_boxes=True, show_labels=False), max_dim=360)
        fc = vh.flag_confidence(r, required)
        conf_label = f"{fc:.2f} conf" if fc is not None else "— conf"
        cells.append(_flat(f"""
            <div style="background:#FFFFFF;border:1px solid #C4C6C0;margin-bottom:10px" title="{it['name']}">
              <img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block"/>
              <div style="display:flex;align-items:center;gap:8px;padding:6px 8px">
                {vh.verdict_badge(r["verdict"])}
                <span class="hv-mono" style="font-size:10.5px;color:#4A4B47;white-space:nowrap">{conf_label}</span>
              </div>
            </div>
        """))

st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat({len(runs)},1fr);gap:0 16px">'
    + "".join(cells) +
    '</div>',
    unsafe_allow_html=True,
)
