"""HI-VIS — compliance-over-time from a labeled series of CCTV stills.

Separate from the Demo page's live batch on purpose: that page is "analyse
whatever's uploaded right now," this page is "build a small, dated dataset
once (site inspection photos, a week of CCTV grabs) and track how
compliance moved across it" -- a different question, a different persisted
dataset (data/timeline/, via timeline_helpers.py), its own upload box.

Each photo gets a manually-assigned date (see the brainstorm behind this:
CCTV exports don't reliably keep EXIF, and for a curated demo narrative you
want to control which photo represents which point anyway, not depend on
metadata). What's persisted is the raw detections, not a baked verdict --
so the trend below always reflects whatever threshold / WHAT COUNTS AS
COMPLIANT rule is currently set on the Demo page, exactly like the Model
Comparison page already does for its own live batch.
"""

import datetime as dt
import hashlib

import plotly.graph_objects as go
import streamlit as st

import detector
import timeline_helpers as th
import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("HISTORICAL DATA", detector.MODEL_LABEL), unsafe_allow_html=True)

model = detector.load_model()
if model is None:
    st.error(f"No trained weights found at `{detector.WEIGHTS_PATH}`. See the Demo page for how to restore them.")
    st.stop()

st.markdown(
    '<div style="font-size:12.5px;color:#4A4B47;margin-bottom:18px">Build a dated compliance '
    'timeline from a series of site photos -- a week of CCTV stills, or repeat visits to the same '
    'site -- by labeling each with the date it was taken. Saved here under '
    '<code>data/timeline/</code>, separate from whatever batch is loaded on the Demo page.</div>',
    unsafe_allow_html=True,
)

_ALL_SLOTS = ("hardhat", "vest", "mask", "gloves", "boots")  # fixed order, matches pages/demo.py

# ---------------------------------------------------------------------------
# 1. upload + label new photos (not yet part of the saved timeline)
# ---------------------------------------------------------------------------

st.session_state.setdefault("_timeline_pending", {})
pending = st.session_state._timeline_pending
saved_keys = set(th.load_manifest().keys())

st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:6px">ADD PHOTOS</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Select CCTV / site photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
    label_visibility="collapsed", key="historical_uploader",
)
for f in uploaded_files or []:
    raw_bytes = f.getvalue()
    key = hashlib.md5(raw_bytes).hexdigest()
    if key in saved_keys or key in pending:
        continue  # already in the timeline, or already waiting to be labeled below
    img = vh.load_image(raw_bytes)
    raw = detector.detect_raw(model, img)
    pending[key] = {"image": img, "raw": raw, "name": f.name}

if pending:
    st.caption(f"{len(pending)} photo{'s' if len(pending) != 1 else ''} awaiting a date before they're added to the timeline:")
    to_save = []  # (key, date_value, caption_value) captured this run, used if the button below is clicked
    for key, p in list(pending.items()):
        c1, c2, c3, c4 = st.columns([1, 2, 3, 1])
        with c1:
            st.image(p["image"], width="stretch")
        with c2:
            date_val = st.date_input("Date taken", value=dt.date.today(), key=f"tl_date_{key}")
        with c3:
            caption_val = st.text_input("Caption (optional)", value="", key=f"tl_caption_{key}",
                                         placeholder="e.g. Week 1 walkthrough")
        with c4:
            if st.button("✕ Discard", key=f"tl_discard_{key}"):
                pending.pop(key, None)
                st.rerun()
        to_save.append((key, date_val, caption_val))

    if st.button(f"Add {len(pending)} photo{'s' if len(pending) != 1 else ''} to timeline",
                 key="tl_add_all", type="primary"):
        for key, date_val, caption_val in to_save:
            p = pending[key]
            th.save_entry(key, p["image"], p["raw"], p["name"], date_val.isoformat(), caption_val)
        st.session_state._timeline_pending = {}
        st.toast(f"Added {len(to_save)} photo{'s' if len(to_save) != 1 else ''} to the timeline.")
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. the saved timeline itself
# ---------------------------------------------------------------------------

manifest = th.load_manifest()
if not manifest:
    st.info("No timeline entries yet. Upload photos above and give each a date to get started.")
    st.stop()

threshold = st.session_state.get("threshold", 0.35)
required = tuple(
    s for s, dflt in (
        ("hardhat", True), ("vest", True), ("mask", True), ("gloves", True), ("boots", True),
    )
    if st.session_state.get(f"require_{s}", dflt)
)

rows, invalid = [], []
for key, meta in manifest.items():
    raw = th.load_raw(key)
    assessment = detector.assess(raw, threshold, required=required)
    persons = assessment["persons"]
    n = len(persons)
    compliant = sum(1 for p in persons if p["verdict"] == "ok")
    row = {
        "key": key, "name": meta.get("name", ""), "caption": meta.get("caption", ""),
        "date_str": meta.get("date", ""), "raw": raw, "n_persons": n,
        "compliance_pct": round(compliant / n * 100, 1) if n else None,
        "missing_counts": {s: sum(1 for p in persons if p["status"][s]["state"] == "missing") for s in _ALL_SLOTS},
        "verdict": assessment["verdict"],
    }
    try:
        row["date"] = dt.date.fromisoformat(row["date_str"])
    except ValueError:
        row["date"] = None
    (rows if row["date"] is not None else invalid).append(row)

rows.sort(key=lambda r: r["date"])

if invalid:
    st.warning(f"{len(invalid)} timeline entr{'y has' if len(invalid) == 1 else 'ies have'} an unreadable date "
               f"and are excluded from the chart below (still shown in the filmstrip).")

st.markdown(
    f'<div style="font-size:12.5px;color:#4A4B47;margin-bottom:14px">Recomputed at threshold '
    f'<b>{threshold:.2f}</b> using the same WHAT COUNTS AS COMPLIANT rule set as the Demo page. '
    f'Change either there to see this update here.</div>',
    unsafe_allow_html=True,
)

# per-date mean compliance -- the line; individual photos -- the dots.
# Two different questions ("what's the trend" vs "what's this one photo"),
# never forced onto the same number.
mean_by_date = {}
for r in rows:
    if r["compliance_pct"] is not None:
        mean_by_date.setdefault(r["date"], []).append(r["compliance_pct"])
dates_sorted = sorted(mean_by_date)
mean_x = dates_sorted
mean_y = [round(sum(mean_by_date[d]) / len(mean_by_date[d]), 1) for d in dates_sorted]

# ---------------------------------------------------------------------------
# hero stat tiles
# ---------------------------------------------------------------------------

dated_rows = [r for r in rows if r["compliance_pct"] is not None]
date_range_txt = f"{rows[0]['date'].isoformat()} → {rows[-1]['date'].isoformat()}" if rows else "—"
delta_txt, delta_sub = "—", "not enough dated photos yet"
if len(dates_sorted) >= 2:
    delta = mean_y[-1] - mean_y[0]
    sign = "+" if delta >= 0 else ""
    delta_txt = f"{sign}{delta:.0f} pts"
    delta_sub = f"{mean_y[0]:.0f}% on {dates_sorted[0].isoformat()} → {mean_y[-1]:.0f}% on {dates_sorted[-1].isoformat()}"

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown(f"""
    <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px;height:100%">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">PHOTOS IN TIMELINE</div>
      <div class="hv-h1" style="font-size:40px;line-height:1;color:#FFFFFF">{len(manifest)}</div>
      <div style="font-size:12px;color:#9B9D97">{len(rows)} dated · {len(invalid)} unreadable date</div>
    </div>""", unsafe_allow_html=True)
with t2:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:16px 20px 14px;height:100%">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">DATE RANGE</div>
      <div class="hv-h1" style="font-size:22px;line-height:1.3;margin-top:6px">{date_range_txt}</div>
    </div>""", unsafe_allow_html=True)
with t3:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:16px 20px 14px;height:100%">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">COMPLIANCE CHANGE</div>
      <div class="hv-h1" style="font-size:40px;line-height:1">{delta_txt}</div>
      <div style="font-size:12px;color:#71736D">{delta_sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# chart 1 -- compliance % trend (its own axis: 0-100%, nothing else sharing it)
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:16px;margin-bottom:2px">COMPLIANCE OVER TIME</div>', unsafe_allow_html=True)
st.caption("Line = mean compliance per date. Dots = individual photos (hover for detail).")

point_x = [r["date"] for r in dated_rows]
point_y = [r["compliance_pct"] for r in dated_rows]
point_text = [
    f"{r['name']}<br>{r['date_str']}<br>{r['compliance_pct']}% compliant ({r['n_persons']} assessed)"
    + (f"<br>{r['caption']}" if r["caption"] else "")
    for r in dated_rows
]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=mean_x, y=mean_y, mode="lines", line=dict(color="#14213D", width=2),
    hoverinfo="skip", showlegend=False,
))
fig1.add_trace(go.Scatter(
    x=point_x, y=point_y, mode="markers",
    marker=dict(size=10, color="#14213D", line=dict(color="#FFFFFF", width=2)),
    text=point_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
))
fig1.update_layout(
    height=340, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#141414", size=12.5),
    xaxis=dict(showgrid=False, linecolor="#C4C6C0", tickfont=dict(color="#4A4B47")),
    yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#E4E5E2", gridwidth=1,
               zeroline=False, tickfont=dict(color="#4A4B47")),
)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# chart 2 -- violation-type breakdown (its own separate chart/axis -- counts,
# never forced onto the % axis above)
# ---------------------------------------------------------------------------

tracked_slots = [s for s in _ALL_SLOTS if any(
    b["key"] in detector.SLOT_ITEMS[s] for r in dated_rows for b in r["raw"]
)]

if tracked_slots:
    st.markdown('<div class="hv-h1" style="font-size:16px;margin:22px 0 2px">VIOLATIONS BY TYPE</div>', unsafe_allow_html=True)
    st.caption("Stacked count of missing-PPE findings per date, by item type.")

    by_date_slot = {}
    for r in dated_rows:
        acc = by_date_slot.setdefault(r["date"], {s: 0 for s in tracked_slots})
        for s in tracked_slots:
            acc[s] += r["missing_counts"][s]

    fig2 = go.Figure()
    for slot in tracked_slots:  # fixed order (matches _ALL_SLOTS) -- never cycled/reordered
        neg_key = detector.SLOT_ITEMS[slot][1]
        meta = detector.CLASS_META[neg_key]
        fig2.add_trace(go.Bar(
            x=dates_sorted, y=[by_date_slot[d][slot] for d in dates_sorted],
            name=meta["label"], marker=dict(color=meta["color"]),
            hovertemplate=f"{meta['label']}: " + "%{y}<extra></extra>",
        ))
    fig2.update_layout(
        barmode="stack", bargap=0.35,
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#141414", size=12.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor="#C4C6C0", tickfont=dict(color="#4A4B47")),
        yaxis=dict(gridcolor="#E4E5E2", gridwidth=1, zeroline=False, tickfont=dict(color="#4A4B47"),
                   rangemode="tozero"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
else:
    st.caption("The loaded model hasn't detected any tracked PPE item across these photos yet -- "
               "nothing to break down by type.")

# ---------------------------------------------------------------------------
# 3. filmstrip -- every saved photo, in date order, click-free (see
# model_compare.py's own "not clickable yet" note -- same scope call here)
# ---------------------------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="hv-h1" style="font-size:16px;margin-bottom:10px">PHOTOS</div>', unsafe_allow_html=True)

film_cols = st.columns(4)
for i, r in enumerate(rows + invalid):
    img = th.load_image(r["key"])
    if img is None:
        continue
    assessment = detector.assess(r["raw"], threshold, required=required)
    thumb = vh.draw_overlay(img, assessment["persons"], show_boxes=True, show_labels=False)
    with film_cols[i % 4]:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #C4C6C0;margin-bottom:10px" title="{r['name']}">
          <img src="data:image/jpeg;base64,{vh.b64_image(thumb, max_dim=320)}"
               style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block"/>
          <div style="padding:6px 8px;display:flex;flex-direction:column;gap:4px">
            <span class="hv-mono" style="font-size:10.5px;color:#4A4B47">{r['date_str'] or 'no date'}</span>
            {vh.verdict_badge(assessment["verdict"])}
            {f'<span style="font-size:11px;color:#71736D">{r["caption"]}</span>' if r["caption"] else ""}
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✕ Remove", key=f"tl_remove_{r['key']}"):
            th.delete_entry(r["key"])
            st.rerun()
