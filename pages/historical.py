"""HI-VIS — compliance-over-time from a labeled series of CCTV stills.

Separate from the Demo page's live batch on purpose: that page is "analyse
whatever's uploaded right now," this page is "build a small, dated dataset
once (site inspection photos, a week of CCTV grabs) and track how
compliance moved across it" -- a different question, a different persisted
dataset (data/timeline/, via timeline_helpers.py), its own upload box.

The trend is the point of this page, so it's what's on screen first --
uploading photos, correcting labels, and retagging dates all live inside
the "MANAGE PHOTOS" toggle at the bottom instead of at the top, so they
don't compete with the charts for space. It's collapsed by default once
there's a timeline to show, and auto-expands when there isn't (nothing
saved yet), so a first-time user still lands on the upload form rather
than an empty page.

Each photo gets a manually-assigned date AND time (see the brainstorm
behind this: CCTV exports don't reliably keep EXIF, and for a curated demo
narrative you want to control which photo represents which point anyway,
not depend on metadata) -- the time matters because a day's batch is
usually several photos, not one, so it's what keeps them ordered correctly
within their day. The two trend charts below still bucket by day (that's
the granularity the story is told at); time mainly drives sort order and
the filmstrip/hover detail. What's persisted is the raw detections, not a
baked verdict -- so the trend always reflects whatever threshold / WHAT
COUNTS AS COMPLIANT rule is currently set on the Demo page, exactly like
the Model Comparison page already does for its own live batch.

The model won't catch every violation on its own, especially on small,
top-down CCTV figures -- a missed NO-vest/NO-boots box just reads as
"not visible" (not "compliant" and not "non-compliant"), which can make a
photo look cleaner than it is. Each photo in the filmstrip below has a
"Correct" button for exactly this: it opens the same box editor the Demo
page's detail view already uses, writing to the same data/corrections/
store, so a correction here counts as this photo's ground truth in both
charts and thumbnails immediately, not just a note for a future retrain.

A photo's date/time (and caption) aren't locked in at upload either --
each filmstrip card also has an "Edit date & time" toggle for fixing a
wrong date after the fact, or nudging a photo earlier/later/into the
future for a demo narrative, without re-uploading it (see
timeline_helpers.update_meta). Re-uploading the same file works too --
it's still matched by content hash, so it's treated as a retag of that
existing entry (prefilled with its current date/time/caption below,
easy to change) rather than silently ignored or duplicated.
"""

import datetime as dt
import hashlib

import plotly.graph_objects as go
import streamlit as st

import annotate_helpers as ah
import detector
import timeline_helpers as th
import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("HISTORICAL DATA", detector.V26_LABEL), unsafe_allow_html=True)

# Pinned to this specific run rather than whatever HIVIS_MODEL_PATH the rest
# of the app defaults to (detector.WEIGHTS_PATH) -- see detector.py's own
# comment on V26_WEIGHTS/yolo26s_css_100e for why this run over the others.
# Note this only affects photos detected from here on: raw detections are
# frozen into data/timeline/ at save time (see timeline_helpers.save_entry),
# so anything already saved keeps whichever model produced it -- re-upload a
# photo to redetect it with this one.
model = detector.load_model(detector.V26_WEIGHTS)
if model is None:
    st.error(f"No trained weights found at `{detector.V26_WEIGHTS}`. See the Demo page for how to restore them.")
    st.stop()



_ALL_SLOTS = ("hardhat", "vest", "mask", "gloves", "boots")  # fixed order, matches pages/demo.py

# Human-readable violation names for the chart below -- detector.CLASS_META's
# own labels ("NO-Hardhat", "NO-Safety Vest") are built for the box editor's
# on-image tags, not prose/legend text.
_VIOLATION_LABELS = {
    "hardhat": "Missing hardhat",
    "vest": "Missing safety vest",
    "mask": "Missing mask",
    "gloves": "Missing gloves",
    "boots": "Missing boots",
}


def _fmt_dt(d):
    """Full ISO datetime -> "2026-08-31 14:30" for hover text and the
    filmstrip -- readable regardless of whether a time was actually set
    (midnight for anything saved before per-photo times existed)."""
    return d.strftime("%Y-%m-%d %H:%M") if d is not None else None


# ---------------------------------------------------------------------------
# 1. the saved timeline itself -- charts up front, so the trend is what's
# on screen when this page loads. Uploading, correcting, and retagging
# photos moved into the "MANAGE PHOTOS" toggle at the bottom (see below)
# instead of competing with the charts for the top of the page.
# ---------------------------------------------------------------------------

manifest = th.load_manifest()

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
    ai_assessment = detector.assess(raw, threshold, required=required)
    # A manual correction (see the editor below) is this photo's ground
    # truth once one exists -- same rule the Demo page already applies to
    # its own live batch (see pages/demo.py's own it["assessment"] logic).
    # Corrections are keyed by filename, same store the Demo page's editor
    # already writes to (data/corrections/), not a second one just for
    # this page -- so correcting a photo here also benefits a future
    # retraining run, same as correcting it there would.
    corrected_boxes = ah.load_existing_correction(meta.get("name", ""))
    if corrected_boxes:
        assessment = detector.assess(ah.boxes_to_raw(corrected_boxes), 0.0, required=required)
    else:
        assessment = ai_assessment
    persons = assessment["persons"]
    n = len(persons)
    compliant = sum(1 for p in persons if p["verdict"] == "ok")
    row = {
        "key": key, "name": meta.get("name", ""), "caption": meta.get("caption", ""),
        "date_str": meta.get("date", ""), "raw": raw, "n_persons": n,
        "compliance_pct": round(compliant / n * 100, 1) if n else None,
        "missing_counts": {s: sum(1 for p in persons if p["status"][s]["state"] == "missing") for s in _ALL_SLOTS},
        "verdict": assessment["verdict"],
        "assessment": assessment, "ai_assessment": ai_assessment,
        "corrected_boxes": corrected_boxes, "is_corrected": corrected_boxes is not None,
    }
    try:
        # stored as a full ISO date+time now ("2026-08-31T14:30:00"), but
        # datetime.fromisoformat also happily parses a bare date ("2026-08-31",
        # midnight) -- keeps this reading any entry saved before per-photo
        # times existed, no migration needed.
        row["dt"] = dt.datetime.fromisoformat(row["date_str"])
    except ValueError:
        row["dt"] = None
    row["date"] = row["dt"].date() if row["dt"] is not None else None  # day-bucket, used by both charts below
    (rows if row["dt"] is not None else invalid).append(row)

rows.sort(key=lambda r: r["dt"])  # exact time, so same-day batches order correctly within their day

# Signature of everything that can move a point on either chart below
# (which day/time it's bucketed into, its caption). Forces a fresh
# st.plotly_chart component identity whenever a metadata edit changes this,
# so an edited date/time is never masked by the browser reusing the
# previous chart render.
_rows_sig = hashlib.md5(
    "|".join(f"{r['key']}:{r['date_str']}:{r['caption']}" for r in rows + invalid).encode()
).hexdigest()[:12]

by_key = {r["key"]: r for r in rows + invalid}

if not manifest:
    st.info("No timeline entries yet. Open **MANAGE PHOTOS** below to upload some and get started.")
else:
    if invalid:
        st.warning(f"{len(invalid)} timeline entr{'y has' if len(invalid) == 1 else 'ies have'} an unreadable date "
                   f"and are excluded from the chart below (still shown in the filmstrip).")



    # One point per date, at that date's mean compliance -- click a point to
    # drill into that day's individual photos below (see the day-detail
    # panel further down).
    mean_by_date = {}
    for r in rows:
        if r["compliance_pct"] is not None:
            mean_by_date.setdefault(r["date"], []).append(r["compliance_pct"])
    dates_sorted = sorted(mean_by_date)
    # Anchored at midday rather than midnight -- one point per day, so
    # midday just centers it visually within that day's span on the axis.
    mean_x = [dt.datetime.combine(d, dt.time(12, 0)) for d in dates_sorted]
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

    # Fixed min-height (not height:100%, which only fills its own column's
    # content box and does nothing to equalize across the three -- Streamlit
    # columns don't stretch to match a sibling's height on their own) so all
    # three tiles line up even though DATE RANGE has one less line of
    # content than the other two. A bit of top margin on each subtitle line
    # gives the big number some breathing room instead of sitting flush
    # against the text underneath it.
    _TILE_MIN_HEIGHT = "118px"
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"""
        <div style="background:#141414;color:#FFFFFF;padding:16px 20px 14px;min-height:{_TILE_MIN_HEIGHT}">
          <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#9B9D97">PHOTOS IN TIMELINE</div>
          <div class="hv-h1" style="font-size:40px;line-height:1;color:#FFFFFF">{len(manifest)}</div>
          <div style="font-size:12px;color:#9B9D97;margin-top:10px">{len(rows)} dated · {len(invalid)} unreadable date</div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:16px 20px 14px;min-height:{_TILE_MIN_HEIGHT}">
          <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">DATE RANGE</div>
          <div class="hv-h1" style="font-size:22px;line-height:1.3;margin-top:10px">{date_range_txt}</div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:16px 20px 14px;min-height:{_TILE_MIN_HEIGHT}">
          <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:#71736D">COMPLIANCE CHANGE</div>
          <div class="hv-h1" style="font-size:40px;line-height:1">{delta_txt}</div>
          <div style="font-size:12px;color:#71736D;margin-top:10px">{delta_sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # chart 1 -- compliance % trend (its own axis: 0-100%, nothing else sharing it)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="hv-h1" style="font-size:16px;margin-bottom:2px">COMPLIANCE OVER TIME</div>', unsafe_allow_html=True)
    st.caption("One dot per day, at that day's mean compliance -- click a day to see its "
               "photos and analysis below.")

    day_photo_count = {d: len(mean_by_date[d]) for d in dates_sorted}
    day_text = [
        f"{d.isoformat()}<br>{mean_y[i]}% compliant (mean of {day_photo_count[d]} photo"
        + ("s" if day_photo_count[d] != 1 else "") + ")"
        for i, d in enumerate(dates_sorted)
    ]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=mean_x, y=mean_y, mode="lines+markers",
        line=dict(color="#14213D", width=2),
        marker=dict(size=10, color="#14213D", line=dict(color="#FFFFFF", width=2)),
        text=day_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
        # One point per day, so a click's customdata is that day's isoformat --
        # read back below to know which day's detail panel to open.
        customdata=[[d.isoformat()] for d in dates_sorted],
    ))
    fig1.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#141414", size=12.5),
        xaxis=dict(showgrid=False, linecolor="#C4C6C0", tickfont=dict(color="#4A4B47")),
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#E4E5E2", gridwidth=1,
                   zeroline=False, tickfont=dict(color="#4A4B47")),
    )
    fig1_event = st.plotly_chart(
        fig1, use_container_width=True, config={"displayModeBar": False}, key=f"tl_fig1_{_rows_sig}",
        on_select="rerun", selection_mode=["points"],
    )

    # A click toggles: clicking the same day's dot again closes the detail
    # panel, clicking a different day's dot switches straight to it. Only
    # points carrying customdata (the marker trace) count -- the mean line
    # is hoverinfo="skip"/unselectable but this guards it either way.
    clicked_day = None
    for pt in fig1_event.selection.points:
        cd = pt.get("customdata")
        if cd:
            clicked_day = cd[0]
            break
    if clicked_day is not None:
        st.session_state["_tl_selected_day"] = (
            None if st.session_state.get("_tl_selected_day") == clicked_day else clicked_day
        )

    selected_day_str = st.session_state.get("_tl_selected_day")
    if selected_day_str:
        selected_day = dt.date.fromisoformat(selected_day_str)
        # `rows`, not `dated_rows` -- a photo with no person detected above
        # threshold has compliance_pct=None (excluded from the mean, since
        # there's nothing to average), but it still belongs to this day and
        # should still show up here, badged "Not assessed" like everywhere
        # else in the app, rather than silently vanishing from the grid.
        day_rows = [r for r in rows if r["date"] == selected_day]
        with st.container(border=True):
            dc1, dc2 = st.columns([6, 1])
            with dc1:
                st.markdown(
                    f'<div class="hv-h1" style="font-size:16px">{selected_day.isoformat()} '
                    f'<span class="hv-mono" style="font-size:12px;color:#71736D;font-weight:400">'
                    f'{len(day_rows)} photo{"s" if len(day_rows) != 1 else ""}</span></div>',
                    unsafe_allow_html=True,
                )
            with dc2:
                if st.button("✕ Close", key="tl_close_day_detail", width="stretch"):
                    st.session_state["_tl_selected_day"] = None
                    st.rerun()

            # compliance analysis for the day: overall rate + missing-by-type,
            # same slot vocabulary/order as the VIOLATIONS BY TYPE chart below.
            day_persons = sum(r["n_persons"] for r in day_rows)
            day_compliant = sum(round(r["compliance_pct"] / 100 * r["n_persons"])
                                 for r in day_rows if r["compliance_pct"] is not None)
            day_pct = round(day_compliant / day_persons * 100, 1) if day_persons else None
            ac1, ac2 = st.columns([1, 2])
            with ac1:
                st.markdown(f"""
                <div style="background:#141414;color:#FFFFFF;padding:14px 18px;height:100%">
                  <div class="hv-mono" style="font-size:10px;letter-spacing:1.5px;color:#9B9D97">DAY COMPLIANCE</div>
                  <div class="hv-h1" style="font-size:32px;line-height:1;color:#FFFFFF">{day_pct if day_pct is not None else '—'}{'%' if day_pct is not None else ''}</div>
                  <div style="font-size:11.5px;color:#9B9D97">{day_compliant}/{day_persons} persons compliant</div>
                </div>""", unsafe_allow_html=True)
            with ac2:
                missing_rows_html = ""
                for slot in _ALL_SLOTS:
                    count = sum(r["missing_counts"][slot] for r in day_rows)
                    if count == 0:
                        continue
                    neg_key = detector.SLOT_ITEMS[slot][1]
                    color = detector.CLASS_META[neg_key]["color"]
                    missing_rows_html += (
                        f"<div style='display:flex;justify-content:space-between;padding:5px 12px;"
                        f"border-bottom:1px solid #F0F1EC;font-size:12.5px'>"
                        f"<span><span style='display:inline-block;width:9px;height:9px;background:{color};"
                        f"margin-right:8px;border-radius:1px'></span>{_VIOLATION_LABELS[slot]}</span>"
                        f"<span class='hv-mono' style='font-weight:600'>{count}</span></div>"
                    )
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #C4C6C0;height:100%">
                  <div class="hv-mono" style="font-size:10px;letter-spacing:1.5px;color:#71736D;padding:8px 12px 0">
                    MISSING PPE THIS DAY</div>
                  {missing_rows_html or "<div style='padding:8px 12px;font-size:12.5px;color:#71736D'>Nothing missing -- fully compliant.</div>"}
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # images with labels (detail=True -- same treatment as the Demo
            # page's single-photo detail view: thicker outlines, class name
            # + confidence baked onto each box), not the unlabeled thumbnail
            # style the filmstrip below uses.
            day_cols = st.columns(min(len(day_rows), 3) or 1)
            for i, r in enumerate(day_rows):
                img = th.load_image(r["key"])
                if img is None:
                    continue
                labeled = vh.draw_overlay(img, r["assessment"]["persons"], show_boxes=True,
                                           show_labels=True, detail=True)
                with day_cols[i % len(day_cols)]:
                    st.image(labeled, width="stretch")
                    st.markdown(
                        f'<div class="hv-mono" style="font-size:10.5px;color:#4A4B47">{_fmt_dt(r["dt"])}</div>'
                        f'{vh.verdict_badge(r["assessment"]["verdict"])}'
                        + (f'<div style="font-size:11px;color:#71736D">{r["caption"]}</div>' if r["caption"] else ""),
                        unsafe_allow_html=True,
                    )

    # ---------------------------------------------------------------------------
    # chart 2 -- violation-type breakdown (its own separate chart/axis -- counts,
    # never forced onto the % axis above)
    # ---------------------------------------------------------------------------

    tracked_slots = [s for s in _ALL_SLOTS if any(
        b["key"] in detector.SLOT_ITEMS[s] for r in dated_rows for b in r["raw"]
    )]

    if tracked_slots:
        st.markdown('<div class="hv-h1" style="font-size:16px;margin:22px 0 2px">VIOLATIONS BY TYPE</div>', unsafe_allow_html=True)
        st.caption("Count of missing-PPE findings per date, by item type -- grouped bars side by "
                   "side (not stacked), so each type's count is read straight off its own bar height.")

        by_date_slot = {}
        for r in dated_rows:
            acc = by_date_slot.setdefault(r["date"], {s: 0 for s in tracked_slots})
            for s in tracked_slots:
                acc[s] += r["missing_counts"][s]

        max_count = max((by_date_slot[d][s] for d in dates_sorted for s in tracked_slots), default=0)
        # Clean integer ticks for a small-count series (0,1,2,...) rather than
        # Plotly's own auto-picked fractional ticks (0.2/0.4/...) for a max of 1.
        dtick = 1 if max_count <= 12 else max(1, round(max_count / 8))

        fig2 = go.Figure()
        for slot in tracked_slots:  # fixed order (matches _ALL_SLOTS) -- never cycled/reordered
            neg_key = detector.SLOT_ITEMS[slot][1]
            color = detector.CLASS_META[neg_key]["color"]
            label = _VIOLATION_LABELS[slot]
            y = [by_date_slot[d][slot] for d in dates_sorted]
            fig2.add_trace(go.Bar(
                x=dates_sorted, y=y, name=label, marker=dict(color=color),
                hovertemplate=f"{label}: " + "%{y}<extra></extra>",
            ))
        fig2.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            barmode="group", bargap=0.25, bargroupgap=0.06,
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#141414", size=12.5),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=False, linecolor="#C4C6C0", tickfont=dict(color="#4A4B47")),
            yaxis=dict(gridcolor="#E4E5E2", gridwidth=1, zeroline=False, tickfont=dict(color="#4A4B47"),
                       rangemode="tozero", dtick=dtick, tick0=0),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key=f"tl_fig2_{_rows_sig}")
    else:
        st.caption("The loaded model hasn't detected any tracked PPE item across these photos yet -- "
                   "nothing to break down by type.")

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("Admin", expanded=not manifest):
    # ---------------------------------------------------------------------------
    # 2. upload + label new photos (not yet part of the saved timeline)
    # ---------------------------------------------------------------------------

    st.session_state.setdefault("_timeline_pending", {})
    pending = st.session_state._timeline_pending
    saved_keys = set(manifest.keys())  # `manifest` already loaded above, for the charts

    st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:6px">ADD PHOTOS</div>', unsafe_allow_html=True)
    st.caption("Re-uploading a photo that's already in the timeline lets you retag its date, time, or "
               "caption -- same content-hash key (see timeline_helpers.save_entry), so it updates that "
               "entry in place rather than duplicating it.")
    uploaded_files = st.file_uploader(
        "Select CCTV / site photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        label_visibility="collapsed", key="historical_uploader",
    )
    for f in uploaded_files or []:
        raw_bytes = f.getvalue()
        key = hashlib.md5(raw_bytes).hexdigest()
        if key in pending:
            continue  # already waiting to be (re-)labeled below
        is_reupload = key in saved_keys
        img = vh.load_image(raw_bytes)
        # Already-saved photo: reuse its stored raw detections rather than
        # rerunning the model (retagging is about the date/caption, not the
        # detections) and pre-fill the form below with its current values.
        raw = th.load_raw(key) if is_reupload else detector.detect_raw(model, img)
        existing = manifest.get(key, {})
        pending[key] = {"image": img, "raw": raw, "name": f.name, "is_reupload": is_reupload,
                         "existing_date": existing.get("date", ""), "existing_caption": existing.get("caption", "")}

    if pending:
        st.caption(f"{len(pending)} photo{'s' if len(pending) != 1 else ''} awaiting a date and time before "
                   f"they're added to the timeline -- set both per photo so a whole day's batch (several "
                   f"photos, different times) orders and reads correctly:")
        to_save = []  # (key, datetime_value, caption_value) captured this run, used if the button below is clicked
        for key, p in list(pending.items()):
            try:
                existing_dt = dt.datetime.fromisoformat(p["existing_date"]) if p.get("existing_date") else None
            except ValueError:
                existing_dt = None
            default_date = existing_dt.date() if existing_dt else dt.date.today()
            default_time = (existing_dt.time().replace(second=0, microsecond=0) if existing_dt
                             else dt.datetime.now().time().replace(second=0, microsecond=0))

            c1, c2, c3, c4, c5 = st.columns([1, 1.4, 1.2, 2.4, 1])
            with c1:
                st.image(p["image"], width="stretch")
                if p.get("is_reupload"):
                    st.caption("Already in timeline -- retagging")
            with c2:
                # min/max deliberately wide open (not just Streamlit's default
                # ~10-year window either side of value) -- this page is also used
                # to pre-stage demo photos with a future date before the real day
                # arrives, so a future date is a legitimate, intentional input here,
                # never something to validate against.
                date_val = st.date_input("Date taken", value=default_date,
                                          min_value=dt.date(2000, 1, 1), max_value=dt.date(2100, 1, 1),
                                          key=f"tl_date_{key}")
            with c3:
                time_val = st.time_input("Time taken", value=default_time,
                                          step=300, key=f"tl_time_{key}")
            with c4:
                caption_val = st.text_input("Caption (optional)", value=p.get("existing_caption", ""), key=f"tl_caption_{key}",
                                             placeholder="e.g. Week 1, morning shift")
            with c5:
                if st.button("✕ Discard", key=f"tl_discard_{key}"):
                    pending.pop(key, None)
                    st.rerun()
            to_save.append((key, dt.datetime.combine(date_val, time_val), caption_val))

        if st.button(f"Save {len(pending)} photo{'s' if len(pending) != 1 else ''} to timeline",
                     key="tl_add_all", type="primary"):
            for key, dt_val, caption_val in to_save:
                p = pending[key]
                th.save_entry(key, p["image"], p["raw"], p["name"], dt_val.isoformat(), caption_val)
            st.session_state._timeline_pending = {}
            st.toast(f"Saved {len(to_save)} photo{'s' if len(to_save) != 1 else ''} to the timeline.")
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # 3. filmstrip -- every saved photo, in date order, click-free (see
    # model_compare.py's own "not clickable yet" note -- same scope call here)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="hv-h1" style="font-size:16px;margin-bottom:10px">PHOTOS</div>', unsafe_allow_html=True)

    film_cols = st.columns(4)
    for i, r in enumerate(rows + invalid):
        img = th.load_image(r["key"])
        if img is None:
            continue
        # Reuses the same correction-aware assessment the charts above were
        # built from (computed once, in the rows loop) -- a corrected photo's
        # thumbnail and verdict badge always match what it contributed to the
        # trend, instead of silently showing the raw model's read again here.
        assessment = r["assessment"]
        thumb = vh.draw_overlay(img, assessment["persons"], show_boxes=True, show_labels=False)
        with film_cols[i % 4]:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #C4C6C0;margin-bottom:10px" title="{r['name']}">
              <img src="data:image/jpeg;base64,{vh.b64_image(thumb, max_dim=320)}"
                   style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block"/>
              <div style="padding:6px 8px;display:flex;flex-direction:column;gap:4px">
                <span class="hv-mono" style="font-size:10.5px;color:#4A4B47">{_fmt_dt(r['dt']) or r['date_str'] or 'no date'}</span>
                {vh.verdict_badge(assessment["verdict"])}
                {'<span class="hv-mono" style="font-size:10px;color:#1B7A3D">✓ MANUALLY CORRECTED</span>' if r["is_corrected"] else ''}
                {f'<span style="font-size:11px;color:#71736D">{r["caption"]}</span>' if r["caption"] else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("✎ Correct", key=f"tl_correct_{r['key']}", width="stretch"):
                    st.session_state["_tl_editing_key"] = r["key"]
                    st.rerun()
            with bc2:
                if st.button("✕ Remove", key=f"tl_remove_{r['key']}", width="stretch"):
                    th.delete_entry(r["key"])
                    st.rerun()
            if st.button("🗓 Edit date & time", key=f"tl_editmeta_btn_{r['key']}", width="stretch"):
                st.session_state["_tl_editing_meta_key"] = (
                    None if st.session_state.get("_tl_editing_meta_key") == r["key"] else r["key"]
                )
                st.rerun()
            if st.session_state.get("_tl_editing_meta_key") == r["key"]:
                # A wrong date at upload, or backdating/future-dating a photo
                # for a demo narrative -- fixed here without re-uploading the
                # photo. Same wide-open min/max as the upload form above:
                # a future date is a legitimate, intentional input on this page.
                cur_dt = r["dt"] or dt.datetime.now()
                with st.form(key=f"tl_meta_form_{r['key']}"):
                    new_date = st.date_input("Date taken", value=cur_dt.date(),
                                              min_value=dt.date(2000, 1, 1), max_value=dt.date(2100, 1, 1),
                                              key=f"tl_editdate_{r['key']}")
                    new_time = st.time_input("Time taken",
                                              value=cur_dt.time().replace(second=0, microsecond=0),
                                              step=300, key=f"tl_edittime_{r['key']}")
                    new_caption = st.text_input("Caption (optional)", value=r["caption"],
                                                 key=f"tl_editcaption_{r['key']}")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        save_clicked = st.form_submit_button("Save", type="primary", width="stretch")
                    with fc2:
                        cancel_clicked = st.form_submit_button("Cancel", width="stretch")
                if save_clicked:
                    new_dt = dt.datetime.combine(new_date, new_time)
                    th.update_meta(r["key"], date_str=new_dt.isoformat(), caption=new_caption)
                    st.session_state["_tl_editing_meta_key"] = None
                    st.toast("Date updated.")
                    st.rerun()
                if cancel_clicked:
                    st.session_state["_tl_editing_meta_key"] = None
                    st.rerun()

    # ---------------------------------------------------------------------------
    # 4. manual correction editor -- full page width (not squeezed into a
    # filmstrip cell), reusing the exact same box editor the Demo page's detail
    # view uses, so it looks and behaves like something already familiar rather
    # than a second, different editor. Relying on the model alone for this
    # page defeats the point of a *hand-curated* demo timeline -- this is the
    # override the rows loop above already checks for via
    # ah.load_existing_correction, on every rerun, so saving here immediately
    # updates both charts and every thumbnail, not just this one.
    # ---------------------------------------------------------------------------

    editing_key = st.session_state.get("_tl_editing_key")
    if editing_key is not None:
        editing_row = by_key.get(editing_key)
        st.markdown("<hr>", unsafe_allow_html=True)
        if editing_row is None:
            st.session_state["_tl_editing_key"] = None
        else:
            img = th.load_image(editing_key)
            st.markdown(
                f'<div class="hv-h1" style="font-size:16px;margin-bottom:4px">CORRECTING: {editing_row["name"]}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Done correcting", key="tl_done_correcting"):
                st.session_state["_tl_editing_key"] = None
                st.rerun()
            it = {
                "key": f"tl_{editing_key}", "image": img, "name": editing_row["name"],
                "corrected_boxes": editing_row["corrected_boxes"],
            }
            ah.render_editor(it, editing_row["ai_assessment"], threshold)
