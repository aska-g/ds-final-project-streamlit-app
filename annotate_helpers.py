"""HI-VIS — manual correction / annotation helpers for the image detail view.

Lets a reviewer fix a wrong AI label directly on a photo — draw a box the
model missed, move or resize one it drew in the wrong place, delete a
spurious one, or just relabel its class — using an interactive canvas
(streamlit-drawable-canvas). This is deliberately Roboflow-Annotate-shaped:
select an object, correct it, save it.

Saved corrections are written as a small YOLO-format dataset under
data/corrections/ — images/ + labels/ + data.yaml, the same layout as every
other dataset under data/ (see data/merged/) — so a teammate can review
them and fold them into a future training run the same way data/merged was
assembled from other sources. The class list/order used here is fixed
(CORRECTION_CLASS_KEYS below) and independent of whichever trained model
produced the original detection, so corrections stay meaningful no matter
which run is loaded when they're made.
"""

import base64
import csv
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st

# streamlit-drawable-canvas (last released for an older numpy) references
# a few numpy aliases — bool8, object0, int0, uint0 — that numpy>=1.24
# removed. This project pins numpy==2.x, so shim them back in before the
# component's import runs, rather than pinning numpy down for one component.
for _old, _new in (("bool8", "bool_"), ("object0", "object_"), ("int0", "intp"), ("uint0", "uintp")):
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))

from streamlit_drawable_canvas import st_canvas  # noqa: E402  (must follow the numpy shim above)

import detector  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent  # this repo's root (flattened out of streamlit_app/)
CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections"
CORRECTIONS_IMAGES = CORRECTIONS_DIR / "images"
CORRECTIONS_LABELS = CORRECTIONS_DIR / "labels"
MANIFEST_PATH = CORRECTIONS_DIR / "manifest.csv"

# Fixed class order for the corrections dataset — sourced from
# detector.CLASS_META (the one place class names/colors are defined, per
# its own docstring) so this never drifts out of sync with the detector.
CORRECTION_CLASS_KEYS = list(detector.CLASS_META.keys())
CLASS_KEY_TO_ID = {k: i for i, k in enumerate(CORRECTION_CLASS_KEYS)}

_UNASSIGNED_COLOR = "#EFE600"  # safety yellow — used for a freshly drawn, not-yet-classed box


def _canvas_size(image, max_w=900):
    """Pick a display size for the canvas — capped at max_w so it doesn't
    blow past the column, never upscaled for a smaller photo. Kept fixed
    for the lifetime of an editing session on this photo so normalized
    (0-1) coordinates stay consistent between reruns."""
    w, h = image.size
    if w <= max_w:
        return w, h
    scale = max_w / w
    return max_w, max(1, round(h * scale))


def seed_boxes(assessment):
    """Build the starting corrected-box list from the AI's own detections —
    one entry per person box, plus one per item slot that was actually
    detected (present or missing) — 'not visible' slots have no box to
    seed, and are exactly the case a reviewer would draw a new box for."""
    boxes = []
    for p in assessment.get("persons", []):
        boxes.append({"class_key": "person", "box": p["box"], "source": "ai"})
        for slot, st_ in p["status"].items():
            if st_["state"] == "notvisible" or not st_.get("box"):
                continue
            boxes.append({"class_key": st_["class_key"], "box": st_["box"], "source": "ai"})
    return boxes


_BOX_STROKE_WIDTH = 4  # was 3 -- still thinner than read-mode's baked-in boxes (7-11px at this
                        # canvas size), deliberately: this stroke also has to stay precisely
                        # draggable/resizable, which a much heavier line makes harder to judge.


def _image_to_data_uri(image):
    """PIL image -> a self-contained data: URI. Embedding the editor's
    background photo straight into the Fabric.js JSON this way (see
    boxes_to_fabric below) means the browser never has to fetch it
    separately at all -- no server-side registration, nothing to garbage
    collect, nothing that can 404. That ruled out two earlier approaches:
    Streamlit's MediaFileManager purges a session's file registrations at
    the start of every full rerun and only keeps what gets re-touched
    *during* that same run, so any rerun that didn't re-render this exact
    photo's editor (viewing a different photo, any widget elsewhere on the
    page, the app waking from idle) silently purged it. A plain static
    file avoided that, but Streamlit Community Cloud's own docs are
    explicit that files an app *writes while running* aren't reliably
    served there -- only ones already committed to the repo -- which is
    why that worked locally and not in production.

    A data: URI passed here (rather than through st_canvas's own
    background_image= parameter) sidesteps both: that parameter always
    routes through Streamlit's real image_to_url(), and the canvas
    frontend unconditionally does `<app origin> + backgroundImageURL`
    itself, which corrupts a data: URI (confirmed by reading Streamlit's
    own component-loading JS -- it always appends `?streamlitUrl=<origin>`
    to a custom component's iframe URL). Fabric.js's own JSON-based
    background image (a plain `img.src = url` load, per its source) has no
    such prefixing, so a data: URI works there without any of this."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def boxes_to_fabric(boxes, canvas_w, canvas_h, background_image=None):
    """corrected-box list (normalized 0-1 xyxy) -> Fabric.js initial_drawing
    for st_canvas. AI-sourced boxes are drawn solid, hand-drawn ones dashed,
    so it's obvious at a glance which is which while you're correcting.

    background_image, if given, is embedded as Fabric's own native
    backgroundImage (see _image_to_data_uri) rather than passed through
    st_canvas's background_image= parameter -- see that function's
    docstring for why. Only needs setting once per fabric_key lifetime
    (the photo itself never changes for a given correction session), not
    recomputed on every rerun."""
    objects = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        meta = detector.CLASS_META.get(b["class_key"])
        color = meta["color"] if meta else _UNASSIGNED_COLOR
        obj = {
            "type": "rect",
            "left": x1 * canvas_w, "top": y1 * canvas_h,
            "width": max(2.0, (x2 - x1) * canvas_w), "height": max(2.0, (y2 - y1) * canvas_h),
            "fill": "rgba(0,0,0,0)", "stroke": color, "strokeWidth": _BOX_STROKE_WIDTH,
            "scaleX": 1, "scaleY": 1, "opacity": 1,
        }
        if b.get("source") == "manual":
            obj["strokeDashArray"] = [7, 5]
        objects.append(obj)
    drawing = {"version": "4.4.0", "objects": objects}
    if background_image is not None:
        # Fabric.js's Image object treats a declared width/height as a
        # crop window into the SOURCE image's natural resolution, not as
        # a target size to scale the whole image into (confirmed by
        # reading fabric's own source: _renderFill draws
        # drawImage(el, sX, sY, sW, sH, ...) where sW/sH are capped by
        # el's natural width/height, using this.width/this.height only to
        # size that window). _canvas_size() caps the display width at
        # 900px, so any wider photo (our CCTV stills are 1254x1254) would
        # get embedded at full natural resolution while this dict
        # declares the smaller canvas_w/canvas_h -- Fabric then rendered
        # only the top-left canvas_w x canvas_h crop of the real photo at
        # 1:1 scale, while the boxes above are correctly positioned for
        # the whole photo scaled down to canvas_w x canvas_h. Resizing
        # here so the embedded image's natural size already equals the
        # declared width/height makes that crop window a no-op.
        bg = background_image.convert("RGB").resize((int(canvas_w), int(canvas_h)))
        drawing["backgroundImage"] = {
            "type": "image",
            "src": _image_to_data_uri(bg),
            "left": 0, "top": 0, "originX": "left", "originY": "top",
            "width": canvas_w, "height": canvas_h, "scaleX": 1, "scaleY": 1,
        }
    return drawing


def boxes_to_raw(boxes):
    """corrected-box list -> the same {"key", "conf", "box"} shape
    detector.detect_raw() produces, at full confidence (these are now
    ground truth, not a model guess) — lets a saved correction be run back
    through detector.assess() so the corrected boxes drive the compliance
    verdict, overlay, exception log, everywhere it["assessment"] is used."""
    return [{"key": b["class_key"], "conf": 1.0, "box": b["box"]}
            for b in boxes if b["class_key"] is not None]


def _obj_to_box(obj, canvas_w, canvas_h, class_key, source):
    """One Fabric.js object (post-interaction) -> a normalized xyxy box,
    accounting for Fabric expressing a resize as scaleX/scaleY rather than
    changed width/height."""
    w = obj.get("width", 0) * obj.get("scaleX", 1)
    h = obj.get("height", 0) * obj.get("scaleY", 1)
    x1, y1 = obj.get("left", 0), obj.get("top", 0)
    x2, y2 = x1 + w, y1 + h
    return {
        "class_key": class_key,
        "box": (max(0.0, min(1.0, x1 / canvas_w)), max(0.0, min(1.0, y1 / canvas_h)),
                max(0.0, min(1.0, x2 / canvas_w)), max(0.0, min(1.0, y2 / canvas_h))),
        "source": source,
    }


def _write_data_yaml():
    names = "\n".join(f"- {detector.CLASS_META[k]['label']}" for k in CORRECTION_CLASS_KEYS)
    (CORRECTIONS_DIR / "data.yaml").write_text(
        f"# Auto-generated by annotate_helpers.save_correction — matches the\n"
        f"# images/ + labels/ layout every other dataset under data/ uses.\n"
        f"names:\n{names}\nnc: {len(CORRECTION_CLASS_KEYS)}\n"
        f"train: images\nval: images\n"
    )


def _append_manifest(source_name, stem, model_label, threshold, n_boxes):
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["source_file", "saved_stem", "saved_at", "model_label", "threshold", "n_boxes"])
        writer.writerow([source_name, stem, time.strftime("%Y-%m-%d %H:%M:%S"), model_label, f"{threshold:.2f}", n_boxes])


def _stem_for(source_name):
    """Filename stem a correction is keyed by — shared by save and load so
    a photo re-uploaded later is matched against its own saved correction."""
    return Path(source_name).stem.replace(" ", "_") or "photo"


def load_existing_correction(source_name):
    """If source_name already has a saved correction under
    data/corrections/labels/, load it back as a corrected-box list. Called
    when a photo is (re-)uploaded, so a fresh browser session — or just
    re-adding the same file to a new batch — picks up a correction you
    already saved instead of reverting to the raw AI output. Returns None
    if there's nothing on disk for this filename yet."""
    label_path = CORRECTIONS_LABELS / f"{_stem_for(source_name)}.txt"
    if not label_path.exists():
        return None
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if not (0 <= cid < len(CORRECTION_CLASS_KEYS)):
            continue
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes.append({
            "class_key": CORRECTION_CLASS_KEYS[cid],
            "box": (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))),
            "source": "ai",
        })
    return boxes


def save_correction(image, boxes, source_name, model_label, threshold):
    """Write the image + a YOLO-format label file into data/corrections/,
    and log a manifest row. A second save for the same source filename
    overwrites the image/label pair in place (manifest keeps every save as
    history) — corrections are meant to reflect your latest pass, not pile
    up as duplicates."""
    CORRECTIONS_IMAGES.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_LABELS.mkdir(parents=True, exist_ok=True)

    stem = _stem_for(source_name)
    img_path = CORRECTIONS_IMAGES / f"{stem}.jpg"
    label_path = CORRECTIONS_LABELS / f"{stem}.txt"

    image.convert("RGB").save(img_path, format="JPEG", quality=92)

    lines = []
    for b in boxes:
        if b["class_key"] is None:
            continue
        x1, y1, x2, y2 = b["box"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        cid = CLASS_KEY_TO_ID[b["class_key"]]
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    _write_data_yaml()
    _append_manifest(source_name, stem, model_label, threshold, len(lines))
    return img_path, label_path


def render_editor(it, assessment, threshold):
    """Render the interactive box editor for one photo in the detail view:
    draw new boxes, move/resize existing ones, delete or relabel any box,
    then save the corrected set as a YOLO label for later retraining."""
    key = it["key"]
    store_key = f"_corr_boxes_{key}"
    version_key = f"_corr_version_{key}"
    fabric_key = f"_corr_fabric_{key}"
    # Seed from a previously saved correction if one exists (this photo's
    # own truth), not always from the raw AI assessment — otherwise the
    # first time you open the editor in a fresh session/browser tab (before
    # store_key exists yet), any hand-drawn box you'd already saved would
    # silently be replaced by the unedited AI output. "Reset to AI
    # detections" below is the explicit, deliberate way back to the raw
    # baseline; opening the editor should never do that on your behalf.
    st.session_state.setdefault(store_key, it.get("corrected_boxes") or seed_boxes(assessment))
    st.session_state.setdefault(version_key, 0)
    boxes = st.session_state[store_key]

    canvas_w, canvas_h = _canvas_size(it["image"])

    # The canvas's own frontend reloads EVERYTHING from scratch whenever the
    # initial_drawing it receives isn't byte-identical to what it already
    # has — that's how it decides "did Python hand me a new seed". Recomputing
    # initial_drawing from `boxes` on every rerun (as this used to do) meant
    # the content drifted by float rounding on every round trip, so it never
    # matched and the canvas reloaded (visibly blinking) after every single
    # interaction — which also wiped out whatever you'd just drawn before you
    # could see it. So this is only (re)built at deliberate reset points
    # below; every other rerun hands the canvas back the exact same dict.
    # synced_key tracks whether the canvas has echoed back a clean,
    # exact match of whatever we last (re)built fabric_key from. Only once
    # that's happened do we trust a LATER report of *more* objects than
    # `boxes` as a box the user actually just drew (see the sync logic
    # below) -- a real hand-drawn box can only exist on top of a
    # confirmed baseline, so an oversized report before we've ever seen
    # one back is some frontend artifact (a mount race, a stale echo),
    # not a user action, and taking it at face value is exactly what
    # silently corrupted a photo's boxes with a batch of unclassed,
    # unrequested ones before this guard existed.
    synced_key = f"_corr_synced_{key}"
    if fabric_key not in st.session_state:
        st.session_state[fabric_key] = boxes_to_fabric(boxes, canvas_w, canvas_h, background_image=it["image"])
        st.session_state[synced_key] = False

    mc1, mc2, mc3 = st.columns([2, 2, 3])
    with mc1:
        mode_label = st.radio("Mode", ["Draw new box", "Move / resize"], key=f"mode_{key}",
                               label_visibility="collapsed")
    mode = "transform" if mode_label == "Move / resize" else "rect"
    with mc2:
        if st.button("Reset to AI detections", key=f"reset_{key}"):
            st.session_state[store_key] = seed_boxes(assessment)
            st.session_state[fabric_key] = boxes_to_fabric(
                st.session_state[store_key], canvas_w, canvas_h, background_image=it["image"])
            st.session_state[synced_key] = False
            st.session_state[version_key] += 1
            st.rerun()
    with mc3:
        st.caption(f"{len(boxes)} box{'es' if len(boxes) != 1 else ''} on this photo — solid outline is the "
                    f"model's, dashed is hand-drawn. Drag anywhere to add a new box; switch to Move / resize to reposition one.")

    # No labels baked into the photo here (that was tried -- see git
    # history -- and dropped: a baked-in label sat behind the live,
    # draggable box outlines one layer up, which always render on top
    # regardless of draw order, so a nearby box's edge could cut across a
    # label's text; and re-baking it after every drag/resize added a
    # visible extra rerun). Plain photo, colors + the class list/swatches
    # below the canvas are still how you tell boxes apart while editing.
    # The background itself is embedded straight into fabric_key above
    # (boxes_to_fabric's background_image=) rather than passed here via
    # background_image=, so there's nothing to compute on this rerun.
    result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=_BOX_STROKE_WIDTH,
        stroke_color=_UNASSIGNED_COLOR,
        height=canvas_h, width=canvas_w,
        drawing_mode=mode,
        initial_drawing=st.session_state[fabric_key],
        update_streamlit=True,
        key=f"canvas_{key}_{st.session_state[version_key]}",
    )

    if result is not None and result.json_data is not None:
        objects = result.json_data.get("objects", [])
        if len(objects) > len(boxes) and st.session_state.get(synced_key):
            # one or more new rectangles appended at the end, AND we've
            # already seen the canvas cleanly echo back our own current
            # seed at least once since the last (re)build of fabric_key --
            # so this genuinely is new, on top of a confirmed baseline, not
            # a stale/inflated report from a component that hasn't settled
            # yet (see synced_key's own comment above; that's what used to
            # let a handful of unrequested, unclassed boxes silently get
            # written into a photo's corrections the moment its editor was
            # opened). The canvas already has these live on screen, so this
            # doesn't need to force a reload right now — but it DOES need
            # to update our own bookkeeping (for the class list below and
            # for saving) AND fabric_key, or a newly hand-drawn box only
            # ever existed in this one browser tab's live canvas: the
            # moment the canvas component remounts (navigate away and back
            # to this photo, Prev/Next, a fresh session picking up an old
            # save) it reloads from fabric_key and the box is gone from the
            # picture — even though `boxes` (and the saved correction)
            # still has it, which is exactly the "the label isn't there
            # when I open it again" bug this used to cause. We extend
            # fabric_key with the exact objects the canvas just reported
            # (not a value recomputed from our own normalized box tuples,
            # which would drift by float rounding on every rerun and cause
            # the constant-reload/blink problem described below) — so this
            # is a one-time, harmless resync right after a box is drawn,
            # not a per-rerun one.
            n_before = len(boxes)
            for obj in objects[n_before:]:
                boxes.append(_obj_to_box(obj, canvas_w, canvas_h, class_key=None, source="manual"))
            st.session_state[fabric_key]["objects"].extend(objects[n_before:])
        elif len(objects) == len(boxes):
            # same boxes, sync any moved/resized geometry into our
            # bookkeeping only — fabric_key stays as-is so the canvas isn't
            # force-reloaded out from under an in-progress drag. This exact
            # match is also what confirms the canvas has caught up with our
            # current seed, so a *later* surplus report can be trusted.
            st.session_state[synced_key] = True
            for i, obj in enumerate(objects):
                boxes[i]["box"] = _obj_to_box(
                    obj, canvas_w, canvas_h, boxes[i]["class_key"], boxes[i]["source"])["box"]
        # (len(objects) > len(boxes) but not yet synced_key: deliberately
        # ignored, same reasoning as synced_key's own comment above — a box
        # can't have been genuinely hand-drawn before the canvas has ever
        # echoed our current seed back cleanly, so trusting it here is how
        # a handful of phantom, unclassed boxes used to get permanently
        # written into a fresh photo's session state the moment its editor
        # was opened. If this ever fires for a real fast double-click draw
        # right at mount, it isn't lost -- it's still live on the user's
        # own canvas and gets reported (and trusted) again on their very
        # next interaction, once synced_key has flipped True.
        # else (fewer objects reported than we're tracking): deliberately
        # ignored. The component reports the canvas's contents back to
        # Streamlit on every mouse-up, and that can fire from a plain click
        # before the async load of the seeded AI boxes has actually
        # finished — a real "0 objects" snapshot mid-load, not the user
        # clearing anything. Treating that as "the boxes are gone" (as
        # this used to) meant it got written back as the new seed and the
        # AI boxes never came back, which is why they disappeared and
        # every draw attempt afterward kept landing on that same emptied
        # state. Double-clicking a box in Move/resize mode does delete it
        # on the canvas (that's Fabric's own behavior), but as a result
        # isn't reflected back into this bookkeeping — use the ✕ button
        # below for a delete that's guaranteed to stick.

    if not boxes:
        st.info("No boxes yet. Switch to **Draw new box** and drag a rectangle over anything the model missed.")
    else:
        st.caption("Class per box — required before you can save:")
        options = [None] + CORRECTION_CLASS_KEYS
        for i, b in enumerate(list(boxes)):
            rc1, rc2, rc3 = st.columns([1, 6, 1])
            meta = detector.CLASS_META.get(b["class_key"])
            color = meta["color"] if meta else _UNASSIGNED_COLOR
            with rc1:
                st.markdown(f'<div style="width:20px;height:20px;background:{color};margin-top:8px;'
                             f'border:1px solid #141414"></div>', unsafe_allow_html=True)
            with rc2:
                sel = st.selectbox(
                    f"Box {i + 1} class",
                    options,
                    index=options.index(b["class_key"]) if b["class_key"] in options else 0,
                    format_func=lambda k: "— select class —" if k is None else detector.CLASS_META[k]["label"],
                    key=f"class_{key}_{i}_{st.session_state[version_key]}",
                    label_visibility="collapsed",
                )
                boxes[i]["class_key"] = sel
            with rc3:
                if st.button("✕", key=f"rm_{key}_{i}_{st.session_state[version_key]}"):
                    boxes.pop(i)
                    st.session_state[fabric_key] = boxes_to_fabric(boxes, canvas_w, canvas_h, background_image=it["image"])
                    st.session_state[synced_key] = False
                    st.session_state[version_key] += 1
                    st.rerun()

    unassigned = sum(1 for b in boxes if b["class_key"] is None)
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        if unassigned:
            st.warning(f"{unassigned} box{'es' if unassigned != 1 else ''} still need a class.")
        if st.button("💾 Save corrected labels", disabled=(unassigned > 0 or not boxes),
                      key=f"save_{key}", type="primary"):
            save_correction(it["image"], boxes, it["name"], detector.MODEL_LABEL, threshold)
            snapshot = [dict(b) for b in boxes]
            # st.session_state._detections is the Demo page's own upload-batch
            # cache (see pages/demo.py) -- syncing "corrected" into it there
            # keeps that page's gallery in sync without a rerun-triggered
            # reload. Other callers of this editor (e.g. the Historical Data
            # page) never populate that cache for their own item keys, so
            # only touch it here if this key is actually in it -- `it`
            # (updated right below, same as always) is what every caller
            # actually relies on for the result of this save.
            if key in st.session_state.get("_detections", {}):
                st.session_state._detections[key]["corrected"] = True
                st.session_state._detections[key]["corrected_boxes"] = snapshot
            it["corrected"] = True
            it["corrected_boxes"] = snapshot
            st.toast(f"Saved corrected labels for {it['name']} — now shown everywhere as this photo's truth.")
            st.rerun()
    with sc2:
        st.caption("Saved to `data/corrections/` as an image + YOLO label pair, "
                    "ready to fold into a future training run.")
