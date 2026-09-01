"""HI-VIS — presentation helpers for the demo page: drawing detection overlays,
building the exception-log rows, and CSV export. Kept free of any Streamlit
imports so it can be unit-tested (and reused by other pages) on its own.
"""

import base64
import functools
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.ExifTags import TAGS

import detector

_FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "IBMPlexMono-Bold.ttf"

# SuperVisor logo. Read from assets/ so the SVG stays a single source of truth,
# and inlined into the page HTML because Streamlit markdown cannot reference
# repo files. The source svg's own width/height attrs are dropped so the CSS
# size below wins.
_LOGO_SVG_SRC = (Path(__file__).resolve().parent / "assets" / "supervisor-logo.svg").read_text().strip()
LOGO_SVG = ('<svg viewBox="0 0 531 98" role="img" aria-label="SuperVisor" '
            'style="display:block;height:28px;width:152px;flex:none;aspect-ratio:531/98">'
            + _LOGO_SVG_SRC[_LOGO_SVG_SRC.index(">") + 1:])


@functools.lru_cache(maxsize=None)
def _label_font(size):
    """Bold monospace font for on-box labels (matches the mono used
    elsewhere in the app for confidence numbers), cached per size. Falls
    back to PIL's tiny built-in bitmap font — still functional, just less
    legible — if the bundled TTF is ever missing, so a checkout without
    assets/ doesn't crash the whole overlay."""
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except Exception:
        return ImageFont.load_default()


def _draw_label(draw, x, y, text, fill_color, alpha, font, align="right", detail=False):
    """A label chip straddling the point (x, y) — y is meant to be the top
    edge of the box the label belongs to, so the chip's bottom half
    overlaps into the box and its top half sits above it, like a name tag
    clipped onto a lapel. align="right" grows the chip rightward from x
    (item labels — items sit inset within their person's box, so this
    never collides with anything); align="left" grows it leftward instead,
    hanging just outside the box to its left (used for the person tag, so
    it clears the item label that may start at almost the same corner).

    Two color treatments:
    - thumbnail (detail=False): a solid chip in the class's own color with
      brand-black text — compact, reads fine at the small sizes it's used
      at (results gallery isn't allowed to call this at all anymore; kept
      for anything else that still wants the classic look).
    - detail (detail=True): a dark translucent chip with the TEXT itself
      in the class's own color, plus a thin border in that color. Keeps
      the label legible against any photo background and stops the chip
      from visually fusing into a same-colored box outline sitting right
      beneath it — the failure mode of a solid same-color chip on a
      same-color box.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = max(3, round(text_h * 0.45)), max(2, round(text_h * 0.28))
    chip_h = text_h + 2 * pad_y
    chip_w = text_w + 2 * pad_x
    x0 = x if align == "right" else x - chip_w
    chip = (x0, y - chip_h / 2, x0 + chip_w, y + chip_h / 2)
    if detail:
        draw.rectangle(chip, fill=_hex_to_rgba("#141414", min(alpha, 235)),
                        outline=_hex_to_rgba(fill_color, alpha), width=1)
        text_fill = _hex_to_rgba(fill_color, alpha)
    else:
        draw.rectangle(chip, fill=_hex_to_rgba(fill_color, alpha))
        text_fill = _hex_to_rgba("#141414", alpha)
    draw.text((chip[0] + pad_x - bbox[0], chip[1] + pad_y - bbox[1]), text,
               font=font, fill=text_fill)

VERDICT_META = {
    "ok":   {"label": "COMPLIANT",     "bg": "#FFFFFF", "fg": "#141414", "border": "#141414"},
    "non":  {"label": "NON-COMPLIANT", "bg": "#EFE600", "fg": "#141414", "border": "#141414"},
    "none": {"label": "NOT ASSESSED",  "bg": "#E4E5E2", "fg": "#4A4B47", "border": "#9B9D97"},
}


def load_image(file_bytes):
    """Open + auto-rotate per EXIF orientation, so overlay boxes (computed on
    the rotated image) line up with what's actually displayed."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def exif_datetime(file_bytes):
    """Best-effort capture timestamp from EXIF. Returns None (never a
    fabricated value) when the photo carries no timestamp — most
    screenshots, downloads, and messaging-app re-saves strip it."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        exif = img.getexif()
        if exif:
            for tag_id in (306,):  # base IFD "DateTime"
                if tag_id in exif and exif[tag_id]:
                    return str(exif[tag_id])
            try:
                sub = exif.get_ifd(0x8769)  # Exif SubIFD
                for tag_id in (0x9003, 0x9004):  # DateTimeOriginal, DateTimeDigitized
                    if tag_id in sub and sub[tag_id]:
                        return str(sub[tag_id])
            except Exception:
                pass
    except Exception:
        pass
    return None


def _hex_to_rgba(hexstr, alpha):
    hexstr = hexstr.lstrip("#")
    r, g, b = int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16)
    return (r, g, b, alpha)


def _box_px(box, w, h):
    x1, y1, x2, y2 = box
    return (x1 * w, y1 * h, x2 * w, y2 * h)


def draw_overlay(image, persons, selected_idx=None, show_boxes=True, show_labels=True, detail=False):
    """Return a copy of `image` with person + item bounding boxes drawn on
    it. `persons` is the list from detector.assess()["persons"]. If
    selected_idx is set, every other person's boxes and labels are dimmed.

    show_labels=False skips the on-box text chips entirely (box outlines
    only) — used for small thumbnails (results gallery, model comparison
    tiles) where the baked-in text shrinks past legible once the whole
    image is downscaled to thumbnail size.

    detail=True is the single-photo detail-view styling: thicker outlines
    and the dark-chip/colored-text label treatment (see _draw_label) —
    meant for the much larger image shown there, not for thumbnails.
    """
    base = image.convert("RGBA")
    if not show_boxes or not persons:
        return base.convert("RGB")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_scale = 1.6 if detail else 1.0
    item_width = max(3, round(min(w, h) / 130 * box_scale))
    person_width = max(2, round(min(w, h) / 200 * box_scale))
    font = _label_font(max(13, round(min(w, h) / 42)))

    for i, p in enumerate(persons):
        dim = selected_idx is not None and selected_idx != i
        alpha = 70 if dim else 255
        person_color = detector.CLASS_META["person"]["color"]
        px1, py1, px2, py2 = _box_px(p["box"], w, h)
        draw.rectangle((px1, py1, px2, py2), outline=_hex_to_rgba(person_color, alpha), width=person_width)
        if show_labels:
            _draw_label(draw, px1, py1, f"P{i + 1}", person_color, alpha, font, align="left", detail=detail)

        for slot in p["status"]:
            st_ = p["status"][slot]
            if st_["state"] == "notvisible" or not st_.get("box"):
                continue
            meta = detector.CLASS_META[st_["class_key"]]
            ix1, iy1, ix2, iy2 = _box_px(st_["box"], w, h)
            draw.rectangle((ix1, iy1, ix2, iy2), outline=_hex_to_rgba(meta["color"], alpha), width=item_width)
            if show_labels:
                label = f"{meta['label']} {st_['conf']:.2f}" if st_.get("conf") is not None else meta["label"]
                _draw_label(draw, ix1, iy1, label, meta["color"], alpha, font, detail=detail)

    return Image.alpha_composite(base, overlay).convert("RGB")


def badge_for(verdict):
    return VERDICT_META.get(verdict, VERDICT_META["none"])


def icon_svg(kind, color, size=12, stroke=3):
    paths = {
        "check": '<path d="M4 12.5l5 5L20 6.5"/>',
        "cross": '<path d="M5 5l14 14M19 5L5 19"/>',
        "warn": '<path d="M12 3L2 21h20L12 3z"/><path d="M12 10v5"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" style="vertical-align:-2px;flex:none">'
            f'{paths.get(kind, "")}</svg>')


def verdict_badge(verdict):
    meta = badge_for(verdict)
    kind = "check" if verdict == "ok" else "warn" if verdict == "non" else "cross"
    icon = icon_svg(kind, meta["fg"], 11, 2.6)
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:10.5px;'
            f'font-weight:700;letter-spacing:.5px;padding:2px 7px;background:{meta["bg"]};'
            f'color:{meta["fg"]};border:1px solid {meta["border"]};white-space:nowrap">'
            f'{icon}{meta["label"]}</span>')


def b64_image(img, max_dim=480, quality=82):
    im = img.copy()
    im.thumbnail((max_dim, max_dim))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def flag_confidence(assessment, required):
    """The confidence number shown on a result tile — always the same 0-1
    metric the DETECTION CONFIDENCE THRESHOLD slider itself filters on, so a
    tile is never left blank just because nothing was flagged:
      - non-compliant: the strongest violation's confidence (what drove the verdict)
      - compliant: the strongest person-detection confidence in the photo
      - not assessed: the best person confidence found, even below threshold
      - genuinely nothing detected at all: None (the only real "—" case)
    """
    confs = [p["status"][slot]["conf"] for p in assessment["persons"] for slot in required
             if p["status"][slot]["state"] == "missing"]
    if confs:
        return max(confs)
    person_confs = [p["conf"] for p in assessment["persons"]]
    if person_confs:
        return max(person_confs)
    return assessment.get("best_person_conf")


# ---------------------------------------------------------------------------
# shared page chrome — style block + header banner, used by every page so
# they all look like one app instead of drifting apart. No Streamlit import
# needed here: callers do st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True).
# ---------------------------------------------------------------------------

HV_STYLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#E4E5E2; }
#MainMenu { visibility:hidden; }
.block-container { max-width:1400px; padding-top:3rem !important; }
.hv-mono { font-family:'IBM Plex Mono',monospace; }
.hv-h1 { font-family:'Barlow Condensed',sans-serif; font-weight:800; letter-spacing:.5px; color:#141414; }
@keyframes hvspin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
@keyframes hvstripe { from { background-position:0 0; } to { background-position:28px 0; } }
[data-testid="stFileUploaderDropzone"] { background:#FFFFFF !important; border:2px dashed #141414 !important; border-radius:0 !important; }
[role="radiogroup"] label { border:1px solid #141414; padding:4px 12px; margin-right:0 !important; background:#FFFFFF; }
/* Buttons: rounded, with a hover lift/shadow and a press-down on click --
   the rest of the app stays hard-edged (that's the brand), but buttons are
   the one thing you physically click, so they get the tactile modern
   treatment: motion and elevation, not just a flat color swap. */
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    border-radius:8px !important; font-weight:600 !important;
    box-shadow:0 1px 2px rgba(20,20,20,0.06) !important;
    transition:transform 120ms ease, box-shadow 120ms ease, background-color 120ms ease, border-color 120ms ease !important;
}
[data-testid="stBaseButton-secondary"] { border:1px solid #C4C6C0 !important; background:#FFFFFF !important; color:#141414 !important; }
[data-testid="stBaseButton-primary"] { border:1px solid #141414 !important; background:#141414 !important; color:#FFFFFF !important; }
[data-testid="stBaseButton-secondary"]:hover { border-color:#141414 !important; box-shadow:0 4px 10px rgba(20,20,20,0.12) !important; transform:translateY(-1px); }
[data-testid="stBaseButton-primary"]:hover { background:#2A2B28 !important; box-shadow:0 4px 10px rgba(20,20,20,0.18) !important; transform:translateY(-1px); }
[data-testid="stBaseButton-secondary"]:active, [data-testid="stBaseButton-primary"]:active { transform:translateY(0); box-shadow:0 1px 2px rgba(20,20,20,0.08) !important; }
/* Borderless "toggle section" expander (e.g. MANAGE PHOTOS / Admin) -- the
   settings-gear icon (passed via st.expander(icon=...)) is the affordance
   now, so the boxed outline around it is just extra chrome. */
[data-testid="stExpander"] > details { border:none !important; background:transparent !important; }
/* Streamlit swaps the custom icon (icon=":material/settings:") for its own
   chevron on hover, regardless of the icon param -- the DOM element's own
   testid literally changes from stExpanderIcon to stIconMaterial when the
   mouse is over the summary bar. Pin the gear glyph in that slot instead
   of fighting the swap: hide whichever native icon is currently rendered
   and draw a fixed gear via ::before, so it never flips to a chevron. */
[data-testid="stExpander"] summary [data-testid="stExpanderIcon"],
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    display:none !important;
}
/* Streamlit paints its own full-row hover background on the summary bar --
   suppress that and instead give just the icon glyph a small, icon-sized
   hover target (like a real icon button), not the whole card width. */
[data-testid="stExpander"] summary:hover {
    background:transparent !important;
}
[data-testid="stExpander"] summary > span > span:first-child {
    display:inline-flex !important; align-items:center !important; justify-content:center !important;
    width:34px !important; height:34px !important; border-radius:6px !important;
    transition:background-color 120ms ease !important;
}
[data-testid="stExpander"] summary > span > span:first-child:hover {
    background:#C4C6C0 !important;
}
[data-testid="stExpander"] summary > span > span:first-child::before {
    content:"⚙" !important; font-family:initial !important; font-size:20px !important;
    color:#141414 !important; line-height:1 !important; display:inline-block !important;
}
[data-testid="stAlert"] { background:#FFFFFF !important; border:1px solid #C4C6C0 !important; border-radius:0 !important; }
[data-testid="stAlertContainer"] { background:#FFFFFF !important; color:#141414 !important; }
[data-testid="stAlertContentInfo"] { color:#141414 !important; }
[data-testid="stCaptionContainer"] { color:#4A4B47 !important; }
hr { border-color:#C4C6C0; }
/* Chart cards (COMPLIANCE OVER TIME / VIOLATIONS BY TYPE) -- want the
   white background + padding a bordered container gives, but Joanna does
   not want the border/rounded-corner card look, and st.container's own
   testid is shared by every bordered container in the app (including the
   legit day-detail-panel one), so target these two by their key= instead
   of the shared testid. */
.st-key-tl_chart1_card, .st-key-tl_chart2_card {
    background:#FFFFFF !important; border:none !important; border-radius:0 !important;
}

</style>
"""


def header_html(subtitle, model_label=None):
    """The black HI-VIS banner every page opens with. `subtitle` is the
    all-caps label next to the logo (e.g. "PPE COMPLIANCE DETECTION");
    `model_label` is optional and renders as a dim mono tag on the right."""
    model_bit = ""
    if model_label:
        model_bit = (f'<div class="hv-mono" style="font-size:11px;color:#8D8F8A;border-left:1px solid #3A3B38;'
                      f'padding-left:14px">{model_label}</div>')
    return f"""
    <div style="background:#141414;color:#FFFFFF;display:flex;align-items:center;gap:16px;
         padding:14px 24px;margin:0 0 20px 0;flex-wrap:wrap">
      {LOGO_SVG}
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:15px;
           letter-spacing:2.5px">{subtitle}</div>
      {model_bit}
    </div>
    """


def build_rows(items, threshold, rule_text, required=("hardhat", "vest")):
    """items: list of {"name", "datetime", "assessment"} — one per uploaded
    photo. Returns one row per assessed person per tracked item in
    `required` (any of hardhat/vest/gloves/boots), covering both compliant
    and non-compliant findings, so the verdict filter in the UI can slice
    either view from the same table. An item outside `required` never
    appears here even if the model found it missing — it just is not part
    of the rule."""
    rows = []
    for item in items:
        assessment = item["assessment"]
        for p_idx, p in enumerate(assessment["persons"]):
            missing = [slot for slot in required if p["status"][slot]["state"] == "missing"]
            if missing:
                for slot in missing:
                    st_ = p["status"][slot]
                    label = detector.CLASS_META[st_["class_key"]]["label"]
                    rows.append({
                        "file": item["name"], "datetime": item["datetime"] or "—",
                        "person": f"Person {p_idx + 1}", "type": slot,
                        "finding": f"{label} — absence detected",
                        "confidence": round(st_["conf"], 2), "verdict": "non-compliant",
                        "threshold": round(threshold, 2), "rule_set": rule_text,
                    })
            else:
                rows.append({
                    "file": item["name"], "datetime": item["datetime"] or "—",
                    "person": f"Person {p_idx + 1}", "type": "—",
                    "finding": "none — required PPE present",
                    "confidence": None, "verdict": "compliant",
                    "threshold": round(threshold, 2), "rule_set": rule_text,
                })
    return rows


def rows_to_csv(rows):
    import csv as _csv
    buf = io.StringIO()
    fields = ["file", "datetime", "person", "type", "finding", "confidence", "verdict", "threshold", "rule_set"]
    writer = _csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
