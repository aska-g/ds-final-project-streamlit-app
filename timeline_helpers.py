"""HI-VIS — historical/CCTV timeline persistence (pages/historical.py).

Lets a batch of CCTV stills, each hand-labeled with the date it was taken,
get saved as a small "compliance over time" dataset -- images/ + raw
detections/ + a manifest, the same data/ layout every other dataset in
this repo uses (see data/merged/, data/corrections/).

Deliberately mirrors annotate_helpers.py's corrections persistence:
content-hash keyed (the same md5-of-file-bytes key used everywhere else in
this app, e.g. pages/demo.py's upload cache) so saving the same photo twice
updates its date/caption in place rather than duplicating it, and what's
persisted is the RAW detections, not a baked compliance verdict -- so the
trends page can re-run detector.assess() at read time against whatever
threshold / WHAT COUNTS AS COMPLIANT rule is currently set on the Demo
page, exactly like model_compare.py already does for the live batch,
instead of freezing today's rule permanently into every past photo.

Kept free of any Streamlit import, same reasoning as view_helpers.py: pure
data-layer functions are easier to unit-test and reuse on their own.
"""

import csv
import json
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
TIMELINE_DIR = REPO_ROOT / "data" / "timeline"
TIMELINE_IMAGES = TIMELINE_DIR / "images"
TIMELINE_RAW = TIMELINE_DIR / "raw"
MANIFEST_PATH = TIMELINE_DIR / "manifest.csv"

_FIELDS = ["key", "name", "date", "caption", "saved_at"]


def load_manifest():
    """-> {key: {"name", "date", "caption", "saved_at"}}, empty if nothing
    saved yet. One row per unique photo (content-hash keyed) -- this
    represents the current curated timeline, not an append-only history
    log, so re-saving the same photo updates its row in place."""
    if not MANIFEST_PATH.exists():
        return {}
    out = {}
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("key"):
                out[row["key"]] = {k: row.get(k, "") for k in _FIELDS if k != "key"}
    return out


def _write_manifest(entries):
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for key, row in entries.items():
            writer.writerow({"key": key, **row})


def save_entry(key, image, raw_boxes, name, date_str, caption):
    """Persist one CCTV photo into the timeline: the image itself, its raw
    (not-yet-threshold-filtered) detections, and a manifest row carrying
    the date/caption assigned to it. `key` is the same content-hash key
    the caller already computed for its own upload-dedup step (see
    pages/historical.py) -- passed in rather than recomputed here."""
    TIMELINE_IMAGES.mkdir(parents=True, exist_ok=True)
    TIMELINE_RAW.mkdir(parents=True, exist_ok=True)

    image.convert("RGB").save(TIMELINE_IMAGES / f"{key}.jpg", format="JPEG", quality=92)
    (TIMELINE_RAW / f"{key}.json").write_text(json.dumps(raw_boxes))

    entries = load_manifest()
    entries[key] = {
        "name": name,
        "date": date_str,
        "caption": caption or "",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_manifest(entries)


def delete_entry(key):
    """Remove one photo from the timeline entirely -- image, raw
    detections, and its manifest row. No undo, same as this app's other
    destructive actions; nothing else references a timeline entry once
    it's out of the manifest."""
    entries = load_manifest()
    entries.pop(key, None)
    _write_manifest(entries)
    (TIMELINE_IMAGES / f"{key}.jpg").unlink(missing_ok=True)
    (TIMELINE_RAW / f"{key}.json").unlink(missing_ok=True)


def load_raw(key):
    """Raw detections for a saved timeline entry -- the same {"key",
    "conf", "box"} shape detector.detect_raw() produces, ready to pass
    straight into detector.assess(). [] if the file's missing (manifest
    row surviving a deleted/moved raw file some other way) rather than a
    crash -- callers just see zero persons/violations for that entry."""
    path = TIMELINE_RAW / f"{key}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_image(key):
    """Saved timeline photo as a PIL image, or None if it's missing."""
    path = TIMELINE_IMAGES / f"{key}.jpg"
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")
