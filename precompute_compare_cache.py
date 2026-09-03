#!/usr/bin/env python3
"""Precompute raw detections for the Model Comparison page's two demo
models (see MODELS below) over the fixed demo photos in assets/images/, and
write them to assets/images/compare_cache.json.

pages/model_compare.py reads this cache instead of loading the models and
running inference itself on every page view/rerun — the whole point being
that opening the page (or moving the threshold slider, which reruns it) is
instant instead of waiting on YOLO each time.

Run this manually:

    python precompute_compare_cache.py

...whenever:
  - a photo is added, removed, or replaced in assets/images/
  - MERGED_M_V2_WEIGHTS or SUPERVISOR_V1_WEIGHTS in detector.py end up
    pointing at a retrained/different run

If the cache is missing entirely, or missing an entry for some image (e.g.
a new photo landed since the last run), pages/model_compare.py falls back
to running that one model live — slower, but the page still works. This
script just makes that fallback unnecessary day to day.
"""

import json

import detector
import view_helpers as vh

IMAGES_DIR = detector.REPO_ROOT / "assets" / "images"  # kept in sync with pages/model_compare.py
CACHE_PATH = IMAGES_DIR / "compare_cache.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Kept in sync with the MODELS list in pages/model_compare.py.
MODELS = [
    {"key": "merged-m-v2", "weights": detector.MERGED_M_V2_WEIGHTS, "label": detector.MERGED_M_V2_LABEL},
    {"key": "supervisorv1", "weights": detector.SUPERVISOR_V1_WEIGHTS, "label": detector.SUPERVISOR_V1_LABEL},
]


def main():
    image_paths = sorted(
        (p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    if not image_paths:
        raise SystemExit(f"No demo images found in {IMAGES_DIR}")

    cache = {"models": {}}
    for m in MODELS:
        print(f"{m['label']} ({m['key']})...")
        model = detector.load_model(m["weights"])
        if model is None:
            print(f"  SKIPPED — weights not found at {m['weights']}")
            continue
        results = {}
        for p in image_paths:
            img = vh.load_image(p.read_bytes())
            raw = detector.detect_raw(model, img)
            results[p.name] = raw
            print(f"  {p.name}: {len(raw)} raw detections")
        cache["models"][m["key"]] = {"weights": str(m["weights"]), "results": results}

    if not cache["models"]:
        raise SystemExit("No model produced any results — nothing written.")

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"Wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
