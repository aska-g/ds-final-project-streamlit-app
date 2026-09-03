"""Draw falcon/outputs/gemini *.json labels onto chosen-pics originals -> falcon/outputs/gemini/*.jpg"""
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

APP = Path(__file__).resolve().parents[1]
GEMINI_DIR = APP / "falcon" / "outputs" / "gemini"
CHOSEN_DIR = APP / "chosen-pics"

# same palette as detector.CLASS_META
CLASS_META = {
    "person":    {"label": "Person",         "color": "#F2F2F2"},
    "hardhat":   {"label": "Hardhat",        "color": "#5899E3"},
    "nohardhat": {"label": "NO-Hardhat",     "color": "#EC5B4C"},
    "vest":      {"label": "Safety Vest",    "color": "#5FBC73"},
    "novest":    {"label": "NO-Safety Vest", "color": "#F2994A"},
    "gloves":    {"label": "Gloves",         "color": "#4FB4D9"},
    "nogloves":  {"label": "NO-Gloves",      "color": "#F0805A"},
    "boots":     {"label": "Boots",          "color": "#A87EF8"},
    "noboots":   {"label": "NO-Boots",       "color": "#D9536B"},
    "mask":      {"label": "Mask",           "color": "#45BFAE"},
    "nomask":    {"label": "NO-Mask",        "color": "#E0678F"},
}

def hex_to_rgb(s): s=s.lstrip("#"); return tuple(int(s[i:i+2],16) for i in (0,2,4))

FONT_PATH = APP / "assets" / "fonts" / "IBMPlexMono-Bold.ttf"
def get_font(size):
    try: return ImageFont.truetype(str(FONT_PATH), size)
    except: return ImageFont.load_default()

def draw_annotated(src: Path, boxes: list, dst: Path):
    img = Image.open(src).convert("RGB")
    w,h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    # box widths scaled to image size
    bw = max(2, round(min(w,h)/280))
    font = get_font(max(12, round(min(w,h)/45)))
    for b in boxes:
        raw_key = str(b.get("key") or b.get("label") or "?").lower().strip()
        # normalize e.g. "no-hardhat" -> nohardhat
        key = raw_key.replace("-", "").replace("_","").replace(" ","")
        meta = CLASS_META.get(key) or {"label": raw_key, "color": "#EFE600"}
        color = hex_to_rgb(meta["color"])
        box = b["box"]
        # box is [x0,y0,x1,y1] normalized 0-1
        x0,y0,x1,y1 = box
        # clamp
        x0,y0,x1,y1 = max(0,x0),max(0,y0),min(1,x1),min(1,y1)
        px0,py0,px1,py1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        # ensure x0<x1
        if px1<=px0: px1=px0+1
        if py1<=py0: py1=py0+1
        # outline with alpha
        draw.rectangle([px0,py0,px1,py1], outline=color+(255,), width=bw)
        # label chip at top-left
        label = meta["label"]
        # include conf if not 1.0? gemini conf is 1.0 always, skip
        try:
            bbox = draw.textbbox((0,0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except: tw,th = (len(label)*6, 12)
        pad_x, pad_y = 4,2
        chip = [px0, max(0, py0 - th - 2*pad_y), px0 + tw + 2*pad_x, py0]
        # dark chip with colored text
        draw.rectangle(chip, fill=(20,20,20,210), outline=color+(255,), width=1)
        draw.text((chip[0]+pad_x, chip[1]+pad_y), label, fill=color+(255,), font=font)
    img.save(dst, quality=95, subsampling=0)
    print(f"wrote {dst.name} ({w}x{h}, {len(boxes)} boxes) <- {src.name}")

if not GEMINI_DIR.exists():
    print(f"missing {GEMINI_DIR}"); raise SystemExit(1)
if not CHOSEN_DIR.exists():
    print(f"missing {CHOSEN_DIR}"); raise SystemExit(1)

for j in sorted(GEMINI_DIR.glob("*.json")):
    stem = j.stem  # e.g. 000037_jpg.rf.... or ppe_0837..._2
    # find source image: try exact stem.jpg, then strip _<num> suffix
    src = CHOSEN_DIR / f"{stem}.jpg"
    if not src.exists():
        m = re.sub(r"_\d+$", "", stem)
        src = CHOSEN_DIR / f"{m}.jpg"
    if not src.exists():
        # try any jpg with prefix?
        print(f"skip {j.name}: no source image for stem {stem}")
        continue
    with open(j, encoding="utf-8") as f:
        boxes = json.load(f)
    if isinstance(boxes, dict) and "detections" in boxes:
        boxes = boxes["detections"]
    if not isinstance(boxes, list):
        print(f"skip {j.name}: not a list")
        continue
    # output jpg named after json stem (so _2 variant stays distinct)
    dst = GEMINI_DIR / f"{stem}.jpg"
    draw_annotated(src, boxes, dst)

print("done")
