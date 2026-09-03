"""Annotate chosen-pics images with Gemini box_2d labels into falcon/outputs/gemini."""
import argparse
import json
from pathlib import Path
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]

def color_for_label(label, label_to_color):
    if label not in label_to_color:
        idx = len(label_to_color) % len(_PALETTE)
        label_to_color[label] = _PALETTE[idx]
    return label_to_color[label]

def annotate_one(image_path: Path, label_path: Path, out_path: Path, label_to_color: dict):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    data = json.loads(label_path.read_text())
    for det in data:
        box = det.get("box_2d")
        label = det.get("label", "unknown")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        # 0-1000 normalized
        xmin = int(round(x1 / 1000 * W))
        ymin = int(round(y1 / 1000 * H))
        xmax = int(round(x2 / 1000 * W))
        ymax = int(round(y2 / 1000 * H))
        xmin = max(0, min(W - 1, xmin))
        ymin = max(0, min(H - 1, ymin))
        xmax = max(0, min(W - 1, xmax))
        ymax = max(0, min(H - 1, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        color = color_for_label(label, label_to_color)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        # label background + text
        text = str(label)
        # text size
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 2
        bg_x0, bg_y0 = xmin, max(0, ymin - th - pad * 2)
        # if not enough space above, draw inside top
        if bg_y0 == 0 and ymin < th + pad * 2:
            bg_y0 = ymin
        bg_x1, bg_y1 = bg_x0 + tw + pad * 2, bg_y0 + th + pad * 2
        bg_x1 = min(W, bg_x1)
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=color)
        draw.text((bg_x0 + pad, bg_y0 + pad), text, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return len(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="chosen-pics")
    ap.add_argument("--label-dir", default="falcon/outputs/gemini")
    ap.add_argument("--output-dir", default="falcon/outputs/gemini")
    ap.add_argument("--suffix", default="_annotated.jpg")
    args = ap.parse_args()
    in_dir = Path(args.input_dir)
    lab_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)

    # deterministic label->color: sorted unique labels across all jsons
    all_labels = set()
    for p in sorted(lab_dir.glob("*.json")):
        try:
            for d in json.loads(p.read_text()):
                all_labels.add(d.get("label", "unknown"))
        except Exception:
            pass
    sorted_labels = sorted(all_labels)
    label_to_color = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(sorted_labels)}

    count = 0
    for lab_path in sorted(lab_dir.glob("*.json")):
        # lab_path name is like "<image_name>.jpg.json" -> strip .json
        image_name = lab_path.name[:-5]  # remove .json
        image_path = in_dir / image_name
        if not image_path.exists():
            # handle _2 variant: ppe_..._2.jpg -> base ppe_....jpg
            if "_2.jpg" in image_name:
                base = image_name.replace("_2.jpg", ".jpg")
                alt = in_dir / base
                if alt.exists():
                    image_path = alt
                else:
                    print(f"[skip] no image for {lab_path.name} -> {image_name}")
                    continue
            else:
                print(f"[skip] no image for {lab_path.name}")
                continue
        out_name = image_name.replace(".jpg", args.suffix)
        # for _2 variant, this yields _2_annotated.jpg which is distinct
        if not out_name.endswith(".jpg"):
            out_name += ".jpg"
        out_path = out_dir / out_name
        n = annotate_one(image_path, lab_path, out_path, label_to_color)
        print(f"[ok] {image_name} ({n} boxes) -> {out_path}")
        count += 1
    # also warn for images without labels
    for img_path in sorted(in_dir.glob("*.jpg")):
        json_name = img_path.name + ".json"
        if not (lab_dir / json_name).exists():
            print(f"[warn] no Gemini label for {img_path.name} (no {json_name})")
    print(f"Done: {count} annotated images in {out_dir}")

if __name__ == "__main__":
    main()
