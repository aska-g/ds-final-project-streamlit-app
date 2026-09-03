#!/usr/bin/env python
"""Batch-run Falcon 9-class detection on chosen-pics using single model load (CPU batch engine)."""
from pathlib import Path
import torch
from PIL import Image

from falcon_perception import build_prompt_for_task, load_and_prepare_model, setup_torch_config

setup_torch_config()

CLASSES = ["person","helmet","safety vest","gloves","boots","no hardhat","no safety vest","no gloves","no boots"]
TASK = "detection"
IMAGE_ROOT = Path("chosen-pics")
OUT_ROOT = Path("falcon/outputs/chosen-pics-9cls")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Load model once
print("Loading model...")
from falcon_perception import PERCEPTION_MODEL_ID
model, tokenizer, model_args = load_and_prepare_model(
    hf_model_id=PERCEPTION_MODEL_ID,
    hf_revision="main",
    hf_local_dir=None,
    device=None,  # auto cpu
    dtype="float32",
    compile=False,
)
device = model.device
print(f"Model ready device={device} dtype={model.dtype}")

from falcon_perception.data import ImageProcessor
from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate

image_processor = ImageProcessor(patch_size=16, merge_size=1)
stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
engine = BatchInferenceEngine(model, tokenizer, kernel_options={})

# Import helpers for saving (reuse from perception_multi)
import sys
sys.path.insert(0, str(Path("falcon/demo").resolve()))
from perception_multi import _save_merged_labeled_image, _save_yolo_txt

images = sorted(IMAGE_ROOT.glob("*.jpg"))
if not images:
    print(f"No images in {IMAGE_ROOT}")
    raise SystemExit(1)

print(f"Found {len(images)} images, classes={CLASSES}")

# chunked inference to avoid OOM on large images (flex_attention materializes BxHxN*N)
CHUNK_SIZE = 3  # 9 classes -> 3 chunks of 3; large images use 2 per chunk with smaller max_dimension

for idx, img_path in enumerate(images, 1):
    stem = img_path.stem  # e.g. 000037_jpg.rf.mTU8ska3I1ccXQsI4CoW
    out_dir = OUT_ROOT / stem
    # resume: skip if already has predictions (allow re-run if empty)
    if (out_dir / "predictions_yolo.txt").exists() and (out_dir / "merged.jpg").exists():
        # check not empty predictions dir from previous 5cls leftover
        try:
            if (out_dir / "predictions_yolo.txt").stat().st_size >= 0:
                print(f"\n[{idx}/{len(images)}] SKIP {img_path.name} (already done)")
                continue
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    # clean empty failed dir from previous OOM (pos_2010)
    if not any(out_dir.iterdir()):
        pass
    print(f"\n[{idx}/{len(images)}] {img_path.name} -> {out_dir}")
    pil_image = Image.open(img_path).convert("RGB")
    W, H = pil_image.size
    max_dim = 640 if max(W, H) > 1000 else 1024
    chunk_sz = 2 if max_dim == 640 else CHUNK_SIZE
    print(f"  image {W}x{H} -> max_dim={max_dim} chunk={chunk_sz}")

    flat = []
    # process in chunks
    chunks = [CLASSES[i:i+chunk_sz] for i in range(0, len(CLASSES), chunk_sz)]
    for ci, chunk in enumerate(chunks):
        print(f"  chunk {ci+1}/{len(chunks)}: {chunk}")
        prompts = [build_prompt_for_task(q, TASK) for q in chunk]
        batch_inputs = process_batch_and_generate(
            tokenizer,
            [(pil_image, p) for p in prompts],
            max_length=4096,
            min_dimension=256,
            max_dimension=max_dim,
        )
        batch_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_inputs.items()}
        try:
            _, aux_out = engine.generate(
                **batch_inputs,
                max_new_tokens=2048,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                seed=42,
                task=TASK,
            )
        except RuntimeError as e:
            print(f"  [OOM] chunk {chunk} failed: {e}")
            # fallback: try chunk size 1 with smaller dim
            flat_chunk = []
            for q in chunk:
                print(f"    retry single {q} with max_dim=512")
                bi1 = process_batch_and_generate(tokenizer, [(pil_image, build_prompt_for_task(q, TASK))], max_length=4096, min_dimension=256, max_dimension=512)
                bi1 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bi1.items()}
                _, a1 = engine.generate(**bi1, max_new_tokens=2048, temperature=0.0, stop_token_ids=stop_token_ids, seed=42, task=TASK)
                aux_out = a1
                # handle single
                from falcon_perception.visualization_utils import pair_bbox_entries as _pair
                for cls_name, aux in zip([q], aux_out):
                    try: n = len(_pair(aux.bboxes_raw)) if hasattr(aux, "bboxes_raw") else 0
                    except: n = 0
                    print(f"      {cls_name:16s}: {n} boxes")
                # render single
                try:
                    from falcon_perception.visualization_utils import render_batch_inference_outputs as _render
                    bi1["__orig_images__"] = [pil_image]
                    _render("BATCH", bi1, aux_out, [], TASK, out_dir=str(out_dir), queries=[q])
                except Exception as re:
                    print(f"      [warn] render single failed: {re}")
                # flat
                try:
                    from falcon_perception.visualization_utils import detections_from_batch_aux as _dets
                    hw = (H, W); pm = bi1.get("pixel_mask")
                    for j, (cls_name, aux) in enumerate(zip([q], aux_out)):
                        pm1 = pm[j, 0] if isinstance(pm, torch.Tensor) and pm.ndim >= 3 else None
                        for d in _dets(aux, pixel_mask_1hw=pm1, orig_hw=hw, segmentation=(TASK=="segmentation")):
                            d2 = dict(d); d2["label"] = cls_name; flat.append(d2)
                except Exception as fe:
                    print(f"      [flat] skip {fe}")
            continue

        from falcon_perception.visualization_utils import pair_bbox_entries
        for cls_name, aux in zip(chunk, aux_out):
            try:
                n = len(pair_bbox_entries(aux.bboxes_raw)) if hasattr(aux, "bboxes_raw") else 0
            except Exception:
                n = 0
            print(f"    {cls_name:16s}: {n} boxes")

        # render per-chunk masks
        try:
            from falcon_perception.visualization_utils import render_batch_inference_outputs
            bi = dict(batch_inputs)
            bi["__orig_images__"] = [pil_image] * len(chunk)
            render_batch_inference_outputs("BATCH", bi, aux_out, [], TASK, out_dir=str(out_dir), queries=chunk)
        except Exception as e:
            print(f"  [warn] render chunk failed: {e}")

        # collect flat
        try:
            from falcon_perception.visualization_utils import detections_from_batch_aux as _dets_from_batch
            hw = (H, W)
            pm = batch_inputs.get("pixel_mask")
            for j, (cls_name, aux) in enumerate(zip(chunk, aux_out)):
                pm1 = pm[j, 0] if isinstance(pm, torch.Tensor) and pm.ndim >= 3 else None
                for d in _dets_from_batch(aux, pixel_mask_1hw=pm1, orig_hw=hw, segmentation=(TASK=="segmentation")):
                    d2 = dict(d)
                    d2["label"] = cls_name
                    flat.append(d2)
        except Exception as e:
            print(f"  [flat] skip chunk {chunk}: {e}")
            import traceback; traceback.print_exc()

    # merged + yolo txt (all chunks combined)
    try:
        _save_merged_labeled_image(pil_image, flat, CLASSES, out_dir / "merged.jpg", task=TASK)
    except Exception as e:
        print(f"[merged] skip {e}")
        import traceback; traceback.print_exc()
    try:
        _save_yolo_txt(flat, CLASSES, out_dir / "predictions_yolo.txt")
    except Exception as e:
        print(f"[yolo] skip {e}")

    try:
        pil_image.save(out_dir / "perception_input.jpg")
    except Exception:
        pass

print("\nDone all images.")
# Copy merged.jpg to comparison dirs
for comp_dir in [Path("falcon/output/comparison"), Path("falcon/outputs/comparison")]:
    comp_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        stem = img_path.stem
        src = OUT_ROOT / stem / "merged.jpg"
        dst = comp_dir / img_path.name  # keep .jpg name
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"copied {src} -> {dst}")
        else:
            print(f"missing src {src}")
