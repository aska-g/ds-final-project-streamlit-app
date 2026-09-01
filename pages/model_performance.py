"""P0 Safety — model performance: baseline classifier metrics/curves, and YOLO run metrics."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _APP_DIR  # this repo's root (flattened out of streamlit_app/)
STATS_PATH = _APP_DIR / "models" / "baseline_stats.json"
HISTORY_PATH = _APP_DIR / "models" / "baseline_history.json"
RUNS_DIR = _REPO_ROOT / "runs"


def _find_col(df, *keywords):
    """First column whose name contains all keywords, e.g. _find_col(df, 'train', 'box_loss')."""
    return next((c for c in df.columns if all(k in c for k in keywords)), None)


st.title("Model performance comparison")

# The ONLY runs that show up on this page — nothing under runs/ is auto-discovered anymore.
# This is deliberate: the repo's runs/ folder can pick up stray training runs from teammates
# (e.g. someone's own experiment landing at runs/detect/runs/ppe_dev/yolo26n/) that aren't
# ready to show here. Add an entry only once a run is one you actually want on this page —
# dict order = display order, "path" is relative to the runs/ folder.
RUN_INFO = {
    "yolov8n_scratch": {
        "label": "YOLOv8n — trained from scratch",
        "path": "scratch/yolov8n_scratch",
    },
    "pretrained_100e": {
        "label": "YOLOv8n — pretrained (bundled with dataset)",
        "path": "pretrained_100e",
        "caption": (
            "Came bundled with the Kaggle dataset download, not trained by us. Started from "
            "COCO-pretrained yolov8n.pt weights and fine-tuned for 100 epochs on the same 10 "
            "classes as our data — a useful ceiling to compare our own runs against."
        ),
        "compare_label": "v8",
        "live_on_compare": True,
    },
    "yolo26s_css_100e": {
        "label": "YOLO26s — css-data, 100 epochs",
        "path": "yolo26s_css_100e",
        "caption": (
            "Trained by a teammate outside this repo, on the same css-data dataset and class "
            "list as pretrained_100e — a genuine apples-to-apples comparison. Different "
            "architecture (YOLO26, not YOLOv8n). Beats pretrained_100e on recall, mAP50 and "
            "mAP50-95 at the final epoch. Shown on Model Comparison as \"v26\"."
        ),
        "compare_label": "v26",
        "live_on_compare": True,
    },
    "yolo26m_css_150e-2": {
        "label": "YOLO26m — css-data, 150 epochs",
        "path": "detect/yolo26m_css_150e-2",
        "caption": (
            "Same css-data dataset and class list as yolo26s_css_100e above, but the larger "
            "YOLO26m backbone trained for 150 epochs (patience=20, ran to completion — no "
            "early stopping). Beats yolo26s_css_100e on every aggregate metric (precision "
            "94,1% vs 89,7%, recall 82,4% vs 79,4%, mAP50 88,5% vs 86,4%, mAP50-95 64,6% vs "
            "58,6%). Shown on Model Comparison as \"css-m-150\"."
        ),
        "compare_label": "css-m-150",
        "live_on_compare": True,
    },
    "yolo26m_css_300e": {
        "label": "YOLO26m — css-data, 300 epochs",
        "path": "detect/yolo26m_css_300e",
        "caption": (
            "Same css-data dataset/vocabulary and YOLO26m backbone as yolo26m_css_150e-2 "
            "above, extended to a 300-epoch schedule (patience=25), early-stopped at 246/300. "
            "Further improves every aggregate metric over the 150-epoch run (precision 96,1% "
            "vs 94,1%, recall 84,2% vs 82,4%, mAP50 90,6% vs 88,5%, mAP50-95 66,1% vs 64,6%) — "
            "the best-performing model in this whole comparison by every aggregate metric, "
            "though it was never wired in as any page's live default. Shown on Model "
            "Comparison as \"css-m-300\"."
        ),
        "compare_label": "css-m-300",
        "live_on_compare": True,
    },
    "yolo26s_merged_100e": {
        "label": "YOLO26s — merged dataset, 100 epochs",
        "path": "detect/yolo26s_merged_100e",
        "caption": (
            "Trained by a teammate outside this repo on a merged dataset (9 classes: person, "
            "helmet/no-helmet, vest/no-vest, gloves/no-gloves, boots/no-boots). Working Person "
            "class (83% recall) plus two PPE items no other run here tracks. Shown on Model "
            "Comparison and wired into the app's compliance logic."
        ),
        "compare_label": "merged",
        "live_on_compare": True,
    },
    "yolo26m_merged_150e": {
        "label": "YOLO26m — merged dataset, 150 epochs",
        "path": "detect/yolo26m_merged_150e",
        "caption": (
            "Same merged dataset/vocabulary as yolo26s_merged_100e above, but the larger "
            "YOLO26m backbone trained for the full 150 epochs (patience=20, ran to "
            "completion). Beats yolo26s_merged_100e on every aggregate metric and every "
            "per-class confusion-matrix diagonal. Superseded as the app's default by the "
            "yolo26m_merged_150ev2 rerun below — kept here for Model Comparison; see that "
            "entry for how it stacks up against yolo26m_mergedpeople_150e."
        ),
        "compare_label": "merged-m",
        "live_on_compare": True,
    },
    "yolo26m_merged_150ev2": {
        "label": "YOLO26m — merged dataset, 150 epochs (rerun v2)",
        "path": "detect/yolo26m_merged_150ev2",
        "caption": (
            "A rerun of yolo26m_merged_150e above: same merged dataset/vocabulary, same "
            "YOLO26m backbone, same 150-epoch/patience=20 config, early-stopped at 146/150 "
            "epochs. Final-epoch metrics are essentially a wash against yolo26m_merged_150e "
            "(precision 91,5% vs 92,2%, recall 84,6% vs 84,0%, mAP50 89,0% vs 89,5%, "
            "mAP50-95 59,1% vs 59,6%) — this is the app's current default on the Demo page. "
            "Shown on Model Comparison as \"merged-m-v2\"."
        ),
        "compare_label": "merged-m-v2",
        "live_on_compare": True,
    },
    "yolo26m_mergedpeople_150e": {
        "label": "YOLO26m — merged + pseudo-labeled Person, 150 epochs",
        "path": "detect/yolo26m_mergedpeople_150e",
        "caption": (
            "Same run setup as yolo26m_merged_150e, trained instead on \"mergedpeople\": "
            "data/merged with ppe_detection_m's Person boxes filled in via pseudo-labeling "
            "(see person_pseudolabels_test.ipynb) rather than left absent. Early-stopped at "
            "134/150 epochs. Slightly better Person recall (88% vs 86%) and aggregate "
            "mAP50-95/recall than yolo26m_merged_150e, but a real trade-off: its confusion "
            "matrix shows it also misclassifies far more true background as \"person\" (31% "
            "vs 16%), driving its lower aggregate precision (91,2% vs 92,2%) — why "
            "yolo26m_merged_150e, not this run, was chosen as the app's default."
        ),
        "compare_label": "mergedpeople",
        "live_on_compare": True,
    },
    "yolo26s_Altec_PPE_100e": {
        "label": "YOLO26s — Altec PPE dataset, 100 epochs",
        "path": "detect/yolo26s_Altec_PPE_100e",
        "caption": (
            "Trained by a teammate outside this repo on a different PPE dataset (10 classes: "
            "Face_masks, Face_shield, Glasses, Gloves, Helmet, Safety_shoes, Safety_vests, plus "
            "lowercase glasses/helmet duplicates) — no Person class at all, confirmed via its "
            "confusion matrix below. Can't drive a per-person compliance verdict, so it doesn't "
            "run live on Model Comparison — training metrics only, here."
        ),
        "compare_label": "Altec",
        "live_on_compare": False,  # no Person class — Model Comparison shows it as an N/A card
    },
}

runs_to_show = []
for key, info in RUN_INFO.items():
    results_path = RUNS_DIR / info["path"] / "results.csv"
    if results_path.exists():
        runs_to_show.append((key, info, results_path))

# Summary table — the 9 models currently on the Model Comparison page, side by side.
# yolov8n_scratch is deliberately excluded: it's shown lower on this page, but was never
# added to Model Comparison, so it's not part of "the 9 models" this table is answering for.
compare_rows = [(key, info, path) for key, info, path in runs_to_show if info.get("compare_label")]

# Comparison overview first — this is what a visitor to this page wants to see before
# anything else. The baseline classifier and the noisy per-model training details (moved
# into collapsed expanders below) both come after it.
if compare_rows:
    st.caption(
        """
        mAP50 = average precision at a loose 0.5 IoU overlap threshold (is the box roughly in the right place)\n
        mAP50-95 = average across stricter thresholds from 0.5 to 0.95 (more demanding number)
        """
    )
    table_rows = []
    for key, info, results_path in compare_rows:
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        map50_95_col = next((c for c in df.columns if "mAP50-95" in c), None)
        map50_col = next((c for c in df.columns if "mAP50" in c and "mAP50-95" not in c), None)
        precision_col = _find_col(df, "precision")
        recall_col = _find_col(df, "recall")
        table_rows.append({
            "Model": info["compare_label"],
            "Run": info.get("label", key),
            "Epochs": len(df),
            "Precision": round(float(last[precision_col]), 3) if precision_col else None,
            "Recall": round(float(last[recall_col]), 3) if recall_col else None,
            "mAP50": round(float(last[map50_col]), 3) if map50_col else None,
            "mAP50-95": round(float(last[map50_95_col]), 3) if map50_95_col else None
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    # Per-class numbers: results.csv is aggregate-only — Ultralytics never writes a per-class
    # CSV, the confusion matrix image is the only place these numbers exist. So this table is
    # manually transcribed from each run's confusion_matrix(_normalized).png diagonal (v8/v26/
    # merged/Altec read 2026-08-27, merged-m/mergedpeople added 2026-08-28, css-m-150/css-m-300/
    # merged-m-v2 added 2026-09-01) — NOT recomputed live like the summary table above. If any
    # of these runs get retrained, re-read the new confusion matrix and update this table by
    # hand.
    #
    # Rows follow the app's own tracked slots (detector.SLOT_ITEMS) — person, then each PPE
    # item's present/absent pair. "—" means that model's training data has no matching class at
    # all (e.g. Altec has no Person, no negative classes for anything). Different models use
    # different words for the same slot (hardhat vs helmet) and different underlying datasets,
    # so a cell is that MODEL's own class name plus its own diagonal value — read each column on
    # its own terms rather than comparing raw numbers across columns as if it were one dataset.
    st.subheader("Per-class numbers")
    PER_CLASS = [
        {"Slot": "Person",                    "v8": "0.80", "v26": "0.83", "css-m-150": "0.90", "css-m-300": "0.89", "merged": "0.83", "merged-m": "0.86", "merged-m-v2": "0.83", "mergedpeople": "0.88", "Altec": "—"},
        {"Slot": "Head protection — present",  "v8": "0.76", "v26": "0.85", "css-m-150": "0.89", "css-m-300": "0.86", "merged": "0.94", "merged-m": "0.95", "merged-m-v2": "0.95", "mergedpeople": "0.95", "Altec": "0.89 / 0.78"},
        {"Slot": "Head protection — absent",   "v8": "0.62", "v26": "0.67", "css-m-150": "0.77", "css-m-300": "0.83", "merged": "0.88", "merged-m": "0.90", "merged-m-v2": "0.89", "mergedpeople": "0.88", "Altec": "—"},
        {"Slot": "Vest — present",             "v8": "0.78", "v26": "0.90", "css-m-150": "0.93", "css-m-300": "0.93", "merged": "0.76", "merged-m": "0.81", "merged-m-v2": "0.83", "mergedpeople": "0.81", "Altec": "0.70"},
        {"Slot": "Vest — absent",              "v8": "0.70", "v26": "0.74", "css-m-150": "0.77", "css-m-300": "0.84", "merged": "0.78", "merged-m": "0.81", "merged-m-v2": "0.80", "mergedpeople": "0.82", "Altec": "—"},
        {"Slot": "Mask — present",             "v8": "0.90", "v26": "0.95", "css-m-150": "0.95", "css-m-300": "0.95", "merged": "—",    "merged-m": "—",    "merged-m-v2": "—",    "mergedpeople": "—",    "Altec": "0.83 / 0.70"},
        {"Slot": "Mask — absent",              "v8": "0.66", "v26": "0.70", "css-m-150": "0.82", "css-m-300": "0.88", "merged": "—",    "merged-m": "—",    "merged-m-v2": "—",    "mergedpeople": "—",    "Altec": "—"},
        {"Slot": "Gloves — present",           "v8": "—",    "v26": "—",    "css-m-150": "—",    "css-m-300": "—",    "merged": "0.86", "merged-m": "0.88", "merged-m-v2": "0.87", "mergedpeople": "0.88", "Altec": "0.56"},
        {"Slot": "Gloves — absent",            "v8": "—",    "v26": "—",    "css-m-150": "—",    "css-m-300": "—",    "merged": "0.83", "merged-m": "0.83", "merged-m-v2": "0.82", "mergedpeople": "0.82", "Altec": "—"},
        {"Slot": "Boots — present",            "v8": "—",    "v26": "—",    "css-m-150": "—",    "css-m-300": "—",    "merged": "0.90", "merged-m": "0.91", "merged-m-v2": "0.92", "mergedpeople": "0.91", "Altec": "0.73"},
        {"Slot": "Boots — absent",             "v8": "—",    "v26": "—",    "css-m-150": "—",    "css-m-300": "—",    "merged": "0.86", "merged-m": "0.89", "merged-m-v2": "0.88", "mergedpeople": "0.89", "Altec": "—"},
    ]
    st.dataframe(pd.DataFrame(PER_CLASS), hide_index=True, width="stretch", height="content")
    st.caption(
        "Numbers are each class's diagonal in its confusion matrix — of every box that was "
        "truly that class, the fraction the model predicted correctly (≈ recall). Blank cells "
        "are classes that model's dataset never had, not a zero score. A cell with two numbers "
        "(Altec's Head protection / Mask rows) is two separate classes that model tracks for "
        "the same slot — e.g. Helmet and a lowercase duplicate \"helmet\" class, or Face_masks "
        "and Face_shield — see the confusion matrices below for the exact class names and full "
        "picture, including classes not in this table (Safety Cone/machinery/vehicle for "
        "v8/v26/css-m-150/css-m-300, Glasses for Altec)."
    )



st.header("Per-model training details")

with st.expander("Baseline classifier"):
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("This model (val accuracy)", f"{stats['val_accuracy']:.0%}")
        col2.metric(f"Majority-class baseline ({stats['majority_class']})", f"{stats['majority_baseline_accuracy']:.0%}")
        col3.metric("Random-guess baseline", f"{stats['random_guess_accuracy']:.0%}")
    else:
        st.info("No trained baseline model yet — run train_baseline_classifier.py first.")

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        st.subheader("Training curves — is this model overfitting?")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Loss per epoch")
            st.line_chart(pd.DataFrame({"train": history["loss"], "val": history["val_loss"]}))
        with col2:
            st.caption("Accuracy per epoch")
            st.line_chart(pd.DataFrame({"train": history["accuracy"], "val": history["val_accuracy"]}))
        st.caption(
            "Training loss/accuracy keep improving while validation plateaus early — a classic "
            "overfitting signature. The model is overconfident on out-of-distribution input."
        )


if not runs_to_show:
    st.info("No YOLO training runs configured yet — add an entry to RUN_INFO in this file.")
else:
    for key, info, results_path in runs_to_show:
        run_dir = results_path.parent
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        map50_95_col = next((c for c in df.columns if "mAP50-95" in c), None)
        map50_col = next((c for c in df.columns if "mAP50" in c and "mAP50-95" not in c), None)

        with st.expander(info.get("label", run_dir.name)):
            if info.get("caption"):
                st.caption(info["caption"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Epochs completed", len(df))
            col2.metric("mAP50", f"{last[map50_col]:.3f}" if map50_col else "n/a")
            col3.metric("mAP50-95", f"{last[map50_95_col]:.3f}" if map50_95_col else "n/a")

            # Per-epoch curves, straight from results.csv (one row per epoch).
            box_train, box_val = _find_col(df, "train", "box_loss"), _find_col(df, "val", "box_loss")
            cls_train, cls_val = _find_col(df, "train", "cls_loss"), _find_col(df, "val", "cls_loss")
            dfl_train, dfl_val = _find_col(df, "train", "dfl_loss"), _find_col(df, "val", "dfl_loss")
            precision_col = _find_col(df, "precision")
            recall_col = _find_col(df, "recall")

            curve_col1, curve_col2 = st.columns(2)
            with curve_col1:
                if all([box_train, cls_train, dfl_train]):
                    st.caption("Loss per epoch (box + cls + dfl, summed)")
                    loss_data = {"train": df[box_train] + df[cls_train] + df[dfl_train]}
                    if all([box_val, cls_val, dfl_val]):
                        loss_data["val"] = df[box_val] + df[cls_val] + df[dfl_val]
                    st.line_chart(pd.DataFrame(loss_data))
                else:
                    st.caption("Loss columns not found in results.csv")
            with curve_col2:
                if map50_col and map50_95_col:
                    st.caption("mAP per epoch")
                    st.line_chart(pd.DataFrame({"mAP50": df[map50_col], "mAP50-95": df[map50_95_col]}))
                else:
                    st.caption("mAP columns not found in results.csv")

            if precision_col and recall_col:
                st.caption("Precision / recall per epoch")
                st.line_chart(pd.DataFrame({"precision": df[precision_col], "recall": df[recall_col]}))

            # results.csv is aggregate-only (one row per epoch, no per-class columns) —
            # Ultralytics never writes a per-class numeric table. The confusion matrix and the
            # Box P/R/F1/PR curves it drops in the run folder ARE the real per-class breakdown,
            # so surface the confusion matrix directly (most readable at a glance) and keep the
            # rest just below.
            cm_path = run_dir / "confusion_matrix_normalized.png"
            if not cm_path.exists():
                cm_path = run_dir / "confusion_matrix.png"
            if cm_path.exists():
                st.caption("Per-class detection breakdown (confusion matrix — diagonal ≈ per-class recall)")
                st.image(str(cm_path), width="stretch")
            else:
                st.caption("No confusion matrix found for this run.")

            # Ultralytics also drops other ready-made plots in the run folder — no need to
            # rebuild these from scratch, just surface them. (No nested expander here — this
            # whole block is already inside one, and Streamlit doesn't allow expanders within
            # expanders.)
            extra_plots = {
                "results.png": "All metrics, Ultralytics' own summary grid",
                "confusion_matrix.png": "Confusion matrix (raw counts)",
                "confusion_matrix_normalized.png": "Confusion matrix (normalized)",
                "BoxPR_curve.png": "Precision-recall curve, per class",
                "BoxP_curve.png": "Precision curve, per class",
                "BoxR_curve.png": "Recall curve, per class",
                "BoxF1_curve.png": "F1 curve, per class",
            }
            available_plots = [(name, caption) for name, caption in extra_plots.items() if (run_dir / name).exists()]
            if available_plots:
                st.caption("More plots from this run (per-class curves, raw confusion matrix, summary grid)")
                for name, caption in available_plots:
                    st.image(str(run_dir / name), caption=caption, width="stretch")



# ── PPE screening: our YOLO vs vision LLMs (local + hosted API) ──────────────
# Same story as RUN_INFO above — nothing under runs/llm is auto-discovered.
# These two runs share the exact same 100 merged-dataset images (seed 42), so
# their presence.csv files stack into one comparison. "model" col in each file:
# yolo + the LLMs listed here. Add the newest matching pair when a fresh run
# lands; order = display order in the table.
LLM_RUNS = [
    "20260828_035813_merged_n100_seed42_yolo-ollama-qwen3-vl-gemma4-minicpm-v",
    "20260831_merged_n100_seed42_yolo-gemini",
]
# Ground truth: per-image presence booleans derived from the merged dataset's
# own YOLO label .txt files (class id in the file => that class is present).
# The dataset itself lives outside this repo, so those 900 rows are frozen into
# this CSV once — regenerate it if LLM_RUNS changes to a different image set.
GT_CSV = RUNS_DIR / "llm" / "_merged_n100_seed42_ground_truth.csv"
# model id in presence.csv -> (display label, where it runs). "ollama" = llava:7b.
LLM_MODELS = {
    "yolo": ("YOLO26s (our detector)", "local"),
    "gemini": ("Gemini 3.6 Flash", "API"),
    "qwen3-vl": ("qwen3-vl:4b", "local"),
    "gemma4": ("gemma4:e4b", "local"),
    "minicpm-v": ("minicpm-v:8b", "local"),
    "ollama": ("llava:7b", "local"),
}

llm_frames = []
for name in LLM_RUNS:
    p = RUNS_DIR / "llm" / name / "presence.csv"
    if p.exists():
        llm_frames.append(pd.read_csv(p))

CLASS_ORDER = ["person", "helmet", "gloves", "boots", "vest",
               "no-helmet", "no-gloves", "no-boots", "no-vest"]

if llm_frames and GT_CSV.exists():
    st.header("PPE screening — our YOLO vs vision LLMs")
    st.caption(
        "Each model was asked, per image, whether at least one instance of each class is "
        "visible (9 classes, 100 merged-dataset images, seed 42) — a yes/no presence call, no "
        "bounding boxes. Scored against the merged dataset's own labels as ground truth. "
        "YOLO26s was trained on this dataset, so it sets the bar; the LLMs are zero-shot."
    )
    llm = pd.concat(llm_frames, ignore_index=True)
    llm["present"] = llm["present"].astype(str).str.lower().eq("true")
    llm["parse_error"] = llm["parse_error"].astype(str).str.lower().eq("true")
    llm = llm.drop_duplicates(["file", "model", "class_name"])  # yolo is in both runs
    # .pivot (not pivot_table) so the bool dtype survives — (file, class, model)
    # is unique after the drop_duplicates above, no aggregation needed.
    cells = llm.pivot(index=["file", "class_name"], columns="model", values="present").fillna(False)

    gt = pd.read_csv(GT_CSV)
    gt["present"] = gt["present"].astype(str).str.lower().eq("true")
    truth = gt.set_index(["file", "class_name"])["present"].reindex(cells.index).fillna(False)

    # matplotlib (for the red-green cell shading) ships with ultralytics, already
    # a pinned dependency — no requirements.txt change needed.
    METRICS = ["Accuracy", "Recall", "Precision", "F1"]
    summary = []
    for mid, (label, where) in LLM_MODELS.items():
        if mid not in cells:
            continue
        pred = cells[mid]
        tp = (pred & truth).sum()
        recall = tp / truth.sum() if truth.sum() else float("nan")
        precision = tp / pred.sum() if pred.sum() else float("nan")
        summary.append({
            "Model": label,
            "Runs": where,
            "Accuracy": (pred == truth).mean(),
            "Recall": recall,
            "Precision": precision,
            "F1": 2 * precision * recall / (precision + recall),
            "Parse failures": int(llm.loc[llm["model"] == mid, "parse_error"].sum()),
        })
    summary_df = pd.DataFrame(summary).set_index("Model")
    st.dataframe(
        summary_df.style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, subset=METRICS)
        .format({m: "{:.1%}" for m in METRICS}),
        width="stretch",
    )
    st.caption(
        "Recall is the one that matters for a safety screen — missing PPE that's really there "
        "is the costly error. YOLO26s clears ~94% recall at ~97% precision; the LLMs trade "
        "high-ish recall for weak precision (they over-call PPE present, worst on the "
        "'no-helmet' / 'no-vest' negative classes)."
    )

    st.subheader("Recall by class")
    per_class = {}
    for mid, (label, _) in LLM_MODELS.items():
        if mid not in cells:
            continue
        col = {}
        for cls in CLASS_ORDER:
            sl_pred = cells.xs(cls, level="class_name")[mid]
            sl_truth = truth.xs(cls, level="class_name")
            pos = sl_truth.sum()
            col[cls] = (sl_pred & sl_truth).sum() / pos if pos else float("nan")
        per_class[label] = col
    per_class_df = pd.DataFrame(per_class).reindex(CLASS_ORDER)
    st.dataframe(
        per_class_df.style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
        .format("{:.0%}", na_rep="—"),
        width="stretch",
    )
    st.caption(
        "Fraction of each class's ground-truth-present images the model also flagged. \"—\" = "
        "that class never appears in the 100-image sample."
    )
