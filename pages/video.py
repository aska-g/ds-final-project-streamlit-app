"""SuperVisor — Video inference (yolo26s_supervisorv1_fixed_nomosaic_300e)."""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

import detector
import view_helpers as vh
from infer_video import annotate_frame

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("VIDEO INFERENCE", detector.SUPERVISOR_V1_LABEL), unsafe_allow_html=True)

model = detector.load_model(detector.SUPERVISOR_V1_WEIGHTS)
if model is None:
    st.error(
        f"No trained weights found at `{detector.SUPERVISOR_V1_WEIGHTS}`.\n\n"
        "Restore that file or point `HIVIS_MODEL_PATH` at a `.pt` you do have."
    )
    st.stop()

with st.container(border=True):
    st.markdown('<div class="hv-h1" style="font-size:18px">SETTINGS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        conf = st.slider("Confidence", 0.10, 0.90, 0.25, 0.05)
    with c2:
        iou = st.slider("NMS IoU", 0.10, 0.90, 0.50, 0.05)
    with c3:
        imgsz = st.selectbox("Image size", [320, 640, 960], index=1)
    with c4:
        device = st.selectbox("Device", ["auto", "cpu", "0"], index=0)
    c5, c6, c7 = st.columns(3)
    with c5:
        stride = st.selectbox("Frame stride", [1, 2, 5, 10], index=1, help="Run inference every Nth frame (higher = faster).")
    with c6:
        max_frames = st.number_input("Max frames (0 = all)", min_value=0, max_value=10000, value=600, step=100, help="File uploads only. Live (webcam/RTSP/URL) runs uncapped — use Streamlit's Stop button to end it.")
    with c7:
        save_output = st.checkbox("Save annotated video", value=True, help="Writes annotated mp4 for download + preview after the run.")

with st.container(border=True):
    st.markdown('<div class="hv-h1" style="font-size:18px">SOURCE</div>', unsafe_allow_html=True)
    st.caption("Upload a file for Streamlit Cloud, or enter a URL / 0 for webcam when running locally. Webcam and RTSP only work when the app runs on your own machine, not on Streamlit Cloud.")
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
    if uploaded is not None and uploaded.size and uploaded.size > 200 * 1024 * 1024:
        st.warning(f"File is {uploaded.size / 1024 / 1024:.1f} MB — processing may be slow or hit memory limits. Consider trimming the clip.")
    url_source = st.text_input(
        "Or URL / webcam",
        placeholder="0 for webcam, or rtsp://… / https://… (leave blank to use uploaded file)",
    )

has_source = bool(uploaded is not None or (url_source and url_source.strip()))
run = st.button("▶ Run video inference", type="primary", disabled=not has_source)
if not has_source:
    st.info("Upload a video or enter a URL/webcam source above to run inference.")
    st.stop()
if not run:
    st.stop()

# Resolve source to something cv2.VideoCapture understands.
tmp_input = None
tmp_output = None
cap = None
writer = None
out_path = None
is_live = uploaded is None and bool(url_source and url_source.strip())
try:
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_input.write(uploaded.getvalue())
        tmp_input.close()
        cap_source = tmp_input.name
        source_label = uploaded.name
    else:
        raw = url_source.strip()
        cap_source = 0 if raw == "0" else raw
        source_label = raw

    dev = None if device == "auto" else device
    if dev:
        try:
            model.to(dev)
        except Exception as e:
            st.warning(f"model.to({dev}) failed: {e} — continuing on default device.")

    st.caption(f"Source: `{source_label}` · conf={conf:.2f} iou={iou:.2f} imgsz={imgsz} device={dev or 'auto'} stride={stride}")

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        st.error(f"Could not open source: {source_label}")
        st.stop()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 120:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 0:
        total = 0

    names = getattr(model, "names", {})
    if save_output:
        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_output.close()
        out_path = Path(tmp_output.name)
        # Writer created lazily once frame size is known (webcam may report 0x0 up front).
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_ph = st.empty()
    prog = st.progress(0, text="Starting…")
    stats_ph = st.empty()

    frame_idx = 0
    processed = 0
    infer_ms = []
    t0 = time.time()
    last_frame_rgb = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        # File uploads respect Max frames; live (webcam/RTSP/URL) is uncapped — stop via Streamlit's Stop button.
        if not is_live and max_frames and frame_idx > max_frames:
            break

        if writer is None and out_path is not None:
            fh, fw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

        do_infer = (frame_idx - 1) % stride == 0
        if do_infer:
            t1 = time.time()
            results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            t2 = time.time()
            infer_ms.append((t2 - t1) * 1000)
            r = results[0]
            annotate_frame(frame, r.boxes, names)
            processed += 1
            cv2.putText(frame, f"frame {frame_idx}" + (f"/{total}" if total else ""), (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        # Throttle Streamlit updates: show every strided frame (or every frame if stride==1).
        if do_infer or stride == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame_rgb = rgb
            frame_ph.image(rgb, caption=f"Frame {frame_idx}" + (f" / {total}" if total else ""), width="stretch")
            if total > 0:
                val = frame_idx / total
                val = 0.0 if val < 0 else (1.0 if val > 1 else val)
                prog.progress(val, text=f"Processing {frame_idx}/{total}")
            elif not is_live and max_frames:
                val = frame_idx / max_frames
                val = 0.0 if val < 0 else (1.0 if val > 1 else val)
                prog.progress(val, text=f"Processing {frame_idx}" + (f"/{max_frames}" if max_frames else ""))
            else:
                # Live / uncapped: no deterministic progress, show running count.
                prog.progress(0, text=f"Live — frame {frame_idx} (stop with Streamlit's Stop button)" if is_live else f"Processing frame {frame_idx}")

        if processed and processed % 30 == 0:
            avg = sum(infer_ms[-30:]) / min(30, len(infer_ms))
            stats_ph.caption(f"Processed {processed} frames · avg infer {avg:.1f} ms (last 30)")

    elapsed = time.time() - t0
    avg_ms = sum(infer_ms) / len(infer_ms) if infer_ms else 0
    prog.progress(1.0, text="Done")
    if frame_idx == 0:
        st.warning("No frames read from source.")
    else:
        st.success(f"Done — {frame_idx} frames read · {processed} inferred · {elapsed:.1f}s · avg infer {avg_ms:.1f} ms · {processed / elapsed:.1f} fps (inferred)")

    if out_path is not None and writer is not None:
        try:
            writer.release()
        except Exception:
            pass
        writer = None
        if out_path.exists() and out_path.stat().st_size > 0:
            data = out_path.read_bytes()
            st.video(data)
            st.download_button("⬇ Download annotated video", data=data, file_name=f"{Path(source_label).stem}_annotated.mp4", mime="video/mp4")
        else:
            st.warning("Annotated video could not be written (codec not available or no frames).")

finally:
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        if writer is not None:
            writer.release()
    except Exception:
        pass
    # Temp files: keep output for download (Streamlit serves from bytes already), but clean input.
    if tmp_input is not None:
        try:
            Path(tmp_input.name).unlink(missing_ok=True)
        except Exception:
            pass
