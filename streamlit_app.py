"""PlantDoc YOLOv8 Streamlit app: try included samples or upload images and videos.

Launch:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import gdown
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "sample_tests"
SAMPLE_IMAGE_PATH = SAMPLES_DIR / "image_test.jpg"
SAMPLE_VIDEO_PATH = SAMPLES_DIR / "video_test.mp4"


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(file_id: str) -> YOLO:
    """Download publicly shared Google Drive weights once, then load YOLO."""
    # Incorporate the Drive ID in the filename so changing models cannot
    # accidentally reuse an older model's saved best.pt file.
    model_tag = hashlib.sha256(file_id.encode("utf-8")).hexdigest()[:16]
    model_path = APP_DIR / "models" / f"plantdoc_{model_tag}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        temp_path = model_path.with_suffix(".download")
        try:
            downloaded = gdown.download(
                id=file_id,
                output=str(temp_path),
                quiet=True,
            )
            if not downloaded or not temp_path.is_file() or temp_path.stat().st_size == 0:
                raise RuntimeError(
                    "Google Drive model download failed. Confirm the sharing "
                    "permission is 'Anyone with the link — Viewer'."
                )
            temp_path.replace(model_path)
        finally:
            temp_path.unlink(missing_ok=True)

    return YOLO(str(model_path))


def infer_image(model: YOLO, image: Image.Image, confidence: float):
    corrected_pil = ImageOps.exif_transpose(image).convert("RGB")
    result = model.predict(corrected_pil, conf=confidence, imgsz=640, verbose=False)[0]
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return result, annotated_rgb


def infer_video(
    model: YOLO, source_bytes: bytes, filename: str, confidence: float,
    preview_every: int = 15,
) -> tuple[bytes, bool, int]:
    """Run inference on each video frame; return encoded video bytes and frame count."""
    ext = Path(filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".m4v"}:
        ext = ".mp4"
    with tempfile.TemporaryDirectory(prefix="plantdoc_video_") as tmp:
        temp = Path(tmp)
        input_path = temp / f"input{ext}"
        raw_output_path = temp / "annotated_raw.mp4"
        h264_output_path = temp / "annotated_h264.mp4"
        input_path.write_bytes(source_bytes)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("Unable to decode the video. Try an H.264 MP4 file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width < 1 or height < 1:
            cap.release()
            raise ValueError("The video has invalid dimensions.")
        if not 1 <= fps <= 240:
            fps = 25.0

        writer = cv2.VideoWriter(
            str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize the MP4 writer.")

        progress = st.progress(0, text="Analyzing video frames...")
        preview = st.empty()
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
                annotated_bgr = result.plot()
                writer.write(annotated_bgr)
                processed += 1
                if processed == 1 or processed % preview_every == 0:
                    preview.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        caption=f"Processing frame {processed}",
                        use_container_width=True,
                    )
                    if total > 0:
                        progress.progress(
                            min(1.0, processed / total),
                            text=f"Processed {processed} of approximately {total} frames",
                        )
        finally:
            cap.release()
            writer.release()

        if processed == 0:
            raise ValueError("The video contains no readable frames.")
        preview.empty()
        progress.progress(1.0, text=f"Finished analyzing {processed} frames")

        # MP4V videos often cannot play in browsers; transcode to H.264 if available.
        if shutil.which("ffmpeg"):
            command = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_output_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                str(h264_output_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return h264_output_path.read_bytes(), True, processed
            except (OSError, subprocess.CalledProcessError):
                pass
        return raw_output_path.read_bytes(), False, processed


st.set_page_config(
    page_title="PlantDoc Disease Detection", page_icon="🌿", layout="wide"
)
st.title("🌿 Plant Disease Detection")
st.caption(
    "Explore YOLOv8 leaf detection with a built-in example or your own media. "
    "Predictions are not a definitive plant-disease diagnosis."
)

st.sidebar.caption("Model source: Google Drive")
confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)

try:
    gdrive_file_id = str(st.secrets["GDRIVE_FILE_ID"]).strip()
except (FileNotFoundError, KeyError):
    st.error(
        "Google Drive model ID is missing. Add GDRIVE_FILE_ID to "
        ".streamlit/secrets.toml locally or your deployment's Secrets settings."
    )
    st.stop()

if not gdrive_file_id:
    st.error("GDRIVE_FILE_ID cannot be empty.")
    st.stop()

try:
    model = load_model(gdrive_file_id)
except Exception as exc:
    st.error(f"Unable to download or load the YOLO model: {exc}")
    st.stop()

if mode == "Image":
    st.subheader("Image detection")
    source = st.radio(
        "Choose an image source", ["Try sample image", "Upload an image"],
        horizontal=True, key="image_source",
    )
    image: Image.Image | None = None
    if source == "Try sample image":
        if SAMPLE_IMAGE_PATH.is_file():
            image = Image.open(SAMPLE_IMAGE_PATH)
            st.caption(
                "Sample: an image downloaded from the internet used for demonstrating inference."
            )
            st.download_button(
                "Download sample image", SAMPLE_IMAGE_PATH.read_bytes(),
                file_name="plantdoc_sample.jpg", mime="image/jpeg",
                key="sample_image_download",
            )
        else:
            st.info(
                "Sample image not yet prepared. Run "
                "`python prepare_demo_samples.py --dataset plantdoc_yolo_class_aware` "
                "from your project folder, then refresh."
            )
    else:
        uploaded = st.file_uploader(
            "Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.getvalue()))
            except Exception as exc:
                st.error(f"Unable to read image: {exc}")

    if image is not None:
        with st.spinner("Detecting leaves and diseases..."):
            result, annotated = infer_image(model, image, confidence)
        left, right = st.columns(2)
        with left:
            st.image(ImageOps.exif_transpose(image), caption="Original", use_container_width=True)
        with right:
            st.image(annotated, caption="YOLOv8 detections", use_container_width=True)

        detected_classes = [int(x) for x in result.boxes.cls.tolist()]
        counts = Counter(result.names[i] for i in detected_classes)
        if counts:
            st.markdown("**Detected classes**")
            st.table([{"Class": name, "Detections": count} for name, count in counts.most_common()])
        else:
            st.info("No objects detected above the chosen confidence threshold.")

elif mode == "Video":
    st.subheader("Video detection")
    source = st.radio(
        "Choose a video source", ["Try sample video", "Upload a video"],
        horizontal=True, key="video_source",
    )
    video_bytes: bytes | None = None
    video_filename = ""
    if source == "Try sample video":
        if SAMPLE_VIDEO_PATH.is_file():
            video_bytes = SAMPLE_VIDEO_PATH.read_bytes()
            video_filename = SAMPLE_VIDEO_PATH.name
            st.caption(
                "Demo clip: a gentle pan/zoom animated from an image from the internet. "
                "It tests video processing, not detection on real moving footage."
            )
            st.video(video_bytes)
            st.download_button(
                "Download sample video", video_bytes,
                file_name="plantdoc_sample_video.mp4", mime="video/mp4",
                key="sample_video_download",
            )
        else:
            st.info(
                "Sample video not yet prepared. Run "
                "`python prepare_demo_samples.py --dataset plantdoc_yolo_class_aware` "
                "from your project folder, then refresh."
            )
    else:
        uploaded = st.file_uploader(
            "Upload a video", type=["mp4", "mov", "avi", "m4v"],
            key="video_upload",
        )
        if uploaded is not None:
            video_bytes = uploaded.getvalue()
            video_filename = uploaded.name
            st.video(video_bytes)

    st.caption(
        "Each frame is processed and labeled. Exported videos do not retain source audio. "
        "For best browser playback, install FFmpeg (`brew install ffmpeg`)."
    )
    if video_bytes is not None:
        source_id = hashlib.sha256(video_bytes).hexdigest()
        result_id = f"{source_id}:{confidence}:{gdrive_file_id}"
        if st.session_state.get("result_video_id") != result_id:
            for key in ("result_video_bytes", "result_video_h264", "result_video_frames"):
                st.session_state.pop(key, None)

        if st.button("Run video detection", type="primary"):
            try:
                output_bytes, h264, n_frames = infer_video(
                    model, video_bytes, video_filename, confidence,
                )
                st.session_state["result_video_id"] = result_id
                st.session_state["result_video_bytes"] = output_bytes
                st.session_state["result_video_h264"] = h264
                st.session_state["result_video_frames"] = n_frames
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

        if (
            st.session_state.get("result_video_id") == result_id
            and st.session_state.get("result_video_bytes")
        ):
            st.success(f"Processed {st.session_state['result_video_frames']} frames.")
            if st.session_state["result_video_h264"]:
                st.video(st.session_state["result_video_bytes"], format="video/mp4")
            else:
                st.warning(
                    "Annotated MP4 saved, but the preview may not play in every browser. "
                    "Install FFmpeg and process again for H.264 output."
                )
            st.download_button(
                "Download annotated video",
                data=st.session_state["result_video_bytes"],
                file_name="plant_disease_annotated.mp4", mime="video/mp4",
            )
