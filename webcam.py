import streamlit as st
import os
import gdown
from ultralytics import YOLO
import sys
st.write(sys.version)

# ✅ Safe import for OpenCV
try:
    import cv2
except ImportError:
    st.error("❌ OpenCV failed to load. Check requirements.txt and Python version.")
    st.stop()

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------------------
# 📥 Download model if needed
# ---------------------------
MODEL_URL = "https://drive.google.com/file/d/1DBjJuUmicuUHrSLchCvpbG8oJK9DOwZo/view?usp=drive_link"  # 🔥 replace

if not os.path.exists("yolo26n.pt"):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, "yolo26n.pt", quiet=False)

# ---------------------------
# 🤖 Load model
# ---------------------------
model = YOLO("yolo26n.pt")

# ---------------------------
# 🎨 UI Setup
# ---------------------------
st.set_page_config(layout="wide")
st.title("🖐️ Sign Language Detection")
st.markdown("Real-time detection using YOLOv8 + Streamlit")

col1, col2 = st.columns([2, 1])
json_box = col2.empty()

# ---------------------------
# 🎥 Video Processor
# ---------------------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.4, verbose=False)

        detections = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append({
                        "class": label,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

                    # 🎯 Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0), 2)

        # 📊 Update JSON panel
        if detections:
            json_box.json(detections)
        else:
            json_box.write("No detection")

        return img

# ---------------------------
# 🚀 Start Webcam
# ---------------------------
with col1:
    webrtc_streamer(
        key="sign-detection",
        video_transformer_factory=VideoProcessor
    )
