import streamlit as st
import cv2
import os
import gdown
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------------------
# 📥 Download model if not exists
# ---------------------------
MODEL_URL = "https://drive.google.com/file/d/1E_RI5Tc_7uchA6n64opvWPatNHBgeoa3/view?usp=drive_link"  # 🔥 replace

if not os.path.exists("best.pt"):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, "best.pt", quiet=False)

# ---------------------------
# 🤖 Load model
# ---------------------------
model = YOLO("best.pt")

# ---------------------------
# 🎨 UI Setup
# ---------------------------
st.set_page_config(layout="wide")
st.title("🖐️ Sign Language Detection")
st.markdown("Real-time detection using YOLOv8 + Streamlit")

col1, col2 = st.columns([2, 1])
json_placeholder = col2.empty()

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

                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,0), 2)

        # Show JSON
        if detections:
            json_placeholder.json(detections)
        else:
            json_placeholder.write("No detection")

        return img

# ---------------------------
# 🚀 Start Webcam
# ---------------------------
with col1:
    webrtc_streamer(
        key="sign-detection",
        video_transformer_factory=VideoProcessor
    )