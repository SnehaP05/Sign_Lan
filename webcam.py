import streamlit as st
import os
import gdown
from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------
# 📥 Load model
# ---------------------------
MODEL_URL = "https://drive.google.com/file/d/1DBjJuUmicuUHrSLchCvpbG8oJK9DOwZo/view?usp=drive_link"

if not os.path.exists("yolo26n.pt"):
    gdown.download(MODEL_URL, "yolo26n.pt", quiet=False)

model = YOLO("yolo26n.pt")

# ---------------------------
# UI
# ---------------------------
st.title("🖐️ Sign Language Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model.predict(img, conf=0.4)

    annotated = results[0].plot()

    st.image(annotated, channels="BGR")

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detections.append({
            "class": model.names[cls],
            "confidence": float(box.conf[0])
        })

    st.json(detections)
