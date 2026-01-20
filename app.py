import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

from utils.detector import testing
from utils.audio import generate_audio

st.set_page_config(
    page_title="Railway Obstacle Detection",
    page_icon="🚆",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f3f4f6; 
    color: #111827;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
    color: #111827 !important;
}

.app-title {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
    text-align: center;
    margin-bottom: 4px;
}

.app-subtitle {
    font-size: 15px;
    color: #374151;
    text-align: center;
    margin-bottom: 28px;
}

.safe-text {
    color: #15803d; 
    font-size: 18px;
    font-weight: 600;
}

.danger-text {
    color: #b91c1c; 
    font-size: 18px;
    font-weight: 600;
}

.stButton > button {
    background-color: #2563eb;
    color: #ffffff !important;
    font-size: 16px;
    font-weight: 600;
    padding: 0.65em;
    border-radius: 8px;
    width: 100%;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

.stButton > button:disabled {
    background-color: #c7c9d1 !important;  
    color: #374151 !important;             
    opacity: 1 !important;               
    cursor: not-allowed;
    border: 1px solid #9ca3af;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-title">Railway Obstacle Detection Dashboard</div>', 
    unsafe_allow_html=True
    )
st.markdown(
    '<div class="app-subtitle">Real-time AI monitoring system for railway track safety</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    return (
        YOLO("model/track.pt"),
        YOLO("model/obstacle.pt")
    )
track_model, obstacle_model = load_models()

input_col, output_col = st.columns([1, 1.5])
with input_col:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader(
        "Upload railway track image",
        type=["jpg", "png", "jpeg"]
    )
    run_btn = st.button("▶ Run Detection",disabled=not uploaded_file)

with output_col:
    st.subheader("Detection Results")
    st.markdown("")
    st.markdown("")
    if not uploaded_file:
        st.info("Upload an image and then click **Run Detection**.")

    elif uploaded_file and not run_btn:
        st.success("Image uploaded successfully. Click **Run Detection**.")

    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner("Running AI safety analysis..."):
            output, msg, status = testing(
                image,
                track_model,
                obstacle_model
            )

        img_left, img_right = st.columns(2)
        with img_left:
            st.image(image, channels="BGR", caption="Input Image")
        with img_right:
            st.image(output, channels="BGR", caption="Annotated Output")
        
        if status == "danger":
            st.markdown(
                f'<div class="danger-text">{msg}</div>',
                unsafe_allow_html=True
            )
            with st.spinner("generating audio..."):
                audio = generate_audio(msg)
            st.audio(audio, format="audio/mp3")
            
        else:
            st.markdown(
                f'<div class="safe-text">{msg}</div>',
                unsafe_allow_html=True
            )