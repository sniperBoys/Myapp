import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Dual Camera App", page_icon="📸")

st.title("📸 Front + Back Camera App")

# Camera selection
camera_choice = st.radio(
    "Select Camera:",
    ["📱 Front Camera", "📷 Back Camera"],
    horizontal=True
)

# Set facing mode
if "Front" in camera_choice:
    facing_mode = "user"
else:
    facing_mode = "environment"

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
    
    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": facing_mode},
        "audio": False,
    },
)

# Capture button
if st.button("📸 Take Photo", use_container_width=True):
    if webrtc_ctx.video_processor:
        frame = webrtc_ctx.video_processor.frame
        if frame is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            
            # Display captured image
            st.image(img, caption=f"Photo from {camera_choice}", use_container_width=True)
            
            # MATLAB style output
            img_np = np.array(img)
            st.write("---")
            st.write("### MATLAB camnet() Output:")
            st.write(f"**size of img:** {img_np.shape}")
            st.write(f"**class of img:** {type(img_np).__name__}")
            st.write(f"**min of img:** {img_np.min()}")
            st.write(f"**max of img:** {img_np.max()}")
            
            st.success(f"✅ Captured from {camera_choice}")
        else:
            st.warning("Please wait for camera to start")
    else:
        st.warning("Camera not started yet")

st.info(f"📷 Current camera: **{camera_choice}**")
