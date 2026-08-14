import streamlit as st
import os
import cv2
import numpy as np
import torch
import web_app

# Set up page configurations
st.set_page_config(page_title="Universal Image Restoration", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Universal Image Restoration")
st.write("Upload your images to automatically restore and enhance blur, noise, and low-light degradation using pre-trained Deep Learning models.")

# Checkpoints existence check
missing_checkpoints = []
for cp in ["checkpoints/nafnet_gopro.pth", "checkpoints/restormer_motion.pth", "checkpoints/restormer_defocus.pth", "checkpoints/mprnet_deblurring.pth"]:
    if not os.path.exists(cp):
        missing_checkpoints.append(cp)

if missing_checkpoints:
    st.error(f"Missing model checkpoints: {missing_checkpoints}. Please ensure checkpoints are pushed to your repository.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

if uploaded_files and not missing_checkpoints:
    st.write(f"Loaded **{len(uploaded_files)}** images.")
    if st.button("Start Restoration", type="primary"):
        for uploaded_file in uploaded_files:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            st.markdown(f"### Processing File: `{uploaded_file.name}`")
            
            with st.spinner("Analyzing image and running restoration model..."):
                # Classify degradation
                model_used = web_app.classify_degradation(img_bgr, uploaded_file.name)
                
                # Inference routing
                if model_used == "nafnet_gopro":
                    restored = web_app.run_nafnet_inference(img_rgb)
                    badge_style = '<span class="badge" style="background-color: #28a745;">NAFNet-GoPro (Motion Blur)</span>'
                elif model_used == "restormer_defocus":
                    restored = web_app.run_restormer_inference(web_app.restormer_defocus, img_rgb)
                    badge_style = '<span class="badge" style="background-color: #007bff;">Restormer-Defocus (Defocus Blur)</span>'
                elif model_used == "restormer_motion":
                    restored = web_app.run_restormer_inference(web_app.restormer_motion, img_rgb)
                    badge_style = '<span class="badge" style="background-color: #6f42c1;">Restormer-Motion (General Blur)</span>'
                else:
                    restored = web_app.run_mprnet_inference(img_rgb)
                    badge_style = '<span class="badge" style="background-color: #ffc107; color: black;">MPRNet-Deblur (Severe/Mixed)</span>'
            
            st.markdown(f"**Restoration Method Routed:** {badge_style}", unsafe_allow_html=True)
            
            # Show original and restored side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Original Blurry Image", use_container_width=True)
            with col2:
                st.image(restored, caption="Restored Enhanced Image", use_container_width=True)
            
            # Download button for individual restored image
            _, restored_bytes = cv2.imencode('.png', cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
            st.download_button(
                label=f"Download Restored {uploaded_file.name}",
                data=restored_bytes.tobytes(),
                file_name=f"restored_{uploaded_file.name}",
                mime="image/png"
            )
            st.markdown("---")
