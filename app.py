import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

# ------------------ Model Setup ------------------
MODEL_PATH = "glaucoma_model.h5"  # make sure this is in the same folder

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found! Please place {MODEL_PATH} in this folder.")
    st.stop()

@st.cache_resource
def load_glaucoma_model():
    return load_model(MODEL_PATH)

model = load_glaucoma_model()
IMG_SIZE = (224, 224)

# ------------------ Preprocessing ------------------
def crop_retina(img):
    gray = img.convert("L")
    np_gray = np.array(gray)
    mask = np_gray > 10
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img.crop((x0, y0, x1, y1))

def predict_image(img):
    img = crop_retina(img)
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        result = "Normal Retina"
        confidence = prediction * 100
    else:
        result = "Glaucoma Detected"
        confidence = (1 - prediction) * 100
    return result, round(confidence, 2), img

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Glaucoma Detection", page_icon="🩺", layout="wide")

# Neon-style header
st.markdown("""
<div style="background-color:#0b0f1a; padding:20px; border-radius:12px;">
<h1 style="color:#00FFC6; text-align:center; font-family:Orbitron;">Glaucoma Detection AI</h1>
<h4 style="color:#E0E0E0; text-align:center; font-family:Segoe UI;">Futuristic Retina Analysis</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Retinal Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Retina", use_column_width=True)
    
    with st.spinner("🔬 Analyzing retina..."):
        result, confidence, processed_img = predict_image(img)
    
    with col2:
        st.image(processed_img, caption="Cropped + Resized Retina", use_column_width=True)

    # Result display
    st.markdown("<br>", unsafe_allow_html=True)
    if "Glaucoma" in result:
        st.markdown(f"<h2 style='color:#FF5252; text-align:center;'>🚨 {result}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#00E676; text-align:center;'>✅ {result}</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h4 style='color:#00FFC6; text-align:center;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; color:#4FC3F7; padding-top:20px; font-family:Consolas;">
Powered by Deep Learning | Designed for Ophthalmologists 👁️
</div>
""", unsafe_allow_html=True)
