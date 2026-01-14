import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ----------------------------
# Load model locally
# ----------------------------
MODEL_PATH = "vgg16_best_model.keras"  # Make sure this file is in the same folder as this script

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    st.info("Loading model...")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐶🐱",
    layout="centered"
)

st.title("🐶🐱 Dogs vs Cats Classification")
st.write("Upload an image and let the AI decide whether it's a Dog or a Cat.")

# ----------------------------
# Image Upload & Prediction
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file:
    # Open image & convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = np.array(image.resize((150,150)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    prob = prediction[0][0]
    label = "Dog" if prob > 0.5 else "Cat"

    st.success(f"Prediction: {label} ({prob*100:.2f}% confidence)")
