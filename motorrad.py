import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page config
st.set_page_config(
    page_title="Motorrad KI",
    page_icon="🏍️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 30px;
    }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🏍️ Motorrad-Erkennung</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Erkenne Motorradmodelle mit KI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    st.write("Lade ein Bild hoch und die KI erkennt das Motorradmodell.")
    st.write("Unterstützte Formate: JPG, PNG")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# Load labels
@st.cache_data
def load_labels():
    with open("labels.txt") as f:
        return [line.strip() for line in f]

labels = load_labels()

# Load bike data
@st.cache_data
def load_data():
    with open("bikes.json") as f:
        return json.load(f)

bike_data = load_data()

# Upload section
uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Dein Bild", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index])
    label = labels[class_index]

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🔍 Ergebnis")
        st.markdown(f"### {label}")

        # Confidence bar
        st.progress(confidence)
        st.write(f"**Confidence:** {confidence:.2%}")

        # Technical data
        if label in bike_data:
            st.markdown("### ⚙️ Technische Daten")
            for key, value in bike_data[label].items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("Keine Daten gefunden.")

        st.markdown('</div>', unsafe_allow_html=True)
