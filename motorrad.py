import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Titel
st.title("🏍️ Motorrad-Erkennung mit KI")

# Modell laden
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/model.h5")

model = load_model()

# Labels laden
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Datenbank laden
with open("data/bikes.json") as f:
    bike_data = json.load(f)

# Bild hochladen
uploaded_file = st.file_uploader("Lade ein Motorradbild hoch", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    label = labels[class_index]

    st.subheader(f"Erkanntes Modell: {label}")
    st.write(f"Vertrauen: {confidence:.2f}")

    # Daten anzeigen
    if label in bike_data:
        st.write("### Technische Daten:")
        st.json(bike_data[label])
    else:
        st.warning("Keine Daten gefunden.")
