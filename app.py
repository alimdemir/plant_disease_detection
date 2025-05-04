import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Modeli yükle
model = load_model("bitki_modeli.h5")
class_names = ["Saglikli", "Hastalikli"]  # Kendi sınıf etiketlerini buraya yaz
img_height, img_width = 224, 224  # Modelin eğitimde kullandığı boyut

st.title("🌿 Bitki Hastalığı Tespiti")

uploaded_file = st.file_uploader("Bir yaprak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Resim', use_column_width=True)

    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🌱 Tahmin: **{predicted_class}**")
