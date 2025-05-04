import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Sınıf isimlerini buraya yaz
class_names = ['Elma_Karalekesi', 'Elma_Saglikli', 'Domates_ErkenYaprakküfü', 'Domates_Saglikli']  # Örnek

# Modeli yükle
model = tf.keras.models.load_model("plant_diesase_model.h5")

st.title("Bitki Hastalık Tahmin Uygulaması")
st.write("Bir yaprak resmi yükleyin, model tahmin etsin.")

uploaded_file = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Görsel', use_column_width=True)

    # Görseli modele uygun hale getir
    img = image.resize((224, 224))  # Modelin giriş boyutuna göre değiştir
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Batch boyutu ekle

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.write(f"### Tahmin: {predicted_class}")
    st.write(f"Güven: {confidence:.2f}")
