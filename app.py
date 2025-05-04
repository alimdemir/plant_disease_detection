import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import h5py

# Sayfa yapılandırması (ilk komut olmalı)
st.set_page_config(
    page_title="Bitki Hastalığı Tespiti",
    page_icon="🌿",
    layout="centered"
)

# Debug bilgisi
st.write("Çalışma dizini:", os.getcwd())
st.write("Dosya listesi:", os.listdir())

# Model dosyasının yolunu belirle
MODEL_PATH = os.path.join(os.getcwd(), "plant_diesase_model.h5")
st.write("Model dosyası yolu:", MODEL_PATH)
st.write("Model dosyası var mı:", os.path.exists(MODEL_PATH))

# Model dosyasının içeriğini kontrol et
try:
    with h5py.File(MODEL_PATH, 'r') as f:
        st.write("Model dosyası içeriği:", list(f.keys()))
except Exception as e:
    st.error(f"Model dosyası okunurken hata oluştu: {str(e)}")

# Modeli yükle
@st.cache_resource
def load_plant_model():
    try:
        st.write("Model yükleniyor...")
        st.write("Dosya boyutu:", os.path.getsize(MODEL_PATH))
        # Modeli yükle
        model = load_model(MODEL_PATH, compile=False)
        # Modeli derle
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

model = load_plant_model()
if model is None:
    st.stop()

class_names = ["Sağlıklı", "Hastalıklı"]
img_height, img_width = 224, 224

# CSS stilleri
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success {
        background-color: #dff0d8;
        color: #3c763d;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve açıklama
st.title("🌿 Bitki Hastalığı Tespiti")
st.markdown("""
    Bu uygulama, bitki yapraklarının sağlıklı olup olmadığını tespit etmenize yardımcı olur.
    Lütfen bir yaprak fotoğrafı yükleyin.
""")

# Dosya yükleme alanı
uploaded_file = st.file_uploader(
    "Bir yaprak fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"],
    help="Desteklenen formatlar: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Görüntüyü yükle ve göster
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Resim', use_column_width=True)

    # Görüntüyü işle
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Tahmin yap
    with st.spinner('Tahmin yapılıyor...'):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

    # Sonucu göster
    st.markdown(f"""
        <div class="success">
            <h3>🌱 Tahmin Sonucu</h3>
            <p>Durum: <strong>{predicted_class}</strong></p>
            <p>Güven: <strong>{confidence:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

# Alt bilgi
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>© 2024 Bitki Hastalığı Tespiti | Geliştirici: [İsminiz]</p>
    </div>
""", unsafe_allow_html=True)
