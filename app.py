import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import h5py

# Sayfa yapılandırması
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
st.write("Dosya boyutu:", os.path.getsize(MODEL_PATH))

# Model dosyasının içeriğini kontrol et
try:
    with h5py.File(MODEL_PATH, 'r') as f:
        st.write("Model dosyası içeriği:", list(f.keys()))
        for key in f.keys():
            st.write(f"Key: {key}, Shape: {f[key].shape if hasattr(f[key], 'shape') else 'No shape'}")
except Exception as e:
    st.error(f"Model dosyası okunurken hata oluştu: {str(e)}")

# Modeli yükle
@st.cache_resource
def load_model():
    try:
        st.write("Model yükleniyor...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        st.success("Model başarıyla yüklendi!")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

# Modeli yükle
model = load_model()
if model is None:
    st.stop()

# Sınıf isimleri
class_names = ['Elma_Karalekesi', 'Elma_Saglikli', 'Domates_ErkenYaprakküfü', 'Domates_Saglikli']

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
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Görsel', use_column_width=True)

    # Görüntüyü işle
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    # Tahmin yap
    with st.spinner('Tahmin yapılıyor...'):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100

    # Sonucu göster
    st.markdown(f"""
        <div style='background-color: #dff0d8; color: #3c763d; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h3>🌱 Tahmin Sonucu</h3>
            <p>Durum: <strong>{predicted_class}</strong></p>
            <p>Güven: <strong>{confidence:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

# Alt bilgi
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>© 2024 Bitki Hastalığı Tespiti</p>
    </div>
""", unsafe_allow_html=True)
