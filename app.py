import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests

# Sayfa yapılandırması
st.set_page_config(
    page_title="Bitki Hastalığı Tespiti",
    page_icon="🌿",
    layout="centered"
)

# Model dosyasının yolunu belirle
MODEL_PATH = "plant_diesase_model.h5"

# Google Drive'dan model indirme
@st.cache_resource
def download_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Model dosyası indiriliyor...")
            
            # Google Drive linki
            url = "https://drive.google.com/uc?export=download&id=1yHv9PV0KlezrKTIVg6yhBf9QM980EfhX"
            
            # Dosyayı indir
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Büyük dosyalar için onay sayfası kontrolü
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = f'https://drive.google.com/uc?export=download&confirm={value}&id=1yHv9PV0KlezrKTIVg6yhBf9QM980EfhX'
                    response = session.get(url, stream=True)
                    break
            
            # Dosyayı indir
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    progress = int(50 * downloaded / total_size) if total_size > 0 else 0
                    st.write(f"İndiriliyor: [{'=' * progress}{' ' * (50 - progress)}] {downloaded}/{total_size} bytes")
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(MODEL_PATH)
            st.write(f"İndirilen dosya boyutu: {file_size} bytes")
            
            if file_size < 1000:  # Eğer dosya çok küçükse
                st.error("İndirilen dosya çok küçük, muhtemelen indirme başarısız oldu.")
                return False
            
            st.success("Model dosyası başarıyla indirildi!")
            return True
        return True
    except Exception as e:
        st.error(f"Model indirilirken hata oluştu: {str(e)}")
        return False

# Modeli indir
if not download_model():
    st.stop()

# Modeli yükle
@st.cache_resource
def load_model():
    try:
        st.write("Model yükleniyor...")
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model başarıyla yüklendi!")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

# Modeli yükle
model = load_model()
if model is None:
    st.stop()

# Model özeti
st.write("Model Özeti:")
model.summary(print_fn=lambda x: st.text(x))

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
