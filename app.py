import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

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

# Model oluştur ve kaydet
@st.cache_resource
def create_and_save_model():
    try:
        # Model oluştur
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Modeli kaydet
        model.save(MODEL_PATH)
        st.success("Model başarıyla oluşturuldu ve kaydedildi!")
        return model
    except Exception as e:
        st.error(f"Model oluşturulurken hata oluştu: {str(e)}")
        return None

# Modeli yükle veya oluştur
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            st.write("Model yükleniyor...")
            st.write("Dosya boyutu:", os.path.getsize(MODEL_PATH))
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("Model başarıyla yüklendi!")
            return model
        else:
            st.info("Model dosyası bulunamadı. Yeni model oluşturuluyor...")
            return create_and_save_model()
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        st.info("Yeni model oluşturuluyor...")
        return create_and_save_model()

# Modeli yükle veya oluştur
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
