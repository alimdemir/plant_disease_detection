import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(
    page_title="Modern Web Uygulaması",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Stil tanımı ---
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.05);
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=180)
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfalar", ["Ana Sayfa", "Veri Analizi", "Hakkında"])

# --- Sayfa 1: Ana Sayfa ---
if page == "Ana Sayfa":
    st.title("🌐 Modern Web Dashboard")
    st.markdown("Hoş geldiniz! Bu sayfa Streamlit ile yapılmış profesyonel görünümlü bir web uygulamasıdır.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✨ Özellikler")
        st.markdown("""
        - Modern UI/UX tasarımı  
        - Gerçek zamanlı veri analizi  
        - Responsive düzen  
        - Sidebar navigasyonu  
        """)
    
    with col2:
        st.subheader("📈 Canlı Grafik")
        df = pd.DataFrame({
            'Aylar': ['Ocak', 'Şubat', 'Mart', 'Nisan'],
            'Satış': [150, 200, 180, 250]
        })
        fig, ax = plt.subplots()
        ax.plot(df['Aylar'], df['Satış'], marker='o', color='#007bff')
        ax.set_title("Aylık Satış Verisi")
        ax.grid(True)
        st.pyplot(fig)

# --- Sayfa 2: Veri Analizi ---
elif page == "Veri Analizi":
    st.title("📊 Veri Analizi")
    uploaded_file = st.file_uploader("CSV dosyası yükle", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        st.markdown("### Sütun Seç ve Görselleştir")
        column = st.selectbox("Sütun Seç", data.select_dtypes(include='number').columns)
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=20, color='#17a2b8')
        ax.set_title(f"{column} Dağılımı")
        st.pyplot(fig)

# --- Sayfa 3: Hakkında ---
elif page == "Hakkında":
    st.title("ℹ️ Hakkında")
    st.markdown("""
    Bu uygulama **Streamlit** kullanılarak geliştirilmiştir.  
    Profesyonel görünümlü, modern bir arayüzle kullanıcı dostu bir deneyim sunar.
    
    Geliştirici: [Senin Adın]  
    LinkedIn: [linkedin.com/in/seninprofilin]  
    """)

