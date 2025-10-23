import streamlit as st
from streamlit_extras.let_it_rain import rain
from PIL import Image
import time

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Prediksi Hewan AI",
    page_icon="🐾",
    layout="wide"
)

# --- CSS Tampilan ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1508672019048-805c876b67e2?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

h1 {
    font-size: 55px !important;
    text-align: center;
    color: #fff3cd;
    text-shadow: 2px 2px 8px #000000;
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% { text-shadow: 0 0 10px #ffd700; }
    50% { text-shadow: 0 0 20px #ffcc00, 0 0 30px #ff9900; }
    100% { text-shadow: 0 0 10px #ffd700; }
}

.sidebar .sidebar-content {
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Navigasi ---
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📸 Upload Gambar", "🔮 Prediksi", "📚 Fakta Hewan", "ℹ️ Tentang"])

# --- Data Fakta Hewan ---
fakta_hewan = {
    "Kucing": {
        "Habitat": "Kucing sering hidup berdampingan dengan manusia di rumah, lingkungan kota, maupun pedesaan.",
        "Fakta Unik": "Kucing dapat melompat hingga enam kali tinggi tubuhnya sendiri dan memiliki kemampuan melihat dalam gelap.",
        "Klasifikasi Ilmiah": "Kingdom: Animalia | Filum: Chordata | Kelas: Mammalia | Ordo: Carnivora | Famili: Felidae",
        "Status Konservasi": "Domestik (Tidak terancam punah)"
    },
    "Anjing": {
        "Habitat": "Anjing hidup bersama manusia di berbagai lingkungan, dari perkotaan hingga pedesaan.",
        "Fakta Unik": "Anjing dapat mengenali lebih dari 150 kata dan memiliki penciuman 40x lebih tajam dari manusia.",
        "Klasifikasi Ilmiah": "Kingdom: Animalia | Filum: Chordata | Kelas: Mammalia | Ordo: Carnivora | Famili: Canidae",
        "Status Konservasi": "Domestik (Tidak terancam punah)"
    },
    "Kupu-Kupu": {
        "Habitat": "Biasanya hidup di taman, hutan, dan daerah tropis yang banyak bunga.",
        "Fakta Unik": "Kupu-kupu bisa merasakan rasa manis menggunakan kakinya!",
        "Klasifikasi Ilmiah": "Kingdom: Animalia | Filum: Arthropoda | Kelas: Insecta | Ordo: Lepidoptera",
        "Status Konservasi": "Sebagian besar aman, tetapi beberapa spesies langka."
    }
}

# --- Halaman Home ---
if page == "🏠 Home":
    st.markdown("<h1>🐾 Dashboard Prediksi Hewan AI 🦋</h1>", unsafe_allow_html=True)
    st.write(
        "<p style='text-align:center; font-size:20px; color:white;'>"
        "Selamat datang di dashboard prediksi hewan AI karya <b>Rini Safariani</b> 🌸<br>"
        "Temukan kecerdasan buatan yang mampu mengenali hewan dan memberikan informasi menarik tentang mereka."
        "</p>", unsafe_allow_html=True
    )
    rain(emoji="🐾", font_size=30, falling_speed=3, animation_length="infinite")

# --- Halaman Upload ---
elif page == "📸 Upload Gambar":
    st.header("📸 Upload Gambar Hewan")
    uploaded_file = st.file_uploader("Unggah gambar hewan di sini...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        st.success("✅ Gambar berhasil diunggah! Silakan lanjut ke halaman Prediksi.")

# --- Halaman Prediksi ---
elif page == "🔮 Prediksi":
    st.header("🔮 Hasil Prediksi Model AI")
    st.info("Model sedang memproses gambar...")

    time.sleep(2)  # simulasi loading

    prediksi = "Kucing"  # contoh hasil prediksi
    st.success(f"✨ Hasil Prediksi: {prediksi}")

    data = fakta_hewan.get(prediksi)
    if data:
        st.subheader("📍 Habitat")
        st.write(data["Habitat"])
        st.subheader("💡 Fakta Unik")
        st.write(data["Fakta Unik"])
        st.subheader("🔬 Klasifikasi Ilmiah")
        st.write(data["Klasifikasi Ilmiah"])
        st.subheader("🌿 Status Konservasi")
        st.write(data["Status Konservasi"])

# --- Halaman Fakta Hewan ---
elif page == "📚 Fakta Hewan":
    st.header("📚 Koleksi Fakta Hewan")
    for nama, info in fakta_hewan.items():
        st.markdown(f"### 🐾 {nama}")
        st.write(f"**Habitat:** {info['Habitat']}")
        st.write(f"**Fakta Unik:** {info['Fakta Unik']}")
        st.write(f"**Klasifikasi Ilmiah:** {info['Klasifikasi Ilmiah']}")
        st.write(f"**Status Konservasi:** {info['Status Konservasi']}")
        st.markdown("---")

# --- Tentang Model ---
elif page == "ℹ️ Tentang":
    st.header("ℹ️ Tentang Proyek Ini")
    st.write("""
    Proyek **Dashboard Prediksi Hewan AI** ini dibuat oleh **Rini Safariani** 🦋  
    Tujuannya untuk mengembangkan sistem pengenalan citra berbasis AI yang dapat:
    - Mengidentifikasi hewan dari gambar
    - Memberikan informasi habitat, fakta unik, dan klasifikasi ilmiah  
    - Menyajikan tampilan elegan dan interaktif 🌿  
    """)
