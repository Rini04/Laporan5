import streamlit as st
from PIL import Image

# ---- Konfigurasi Halaman ----
st.set_page_config(
    page_title="Dashboard Prediksi Model AI 🐾",
    page_icon="🐶",
    layout="wide",
)

# ---- Background Image (pakai CSS) ----
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2017/02/20/18/03/dog-2083492_1280.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(20, 20, 30, 0.95);
}
h1 {
    color: #fff;
    text-shadow: 2px 2px 6px #000;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---- Judul dengan Animasi ----
st.markdown("""
    <h1 style='text-align:center; font-size:50px; animation: bounce 1.5s infinite;'>
        🐾 Dashboard Prediksi Model AI 🦊
    </h1>
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
""", unsafe_allow_html=True)

st.write("Selamat datang di dashboard interaktif berbasis AI dengan tema binatang 🐕🐈")

# ---- Hapus Status Model ----
# (Tidak ada lagi kotak status model ditampilkan)

# ---- Contoh Tampilan / Gambar ----
st.image("https://cdn.pixabay.com/photo/2016/02/19/11/53/dog-1209621_1280.jpg", caption="Contoh Tampilan Dashboard AI Bertema Hewan 🐾")
