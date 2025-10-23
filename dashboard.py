import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests

# === Konfigurasi dasar halaman ===
st.set_page_config(page_title="Dashboard Prediksi AI", page_icon="🐾", layout="wide")

# === CSS untuk background dan gaya ===
page_bg = """
<style>
/* Background utama */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1518791841217-8f162f1e1131");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

/* Sidebar dibuat transparan gelap elegan */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.75);
    color: white;
}

/* Judul */
h1 {
    color: #FFD700;
    text-shadow: 2px 2px 5px #000000;
}

/* Teks umum */
p, label, span, div {
    color: #f8f8f8 !important;
}

/* Upload box transparan */
section[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255, 255, 255, 0.1);
    border: 2px dashed #FFD700;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# === Sidebar Navigasi ===
st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📤 Upload Gambar", "🔮 Prediksi", "ℹ️ Tentang Model"])

# === Fungsi Animasi ===
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animal = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_pprxh53t.json")

# === Halaman Home ===
if menu == "🏠 Home":
    st_lottie(lottie_animal, height=200, key="animal")
    st.markdown("<h1>🐾 Dashboard Prediksi Model AI</h1>", unsafe_allow_html=True)
    st.write("Selamat datang di dashboard prediksi berbasis citra bertema hewan 🐶🐱🐰! Gunakan navigasi di samping kiri untuk mengunggah gambar dan melihat hasil prediksi.")

# === Halaman Upload Gambar ===
elif menu == "📤 Upload Gambar":
    st.markdown("<h1>📤 Upload Gambar</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah gambar untuk diprediksi", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang kamu unggah", use_container_width=True)
        st.session_state['uploaded_file'] = uploaded_file
        st.success("✅ Gambar berhasil diunggah!")

# === Halaman Prediksi ===
elif menu == "🔮 Prediksi":
    st.markdown("<h1>🔮 Hasil Prediksi</h1>", unsafe_allow_html=True)
    if 'uploaded_file' in st.session_state:
        st.image(st.session_state['uploaded_file'], caption="Gambar yang diprediksi", use_container_width=True)
        st.info("✨ Hasil prediksi akan tampil di sini setelah model AI dijalankan.")
    else:
        st.warning("⚠️ Harap unggah gambar terlebih dahulu di menu 'Upload Gambar'!")

# === Halaman Tentang Model ===
elif menu == "ℹ️ Tentang Model":
    st.markdown("<h1>ℹ️ Tentang Model</h1>", unsafe_allow_html=True)
    st.write("""
    Dashboard ini dikembangkan oleh **Repa Cantikk ✨**,  
    menampilkan hasil prediksi model AI untuk mengenali citra bertema hewan.  
    Dikembangkan dengan framework **Streamlit** dan **TensorFlow** 🧠🐾.
    """)

