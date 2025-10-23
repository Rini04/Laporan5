import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==============================
# ⚙️ Konfigurasi Dashboard
# ==============================
st.set_page_config(page_title="Dashboard Prediksi Model", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f, #12121c);
        color: white;
        padding-top: 2rem;
    }

    /* Judul Sidebar */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f5f5f5;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Navigasi Tombol */
    .stButton button {
        background-color: #26263a !important;
        color: #f5f5f5 !important;
        border-radius: 12px !important;
        border: 1px solid #444;
        transition: 0.3s;
    }

    .stButton button:hover {
        background-color: #5a5ad1 !important;
        color: white !important;
        border-color: #5a5ad1 !important;
        transform: scale(1.02);
    }

    /* Kartu Status */
    .status-card {
        background: rgba(40, 40, 55, 0.9);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        color: #f0f0f0;
    }

    .status-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffd166;
    }

    .status-text {
        margin-top: 10px;
        font-size: 1.05rem;
        color: #d3d3d3;
    }

    </style>
""", unsafe_allow_html=True)

# ==============================
# 🧭 Sidebar Navigasi
# ==============================
st.sidebar.markdown("<div class='sidebar-title'>🧭 Navigasi</div>", unsafe_allow_html=True)
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📤 Upload Gambar", "🔍 Prediksi", "ℹ️ Tentang Model"])

# ==============================
# 📦 Load Model
# ==============================
model_path = "model/model_Rini_Laporan 2.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
else:
    st.error("❌ Model tidak ditemukan di direktori 'model/'.")
    st.stop()

# ==============================
# 🏠 HALAMAN HOME
# ==============================
if menu == "🏠 Home":
    st.markdown("### 🧠 Dashboard Prediksi Model AI")
    st.markdown("Selamat datang di dashboard interaktif untuk melakukan prediksi berbasis citra.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div class='status-card'>
            <div class='status-title'>📦 Status Model</div>
            <div class='status-text'>✅ Model berhasil dimuat.</div>
            <div class='status-text'>📁 Lokasi: <b>model/model_Rini_Laporan 2.h5</b></div>
            <div class='status-text'>🧩 Input shape: <b>{}</b></div>
        </div>
        """.format(input_shape), unsafe_allow_html=True)

    with col2:
        st.image("https://cdn.dribbble.com/users/1129231/screenshots/16049066/media/54e84bb2e9d2d0e7b55cc02b28514c56.png",
                 caption="Contoh Tampilan Dashboard AI", use_container_width=True)

# ==============================
# 📤 HALAMAN UPLOAD GAMBAR
# ==============================
elif menu == "📤 Upload Gambar":
    st.header("📤 Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar untuk prediksi", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar Diupload", use_container_width=True)
        st.success("✅ Gambar berhasil diunggah. Silakan lanjut ke tab *Prediksi* untuk melihat hasil.")

# ==============================
# 🔍 HALAMAN PREDIKSI
# ==============================
elif menu == "🔍 Prediksi":
    st.header("🔍 Hasil Prediksi Model")
    uploaded_file = st.file_uploader("Upload ulang jika belum ada gambar:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((128, 128))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        preds = model.predict(img_array)
        st.image(image, caption="Gambar yang Diprediksi", use_container_width=True)
        st.success(f"✅ Hasil Prediksi: {np.argmax(preds)}")

# ==============================
# ℹ️ HALAMAN TENTANG MODEL
# ==============================
elif menu == "ℹ️ Tentang Model":
    st.header("ℹ️ Tentang Model")
    st.markdown("""
    Model ini dilatih untuk melakukan klasifikasi gambar berbasis CNN.
    - Framework: **TensorFlow / Keras**
    - Input shape: {}
    - Status: **Aktif**
    """.format(input_shape))
