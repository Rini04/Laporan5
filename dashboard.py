import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os

# ================================
# Konfigurasi Awal Halaman
# ================================
st.set_page_config(page_title="Animal Vision AI", layout="wide", page_icon="🐾")

# ================================
# Fungsi untuk Set Background
# ================================
def set_bg_from_local(image_file):
    """Mengatur background Streamlit menggunakan file gambar lokal"""
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Kotak konten semi-transparan agar teks tetap terbaca */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.55);
            border-radius: 15px;
            padding: 2rem;
        }}
        h1, h2, h3, p, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ================================
# Atur Gambar Background
# ================================
# Ganti path di bawah ini sesuai gambar kamu, misal: "images/tiger_bg.jpg"
bg_path = "images/animal_background.jpg"
if os.path.exists(bg_path):
    set_bg_from_local(bg_path)
else:
    st.warning("⚠️ Gambar background belum ditemukan. Simpan gambar di folder `images/`.")

# ================================
# Sidebar Navigasi
# ================================
st.sidebar.title("🐾 Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Status Model", "📷 Prediksi Gambar", "ℹ️ Tentang Aplikasi"]
)

# ================================
# Halaman 1: Status Model
# ================================
if page == "📊 Status Model":
    st.title("📦 Status Model")
    st.markdown("Lihat detail model yang berhasil dimuat di sistem.")

    st.markdown("""
    <div style='background-color:rgba(30,30,50,0.8); padding:20px; border-radius:15px; box-shadow:0 4px 8px rgba(0,0,0,0.5); color:white'>
        <h3>📌 <b>Status Model</b></h3>
        <p>✅ <b>Model berhasil dimuat</b></p>
        <p>📁 <b>Lokasi:</b> <code>model/model_Rini_Laporan 2.h5</code></p>
        <p>🧩 <b>Input model:</b> (None, 128, 128, 3)</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# Halaman 2: Prediksi Gambar
# ================================
elif page == "📷 Prediksi Gambar":
    st.title("📷 Prediksi Gambar Hewan")
    st.markdown("Unggah gambar hewan untuk mendeteksi jenisnya menggunakan model AI.")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)

        model_path = "model/model_Rini_Laporan 2.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.success("✅ Model berhasil dimuat")

            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            st.markdown(f"### 🐶 Prediksi: **Kelas {predicted_class}**")
            st.markdown(f"**Tingkat keyakinan:** {confidence:.2f}%")
        else:
            st.error("❌ Model belum ditemukan di folder 'model/'. Pastikan path benar!")

# ================================
# Halaman 3: Tentang Aplikasi
# ================================
elif page == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Animal Vision AI** adalah aplikasi berbasis *deep learning* yang dirancang untuk
    mengenali jenis hewan dari gambar.  
    Dibangun menggunakan **Streamlit** dan **TensorFlow**, aplikasi ini memiliki fitur:
    - Klasifikasi gambar hewan 🦁  
    - Tampilan antarmuka modern 🌈  
    - Navigasi yang mudah digunakan 💡

    **Dikembangkan oleh:** Rini Safariani 
    """)

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("<p style='text-align:center; color:white;'>© 2025 Rini | Animal Vision AI</p>", unsafe_allow_html=True)
