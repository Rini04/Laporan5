import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Konfigurasi awal halaman
st.set_page_config(page_title="Animal Vision AI", layout="wide", page_icon="🐾")

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

    # Box status model
    st.markdown("""
    <div style='background-color:#1e1e2f; padding:20px; border-radius:15px; box-shadow:0 4px 8px rgba(0,0,0,0.3); color:white'>
        <h3 style='margin-bottom:10px;'>📌 <b>Status Model</b></h3>
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

        # Simulasi load model
        model_path = "model/model_Rini_Laporan 2.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.success("Model berhasil dimuat ✅")

            # Ubah gambar jadi numpy array dan prediksi
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            st.markdown(f"### 🐶 Prediksi: **Kelas {predicted_class}**")
            st.markdown(f"**Tingkat keyakinan:** {confidence:.2f}%")
        else:
            st.error("Model belum ditemukan di folder 'model/'. Pastikan path benar!")

# ================================
# Halaman 3: Tentang Aplikasi
# ================================
elif page == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Animal Vision AI** adalah aplikasi berbasis *deep learning* yang dirancang untuk
    mengidentifikasi jenis hewan dari gambar.  
    Dibangun menggunakan **Streamlit** dan **TensorFlow**, aplikasi ini dapat:
    - Melakukan klasifikasi gambar hewan 🐾  
    - Menampilkan status dan konfigurasi model AI 🔍  
    - Memberikan antarmuka interaktif dan mudah digunakan 💡

    **Dikembangkan oleh:** Rini Safariani 
    """)

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 Rini | Animal Vision AI</p>", unsafe_allow_html=True)
