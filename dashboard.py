import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
from PIL import Image
import base64

# ====================== FUNGSI UNTUK BACKGROUND GAMBAR ======================
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1574158622682-e40e69881006");
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# ====================== SIDEBAR NAVIGASI ======================
st.sidebar.markdown("## 🧭 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📤 Upload Gambar", "🔮 Prediksi", "ℹ️ Tentang Model"])

# ====================== HALAMAN HOME ======================
if menu == "🏠 Home":
    st.markdown(
        """
        <style>
        .title {
            font-size: 48px;
            color: #FFD700;
            text-align: center;
            text-shadow: 2px 2px 10px #000000;
            font-weight: 800;
            animation: pulse 3s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); color: #FFD700; }
            50% { transform: scale(1.08); color: #FFA500; }
            100% { transform: scale(1); color: #FFD700; }
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #fff;
            font-weight: 500;
        }
        .footer {
            text-align: center;
            margin-top: 80px;
            font-size: 16px;
            color: #f2f2f2;
        }
        </style>

        <div class="title">🐾 Dashboard Prediksi Hewan AI 🐾</div>
        <p class="subtitle">Selamat datang di dashboard prediksi hewan AI karya <b>Rini Safariani</b> ✨<br>
        Temukan kecerdasan buatan dalam mengenali berbagai jenis hewan dengan tampilan elegan dan interaktif.</p>

        <div class="footer">
        🐾 <i>Animal Vision AI Dashboard by Rini Safariani 🌷</i><br>
        © 2025 All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# ====================== HALAMAN UPLOAD GAMBAR ======================
elif menu == "📤 Upload Gambar":
    st.header("📤 Upload Gambar Hewan")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diupload', use_container_width=True)
        st.success("✅ Gambar berhasil diunggah!")

# ====================== HALAMAN PREDIKSI ======================
elif menu == "🔮 Prediksi":
    st.header("🔮 Hasil Prediksi Hewan")
    st.info("Fitur prediksi akan menampilkan jenis hewan berdasarkan gambar yang diunggah.")

# ====================== HALAMAN TENTANG MODEL ======================
elif menu == "ℹ️ Tentang Model":
    st.header("ℹ️ Tentang Model AI")
    st.success("Model berhasil dimuat: `model/model_Rini_Laporan2.h5`")
    st.write("Input model: `(None, 128, 128, 3)`")
    st.markdown(
        """
        **Penjelasan Singkat:**  
        Model ini dilatih menggunakan dataset berbagai jenis hewan untuk mendeteksi dan mengenali jenis hewan secara otomatis.
        Dashboard ini dibuat dengan teknologi *Streamlit* dan *TensorFlow* untuk mempermudah interaksi pengguna.
        """
    )

