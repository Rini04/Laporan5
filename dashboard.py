import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🧠 Image Classification & Object Detection", layout="wide")

# Styling CSS Custom
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            color: #2b2d42;
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #6c757d;
            margin-bottom: 30px;
        }
        .stImage {
            border-radius: 10px;
        }
        .result-box {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        footer {
            text-align: center;
            color: #adb5bd;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Rini Safariani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🧠 Image Classification & Object Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Aplikasi ini menggunakan <b>YOLOv8</b> untuk deteksi objek dan <b>TensorFlow</b> untuk klasifikasi gambar.</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES FILE YANG DI-UPLOAD
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        # ==========================
        # MODE DETEKSI OBJEK
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            with st.spinner("Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="🧩 Hasil Deteksi Objek", use_container_width=True)

        # ==========================
        # MODE KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("🧾 Hasil Klasifikasi Gambar")
            with st.spinner("Sedang mengklasifikasi gambar..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

            st.success(f"🎯 **Prediksi:** {class_index}")
            st.info(f"📊 **Probabilitas:** {confidence:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu melalui sidebar untuk memulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
    <footer>
        👩‍💻 Dibuat oleh <b>Rini Safariani</b> — Menggabungkan <b>YOLOv8</b> & <b>TensorFlow</b> untuk Analisis Gambar.<br>
        <span style="font-size:13px;">© 2025 — All rights reserved.</span>
    </footer>
""", unsafe_allow_html=True)
