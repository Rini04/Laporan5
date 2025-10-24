import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
from PIL import Image
import glob
import os
import cv2

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="🐾 Animal Vision AI - YOLO + TensorFlow", layout="wide")

# ================================
# CSS STYLE — Tema Cantik Elegan
# ================================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #1b2735, #090a0f);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1500&q=80');
            background-size: cover;
            background-attachment: fixed;
            background-blend-mode: overlay;
        }
        .title {
            text-align:center;
            color:#FFD700;
            font-size:45px;
            font-weight:900;
            margin-top:10px;
            text-shadow: 2px 2px 10px rgba(255,215,0,0.7);
        }
        .subtitle {
            text-align:center;
            color:#f0f0f0;
            font-size:18px;
            margin-bottom:25px;
        }
        .model-box, .result-box {
            background: rgba(255,255,255,0.12);
            padding:20px;
            border-radius:16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }
        .animal-name {
            color: #FFD700;
            font-weight: 700;
            font-size: 24px;
            text-shadow: 0 0 10px rgba(255,215,0,0.6);
        }
        footer {
            text-align:center;
            color:#ccc;
            margin-top:35px;
            padding:10px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODEL YOLO + TENSORFLOW
# ================================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Rini Safariani_Laporan 4.pt")  # YOLO hasil training kamu
    except:
        yolo_model = YOLO("yolov8n.pt")  # fallback ke YOLO bawaan
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model YOLO dan TensorFlow berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ================================
# CLASS NAMES & INFO
# ================================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]

animal_info = {
    "spider": {"nama":"🕷️ Laba-laba","habitat":"Taman dan pepohonan","makanan":"Serangga kecil","fakta":"Jaring sutranya lebih kuat dari baja."},
    "cat": {"nama":"🐱 Kucing","habitat":"Rumah & kota","makanan":"Ikan & daging","fakta":"Kucing tidur hingga 16 jam per hari."},
    "dog": {"nama":"🐶 Anjing","habitat":"Rumah & taman","makanan":"Daging & tulang","fakta":"Anjing sangat setia pada pemiliknya."},
    "chicken": {"nama":"🐔 Ayam","habitat":"Kandang & kebun","makanan":"Biji-bijian","fakta":"Ayam bisa mengenali wajah manusia."},
    "horse": {"nama":"🐴 Kuda","habitat":"Padang & peternakan","makanan":"Rumput & gandum","fakta":"Kuda bisa tidur sambil berdiri."},
    "butterfly": {"nama":"🦋 Kupu-kupu","habitat":"Kebun bunga","makanan":"Nektar bunga","fakta":"Kupu-kupu mencicipi dengan kakinya."},
    "fish": {"nama":"🐟 Ikan","habitat":"Air tawar & laut","makanan":"Plankton & serangga air","fakta":"Beberapa ikan tidur dengan mata terbuka."}
}

# ================================
# NAVIGATION
# ================================
st.sidebar.title("🐾 Navigasi")
page = st.sidebar.radio(
    "Pilih Mode:",
    ["🧠 Model Info", "🖼️ Deteksi & Klasifikasi", "ℹ️ Tentang Aplikasi"]
)

# ================================
# PAGE 1 — MODEL INFO
# ================================
if page == "🧠 Model Info":
    st.markdown("<div class='title'>📦 Status Model</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-box'>
        <h4>✅ Model YOLOv8 dan TensorFlow berhasil dimuat</h4>
        <p><b>📁 YOLO:</b> Deteksi objek dengan bounding box</p>
        <p><b>📁 TensorFlow:</b> Klasifikasi jenis hewan</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# PAGE 2 — DETEKSI + KLASIFIKASI
# ================================
elif page == "🖼️ Deteksi & Klasifikasi":
    st.markdown("<div class='title'>🐾 Deteksi dan Klasifikasi Hewan</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Gabungan YOLOv8 + TensorFlow 🌸</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah gambar (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar Asli", width=400)

        st.markdown("---")

        # 🔹 YOLO Object Detection
        st.subheader("🔍 Hasil Deteksi Objek (YOLOv8)")
        with st.spinner("Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)

        # 🔹 Klasifikasi TensorFlow
        st.subheader("🧠 Hasil Klasifikasi Gambar")
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("Menganalisis gambar..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label = class_names[class_index]

        info_obj = animal_info[label]
        st.markdown(f"""
        <div class='result-box'>
            <h3 class='animal-name'>{info_obj['nama']}</h3>
            <b>🌍 Habitat:</b> {info_obj['habitat']}<br>
            <b>🍽️ Makanan:</b> {info_obj['makanan']}<br>
            <b>💡 Fakta menarik:</b> {info_obj['fakta']}<br><br>
            <i>Confidence:</i> <b>{confidence*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📁 Silakan unggah gambar untuk mulai mendeteksi dan mengklasifikasi.")

# ================================
# PAGE 3 — ABOUT
# ================================
elif page == "ℹ️ Tentang Aplikasi":
    st.markdown("<div class='title'>🌷 Tentang Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-box'>
        <p><b>Animal Vision AI</b> adalah aplikasi gabungan antara <b>YOLOv8</b> (untuk deteksi objek)
        dan <b>TensorFlow</b> (untuk klasifikasi gambar hewan).</p>
        <ul>
            <li>🔹 Deteksi objek otomatis dengan bounding box</li>
            <li>🔹 Klasifikasi hewan berdasarkan citra</li>
            <li>🎨 Antarmuka elegan dan interaktif</li>
        </ul>
        <p>💛 Dikembangkan oleh <b>Rini Safariani</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<footer>
    🌷 <b>Animal Vision AI</b> — by Rini<br>
    YOLOv8 + TensorFlow in one elegant dashboard ✨
</footer>
""", unsafe_allow_html=True)
