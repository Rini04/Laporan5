# =====================================
# ANIMAL VISION AI — YOLO + KERAS MODEL
# =====================================

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

# ==============================
# CUSTOM STYLE
# ==============================
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

# ==============================
# LOAD MODEL YOLO + KERAS
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Rini_Safariani_Laporan4.pt")  # model YOLO
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan2.h5")  # model klasifikasi
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model YOLO & Keras berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==============================
# CLASS NAMES & ANIMAL INFO
# ==============================
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

# ==============================
# FUNGSI PEMROSESAN GAMBAR
# ==============================
def preprocess_image(pil_img, size=(128, 128)):
    img_resized = pil_img.resize(size)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_class(pil_img):
    arr = preprocess_image(pil_img)
    preds = classifier.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, confidence

def detect_objects(pil_img):
    """Deteksi objek dengan YOLO dan beri bounding box."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        pil_img.save(tmp.name)
        results = yolo_model(tmp.name)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), results

# ==============================
# HALAMAN UTAMA
# ==============================
st.markdown("<div class='title'>🐾 Animal Vision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi & Klasifikasi Hewan menggunakan YOLOv8 dan TensorFlow 🌸</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar asli", width=400)

    with st.spinner("🔍 Mendeteksi objek dan mengenali hewan..."):
        detected_img, yolo_results = detect_objects(img)
        label, conf = predict_class(img)

    st.markdown("### 🧩 Hasil Deteksi Objek")
    st.image(detected_img, caption="📦 Gambar dengan Bounding Box", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 Hasil Klasifikasi")

    if label in animal_info:
        info_obj = animal_info[label]
        st.markdown(f"""
        <div class='result-box'>
            <h3 class='animal-name'>{info_obj['nama']}</h3>
            <b>🌍 Habitat:</b> {info_obj['habitat']}<br>
            <b>🍽️ Makanan:</b> {info_obj['makanan']}<br>
            <b>💡 Fakta menarik:</b> {info_obj['fakta']}<br><br>
            <i>Confidence:</i> <b>{conf*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Hasil klasifikasi: {label} (Confidence: {conf:.2%})")
else:
    st.info("📁 Silakan unggah gambar untuk mulai analisis hewan.")

# ==============================
# FOOTER
# ==============================
st.markdown("""
<footer>
    🌷 <b>Animal Vision AI</b> — by Rini Safariani<br>
    Model: YOLO (.pt) & TensorFlow (.h5) berada di folder <code>model/</code>
</footer>
""", unsafe_allow_html=True)
