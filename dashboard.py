# =====================================
# FIX ENVIRONMENT & AUTO CLEAR CACHE
# =====================================
import os
os.environ["PYTHONWATCHDOG_DISABLE"] = "1"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
st.cache_data.clear()          # 🧹 Bersihkan cache data otomatis
st.cache_resource.clear()      # 🧹 Bersihkan cache model otomatis

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from PIL import Image
import numpy as np
import glob

# =====================================
# PAGE CONFIG & THEME
# =====================================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

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
            text-shadow:2px 2px 10px rgba(255,215,0,0.7);
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
            font-weight:700;
            font-size:24px;
            text-shadow:0 0 10px rgba(255,215,0,0.6);
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

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource(show_spinner=False)
def load_models():
    h5_files = glob.glob("model/*.h5")
    pt_files = glob.glob("model/*.pt")

    classifier = tf.keras.models.load_model(h5_files[0]) if h5_files else None
    yolo_model = YOLO(pt_files[0]) if pt_files else None

    return classifier, yolo_model

classifier, yolo_model = load_models()

# =====================================
# ANIMAL INFO
# =====================================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]

animal_info = {
    "spider": {"nama":"🕷️ Laba-laba","habitat":"Taman & pepohonan","makanan":"Serangga kecil","fakta":"Jaring sutranya lebih kuat dari baja."},
    "cat": {"nama":"🐱 Kucing","habitat":"Rumah & kota","makanan":"Ikan & daging","fakta":"Kucing tidur hingga 16 jam per hari."},
    "dog": {"nama":"🐶 Anjing","habitat":"Rumah & taman","makanan":"Daging & tulang","fakta":"Anjing sangat setia pada pemiliknya."},
    "chicken": {"nama":"🐔 Ayam","habitat":"Kandang & kebun","makanan":"Biji-bijian","fakta":"Ayam bisa mengenali wajah manusia."},
    "horse": {"nama":"🐴 Kuda","habitat":"Padang & peternakan","makanan":"Rumput & gandum","fakta":"Kuda bisa tidur sambil berdiri."},
    "butterfly": {"nama":"🦋 Kupu-kupu","habitat":"Kebun bunga","makanan":"Nektar bunga","fakta":"Kupu-kupu mencicipi dengan kakinya."},
    "fish": {"nama":"🐟 Ikan","habitat":"Air tawar & laut","makanan":"Plankton & serangga air","fakta":"Beberapa ikan tidur dengan mata terbuka."}
}

# =====================================
# SIDEBAR NAVIGATION
# =====================================
st.sidebar.title("🐾 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🧠 Model Info", "🖼️ Prediksi Hewan", "🎯 Deteksi Objek (YOLO)", "ℹ️ Tentang Aplikasi"]
)

# =====================================
# PAGE 1 — MODEL INFO
# =====================================
if page == "🧠 Model Info":
    st.markdown("<div class='title'>📦 Status Model</div>", unsafe_allow_html=True)
    if classifier:
        st.success("✅ Model Klasifikasi dimuat.")
    else:
        st.error("❌ Model .h5 tidak ditemukan.")
    if yolo_model:
        st.success("✅ Model YOLO dimuat.")
    else:
        st.error("❌ Model .pt tidak ditemukan.")

# =====================================
# PAGE 2 — PREDIKSI HEWAN (Klasifikasi)
# =====================================
elif page == "🖼️ Prediksi Hewan":
    st.markdown("<div class='title'>🐾 Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Klasifikasi Gambar Hewan dengan Model Cerdas dan Tampilan Cantik 🌸</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Unggah gambar hewan:", type=["jpg","jpeg","png"])

    if uploaded and classifier:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📸 Gambar diunggah", width=400)
        st.markdown("---")
        arr = image.img_to_array(img.resize((128,128)))/255.0
        arr = np.expand_dims(arr, axis=0)
        preds = classifier.predict(arr)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = class_names[idx]
        info = animal_info[label]
        st.markdown(f"""
        <div class='result-box'>
            <h3 class='animal-name'>{info['nama']}</h3>
            🌍 Habitat: {info['habitat']}<br>
            🍽️ Makanan: {info['makanan']}<br>
            💡 Fakta: {info['fakta']}<br><br>
            Confidence: <b>{conf*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)

# =====================================
# PAGE 3 — DETEKSI OBJEK YOLO
# =====================================
elif page == "🎯 Deteksi Objek (YOLO)":
    st.markdown("<div class='title'>🎯 Deteksi Objek dengan YOLOv8</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📸 Unggah gambar untuk deteksi:", type=["jpg","jpeg","png"])

    if uploaded and yolo_model:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="🖼️ Gambar Asli", width=500)
        st.markdown("---")
        with st.spinner("🔍 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="📦 Hasil Deteksi Objek", use_container_width=True)

# =====================================
# PAGE 4 — ABOUT
# =====================================
elif page == "ℹ️ Tentang Aplikasi":
    st.markdown("<div class='title'>🌷 Tentang Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-box'>
        <p><b>Animal Vision AI</b> adalah aplikasi AI yang dapat:</p>
        <ul>
            <li>🔹 Mendeteksi objek dengan <b>YOLOv8</b></li>
            <li>🔹 Mengklasifikasi hewan menggunakan <b>TensorFlow</b></li>
            <li>🎨 Didesain dengan tampilan elegan dan interaktif</li>
        </ul>
        <p>💛 Dibuat oleh <b>Rini</b> untuk eksplorasi AI dan desain aplikasi modern.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<footer>
🌷 <b>Animal Vision AI</b> — by Rini<br>
Letakkan file model di folder <code>model/</code> (format .h5 dan .pt)
</footer>
""", unsafe_allow_html=True)
