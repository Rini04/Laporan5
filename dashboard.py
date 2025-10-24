import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
from PIL import Image
import glob
import os
import tempfile

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

# ================================
# CSS STYLE
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
# LOAD MODELS
# ================================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_models():
    yolo_path = find_first("*.pt")
    h5_path = find_first("*.h5")

    yolo_model = YOLO(yolo_path) if yolo_path else None
    classifier = tf.keras.models.load_model(h5_path) if h5_path else None
    return yolo_model, classifier, yolo_path, h5_path

yolo_model, classifier, yolo_path, h5_path = load_models()

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
# SIDEBAR
# ================================
st.sidebar.title("🐾 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🧠 Model Info", "🖼️ Prediksi Hewan", "📦 Deteksi Objek (YOLO)", "ℹ️ Tentang Aplikasi"]
)

# ================================
# PAGE 1 — MODEL INFO
# ================================
if page == "🧠 Model Info":
    st.markdown("<div class='title'>📦 Status Model</div>", unsafe_allow_html=True)
    if not yolo_model and not classifier:
        st.error("❌ Tidak ditemukan model (.pt / .h5) di folder 'model/'.")
    else:
        st.markdown(f"""
        <div class='model-box'>
            <h4>✅ Model berhasil dimuat</h4>
            <p><b>📁 YOLO:</b> {yolo_path or 'Tidak ditemukan'}</p>
            <p><b>📁 Classifier:</b> {h5_path or 'Tidak ditemukan'}</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# PAGE 2 — KLASIFIKASI
# ================================
elif page == "🖼️ Prediksi Hewan":
    st.markdown("<div class='title'>🐾 Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Klasifikasi Gambar Hewan menggunakan Deep Learning 🌸</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar yang diunggah", width=400)
        st.markdown("---")

        if classifier:
            img_resized = img.resize((128, 128))
            arr = np.expand_dims(image.img_to_array(img_resized)/255.0, axis=0)
            preds = classifier.predict(arr)
            idx = int(np.argmax(preds))
            label = class_names[idx] if idx < len(class_names) else "unknown"
            conf = float(np.max(preds))

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
                st.warning(f"Prediksi: {label} (confidence {conf:.2%})")
        else:
            st.error("Model klasifikasi tidak ditemukan.")
    else:
        st.info("📁 Unggah gambar untuk mulai klasifikasi.")

# ================================
# PAGE 3 — DETEKSI OBJEK YOLO
# ================================
elif page == "📦 Deteksi Objek (YOLO)":
    st.markdown("<div class='title'>📦 Deteksi Objek YOLO</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Unggah gambar (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar Asli", width=400)
        st.markdown("---")

        if yolo_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                img.save(temp.name)
                results = yolo_model(temp.name)
                result_img = results[0].plot()
            st.image(result_img, caption="🧩 Hasil Deteksi (dengan kotak)", use_container_width=True)
        else:
            st.error("Model YOLO tidak ditemukan.")
    else:
        st.info("📁 Unggah gambar untuk mendeteksi objek.")

# ================================
# PAGE 4 — ABOUT
# ================================
elif page == "ℹ️ Tentang Aplikasi":
    st.markdown("<div class='title'>🌷 Tentang Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-box'>
    <p><b>Animal Vision AI</b> adalah aplikasi AI yang mengenali dan mendeteksi hewan melalui gambar,
    menggabungkan <b>TensorFlow</b> untuk klasifikasi dan <b>YOLOv8</b> untuk deteksi objek.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<footer>
    🌷 <b>Animal Vision AI</b> — by Rini<br>
    Letakkan file model di folder <code>model/</code> (format .h5 & .pt)
</footer>
""", unsafe_allow_html=True)
