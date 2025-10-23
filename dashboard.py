import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import glob
import os
import time

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

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
        .model-box {
            background: rgba(255,255,255,0.12);
            padding:20px;
            border-radius:16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }
        .result-box {
            background: rgba(255,255,255,0.15);
            padding:25px;
            border-radius:20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            backdrop-filter: blur(10px);
        }
        .animal-name {
            color: #FFD700;
            font-weight: 700;
            font-size: 24px;
            text-shadow: 0 0 10px rgba(255,215,0,0.6);
        }
        .loader {
            text-align:center;
            margin-top:20px;
        }
        .loader img {
            width:150px;
            animation: spin 3s linear infinite;
        }
        @keyframes spin {
            0% {transform: rotate(0deg);}
            100% {transform: rotate(360deg);}
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
# MODEL LOADING
# ================================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_model():
    h5_path = find_first("*.h5")
    if not h5_path:
        return None, "no_model"
    try:
        model = tf.keras.models.load_model(h5_path)
        return model, h5_path
    except Exception as e:
        return None, f"error:{e}"

model, info = load_model()

# ================================
# CLASS NAMES
# ================================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]

# ================================
# ANIMAL INFO
# ================================
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
# HEADER
# ================================
st.markdown("<div class='title'>🐾 Animal Vision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi Gambar Hewan dengan Model Cerdas dan Tampilan Cantik 🌸</div>", unsafe_allow_html=True)

# ================================
# SIDEBAR - STATUS MODEL
# ================================
st.sidebar.header("📦 Status Model")

if model is None:
    if info == "no_model":
        st.sidebar.error("❌ Tidak ditemukan file .h5 di folder 'model/'.")
    else:
        st.sidebar.error(f"❌ Gagal memuat model: {info}")
else:
    st.sidebar.markdown(f"""
    <div class='model-box'>
        <h4>✅ Model berhasil dimuat</h4>
        <p><b>📁 Lokasi:</b><br>{info}</p>
        <p><b>🔢 Input model:</b><br>(None, 128, 128, 3)</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# UPLOAD GAMBAR
# ================================
uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

def preprocess_image(pil_img, size=(128, 128)):
    img_resized = pil_img.resize(size)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, confidence

# ================================
# MAIN DISPLAY
# ================================
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Gambar yang diunggah", width=400)
    except Exception as e:
        st.error(f"❌ Gagal membuka gambar: {e}")
        st.stop()

    st.markdown("---")

    if model is None:
        st.error("Model tidak tersedia. Letakkan file .h5 di folder 'model/'.")
    else:
        # Animasi lucu 🐱✨
        with st.spinner("🔮 Menganalisis gambar..."):
            placeholder = st.empty()
            with placeholder.container():
                st.markdown("""
                <div class="loader">
                    <img src="https://media.tenor.com/JL1JQJt_1fEAAAAi/cat-cute.gif">
                    <p>✨ Kucing imut lagi mikir... tunggu sebentar ya! ✨</p>
                </div>
                """, unsafe_allow_html=True)
            time.sleep(3)  # simulasi proses
            placeholder.empty()

            try:
                label, conf = predict_image(model, img)
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.stop()

        if label not in animal_info:
            st.warning(f"Prediksi: {label} (data tidak lengkap). Confidence: {conf:.2%}")
        else:
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
    st.info("📁 Unggah gambar untuk mulai klasifikasi. Pastikan file model (.h5) sudah ada di folder 'model/'.")

# ================================
# FOOTER
# ================================
st.markdown("""
<footer>
    🌷 <b>Animal Vision AI</b> — by Repa Cantikk 💕<br>
    Letakkan file model di folder <code>model/</code> (format .h5)
</footer>
""", unsafe_allow_html=True)
