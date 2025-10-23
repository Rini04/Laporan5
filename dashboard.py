import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os, glob

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Dashboard Prediksi Hewan AI", layout="wide")

# ==========================
# CSS THEME (ELEGANT ANIMAL STYLE)
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #141E30, #243B55);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1517849845537-4d257902454a');
            background-size: cover;
            background-attachment: fixed;
            background-blend-mode: overlay;
        }
        .title {
            text-align:center;
            color:#FFD369;
            font-size:46px;
            font-weight:900;
            margin-top:10px;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.7);
        }
        .subtitle {
            text-align:center;
            color:#EEEEEE;
            font-size:18px;
            margin-bottom:30px;
        }
        .navbox {
            background: rgba(255,255,255,0.08);
            padding:15px;
            border-radius:15px;
            margin-top:10px;
        }
        footer {
            text-align:center;
            color:#ccc;
            margin-top:40px;
            padding:10px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# MODEL LOADING
# ==========================
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

# ==========================
# CLASS LABELS
# ==========================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📤 Upload Gambar", "🔮 Prediksi", "ℹ️ Tentang Model"])

# ==========================
# HALAMAN HOME
# ==========================
if page == "🏠 Home":
    st.markdown("<div class='title'>🐾 Dashboard Prediksi Hewan AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Selamat datang di dashboard prediksi hewan AI karya <b>Rini Safariani</b> 💫<br>Temukan kecerdasan buatan dalam mengenali berbagai jenis hewan dengan tampilan elegan dan interaktif.</div>", unsafe_allow_html=True)

# ==========================
# HALAMAN UPLOAD GAMBAR
# ==========================
elif page == "📤 Upload Gambar":
    st.markdown("<div class='title'>📸 Upload Gambar Hewan</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah gambar hewan (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.session_state["uploaded_img"] = img
        st.success("✅ Gambar berhasil diunggah! Silakan menuju menu 'Prediksi' untuk analisis.")

# ==========================
# HALAMAN PREDIKSI
# ==========================
elif page == "🔮 Prediksi":
    st.markdown("<div class='title'>🔮 Prediksi Hewan</div>", unsafe_allow_html=True)

    if "uploaded_img" not in st.session_state:
        st.warning("⚠️ Silakan unggah gambar terlebih dahulu melalui menu 'Upload Gambar'.")
    else:
        img = st.session_state["uploaded_img"]
        st.image(img, caption="Gambar yang akan diprediksi", use_container_width=True)

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

        if model is None:
            st.error("❌ Model belum tersedia di folder 'model/'.")
        else:
            with st.spinner("⏳ Menganalisis gambar..."):
                try:
                    label, conf = predict_image(model, img)
                    st.success(f"🌟 Hasil Prediksi: {label} ({conf*100:.2f}% akurasi)")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==========================
# HALAMAN TENTANG MODEL
# ==========================
elif page == "ℹ️ Tentang Model":
    st.markdown("<div class='title'>ℹ️ Tentang Model</div>", unsafe_allow_html=True)
    if model is None:
        st.warning("Model belum dimuat.")
    else:
        st.success(f"✅ Model berhasil dimuat dari: {info}")
        st.markdown("📐 **Arsitektur Model AI**:")
        st.json({"Input Shape": "(None, 128, 128, 3)", "Output Classes": class_names})

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🐾 <b>Animal Vision AI Dashboard</b> by <b>Rini Safariani</b> 🌷<br>
    © 2025 All Rights Reserved.
</footer>
""", unsafe_allow_html=True)
