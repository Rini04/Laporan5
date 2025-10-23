import os
os.system("echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches")  # Fix inotify limit

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import glob

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Animal Vision AI by Rini Safariani", layout="wide")

# ==========================
# STYLING (Elegant Forest Animal Theme)
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #141E30, #243B55);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1508675801603-7a89e9e9c5cf');
            background-size: cover;
            background-attachment: fixed;
            background-blend-mode: overlay;
        }
        .title {
            text-align:center;
            font-size:50px;
            font-weight:900;
            margin-top:10px;
            background: linear-gradient(90deg, #FFCC70, #F3E99F, #FFD966, #F9F871);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: move 3s linear infinite;
        }
        @keyframes move {
            0% {background-position: 0%;}
            100% {background-position: 100%;}
        }
        .subtitle {
            text-align:center;
            color:#E5E5E5;
            font-size:18px;
            margin-bottom:25px;
        }
        .result-box {
            background: rgba(255,255,255,0.15);
            padding:20px;
            border-radius:16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
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
# ANIMAL INFORMATION
# ==========================
animal_info = {
    "spider": {
        "nama": "🕷️ Laba-laba",
        "habitat": "Biasanya ditemukan di taman, rumah, dan pepohonan.",
        "makanan": "Serangga kecil seperti lalat atau nyamuk.",
        "fakta": "Laba-laba membuat jaring sutra yang kuat untuk menangkap mangsanya."
    },
    "cat": {
        "nama": "🐱 Kucing",
        "habitat": "Rumah dan lingkungan manusia.",
        "makanan": "Ikan, daging, dan makanan kucing kering.",
        "fakta": "Kucing dapat tidur hingga 16 jam sehari!"
    },
    "dog": {
        "nama": "🐶 Anjing",
        "habitat": "Rumah atau lingkungan manusia.",
        "makanan": "Daging, tulang, dan makanan anjing kering.",
        "fakta": "Anjing dikenal sangat setia terhadap pemiliknya."
    },
    "chicken": {
        "nama": "🐔 Ayam",
        "habitat": "Kandang dan ladang peternakan.",
        "makanan": "Biji-bijian dan serangga kecil.",
        "fakta": "Ayam dapat mengenali lebih dari 100 wajah manusia!"
    },
    "horse": {
        "nama": "🐴 Kuda",
        "habitat": "Padang rumput dan peternakan.",
        "makanan": "Rumput, jerami, dan gandum.",
        "fakta": "Kuda bisa tidur sambil berdiri."
    },
    "butterfly": {
        "nama": "🦋 Kupu-kupu",
        "habitat": "Kebun, hutan, dan ladang bunga.",
        "makanan": "Nektar dari bunga.",
        "fakta": "Kupu-kupu mencicipi rasa dengan kakinya!"
    },
    "fish": {
        "nama": "🐟 Ikan",
        "habitat": "Air tawar dan laut.",
        "makanan": "Plankton, cacing, dan serangga air.",
        "fakta": "Beberapa ikan bisa tidur dengan mata terbuka!"
    }
}

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🐾 Animal Vision AI by Rini Safariani 🦋</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Sistem Cerdas Klasifikasi Hewan – Elegan, Informatif, dan Menarik</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.header("📂 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Upload Gambar", "Tentang"])

# ==========================
# HALAMAN: UPLOAD GAMBAR
# ==========================
if menu == "Upload Gambar":
    uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])
    
    def preprocess_image(pil_img, size=(224, 224)):
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

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Gambar yang diunggah", width='stretch')
        except Exception as e:
            st.error(f"❌ Gagal membuka gambar: {e}")
            st.stop()

        st.markdown("---")

        if model is None:
            st.error("Model tidak tersedia. Letakkan file .h5 di folder 'model/'.")
        else:
            with st.spinner("🔮 Menganalisis gambar..."):
                try:
                    label, conf = predict_image(model, img)
                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")
                    st.stop()

            if label not in animal_info:
                st.warning(f"Prediksi: {label} (data tidak lengkap). Confidence: {conf:.2%}")
            else:
                info_obj = animal_info[label]
                st.success(f"🌟 Teridentifikasi: {info_obj['nama']} — Confidence: {conf*100:.2f}%")
                st.markdown(f"""
                <div class='result-box'>
                    <h3>{info_obj['nama']}</h3>
                    <b>🌍 Habitat:</b> {info_obj['habitat']}<br>
                    <b>🍽️ Makanan:</b> {info_obj['makanan']}<br>
                    <b>💡 Fakta menarik:</b> {info_obj['fakta']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📁 Unggah gambar untuk memulai klasifikasi. Pastikan file model (.h5) sudah ada di folder 'model/'.")

# ==========================
# HALAMAN: TENTANG
# ==========================
if menu == "Tentang":
    st.markdown("""
    ### 🐾 Tentang Aplikasi
    **Animal Vision AI by Rini Safariani** adalah sistem berbasis AI yang dapat mengenali berbagai jenis hewan seperti kucing, anjing, ikan, ayam, dan lainnya.  
    Didesain dengan tampilan elegan dan informatif, aplikasi ini cocok untuk pembelajaran dan eksplorasi dunia fauna.

    🌟 **Fitur utama:**
    - Klasifikasi citra hewan menggunakan model .h5
    - Informasi habitat, makanan, dan fakta unik setiap hewan
    - Tampilan mewah dan elegan dengan latar alami
    """)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🐾 <b>Animal Vision AI</b> • created by <b>Rini Safariani</b> 💖<br>
    Jelajahi dunia hewan dengan teknologi kecerdasan buatan 🌿
</footer>
""", unsafe_allow_html=True)
