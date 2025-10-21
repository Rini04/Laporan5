import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🐾 Animal Vision Pro", layout="wide")

# ==========================
# CUSTOM CSS — Tema Pastel Floral Cantik
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fde2e4 0%, #fad2e1 30%, #e2ece9 70%, #bee1e6 100%);
            background-attachment: fixed;
            color: #2c2c2c;
            font-family: 'Poppins', sans-serif;
        }
        .title {
            text-align: center;
            color: #5a189a;
            font-size: 45px;
            font-weight: 800;
            margin-top: 10px;
            text-shadow: 0 2px 6px rgba(90, 24, 154, 0.2);
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #6d597a;
            margin-bottom: 30px;
        }
        .result-box {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(8px);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #f4acb7;
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #d48cb4;
            transform: scale(1.03);
        }
        footer {
            text-align: center;
            color: #5c4d7d;
            font-size: 15px;
            margin-top: 40px;
            background: rgba(255, 255, 255, 0.5);
            padding: 10px;
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/model_Rini_Laporan_2.h5")
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# DATA INFORMASI HEWAN
# ==========================
animal_info = {
    "cat": {
        "nama": "Kucing 🐱",
        "jenis": "Domestik, Persia, Maine Coon, Siam, Bengal",
        "makanan": "Daging, ikan, makanan kucing kering",
        "habitat": "Rumah, taman, lingkungan manusia",
        "fakta": "Kucing bisa tidur 12–16 jam per hari dan punya refleks melompat luar biasa."
    },
    "dog": {
        "nama": "Anjing 🐶",
        "jenis": "Labrador, Bulldog, German Shepherd, Pomeranian",
        "makanan": "Daging, tulang, makanan anjing komersial",
        "habitat": "Rumah dan lingkungan manusia",
        "fakta": "Hidung anjing bisa mencium 100.000 kali lebih tajam dari manusia!"
    },
    "fish": {
        "nama": "Ikan 🐟",
        "jenis": "Mas, Guppy, Koi, Salmon, Lele",
        "makanan": "Plankton, serangga air, pelet ikan",
        "habitat": "Sungai, laut, danau, akuarium",
        "fakta": "Beberapa ikan seperti hiu tidak punya tulang, hanya tulang rawan."
    },
    "chicken": {
        "nama": "Ayam 🐔",
        "jenis": "Kampung, Broiler, Petelur, Silkie",
        "makanan": "Biji-bijian, serangga kecil, dedak",
        "habitat": "Kandang dan area peternakan",
        "fakta": "Ayam bisa mengenali lebih dari 100 wajah manusia dan hewan."
    },
    "horse": {
        "nama": "Kuda 🐴",
        "jenis": "Arab, Poni, Thoroughbred",
        "makanan": "Rumput, jerami, biji-bijian",
        "habitat": "Padang rumput, peternakan, istal",
        "fakta": "Kuda bisa tidur sambil berdiri karena mekanisme pengunci kaki."
    },
    "butterfly": {
        "nama": "Kupu-kupu 🦋",
        "jenis": "Monarch, Swallowtail, Morpho, Sulphur",
        "makanan": "Nektar bunga, sari tumbuhan",
        "habitat": "Taman, padang rumput, hutan",
        "fakta": "Kupu-kupu merasakan rasa lewat kaki mereka!"
    },
    "spider": {
        "nama": "Laba-laba 🕷️",
        "jenis": "Tarantula, Black Widow, Jumping Spider",
        "makanan": "Serangga kecil",
        "habitat": "Sudut rumah, taman, hutan, gua",
        "fakta": "Jaring laba-laba lima kali lebih kuat dari baja dengan berat yang sama."
    }
}

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🐾 Animal Vision Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi Hewan + Info Lengkap by Repa Cantikk 💖</div>", unsafe_allow_html=True)

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📸 Unggah gambar hewan di bawah ini:", type=["jpg", "jpeg", "png"])

# ==========================
# PREDIKSI & HASIL
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)

    with st.spinner("🔍 Sedang menganalisis gambar..."):
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        pred = model.predict(img_array)
        class_names = ["butterfly", "cat", "chicken", "dog", "fish", "horse", "spider"]
        class_idx = np.argmax(pred)
        class_label = class_names[class_idx]
        confidence = np.max(pred)

    # ==========================
    # TAMPILKAN HASIL
    # ==========================
    st.success(f"🎯 Hewan Terdeteksi: {animal_info[class_label]['nama']} ({confidence*100:.2f}%)")

    info = animal_info[class_label]
    st.markdown(f"""
    <div class='result-box'>
        <h3>{info['nama']}</h3>
        <b>🌿 Jenis-jenis:</b> {info['jenis']}<br>
        <b>🍽️ Makanan:</b> {info['makanan']}<br>
        <b>🏞️ Habitat:</b> {info['habitat']}<br>
        <b>💡 Fakta menarik:</b> {info['fakta']}
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("📁 Silakan unggah gambar hewan terlebih dahulu untuk mulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🌸 <b>Animal Vision Pro</b> by <b>Repa Cantikk</b><br>
    TensorFlow • Image Recognition • Cute & Smart Interface<br>
    <span style="font-size:13px;">© 2025 All rights reserved.</span>
</footer>
""", unsafe_allow_html=True)
