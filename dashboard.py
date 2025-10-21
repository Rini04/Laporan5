import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import glob
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

# ==========================
# CSS THEME (ELEGANT ANIMAL STYLE)
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #141E30, #243B55);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1546182990-dffeafbe841d');
            background-size: cover;
            background-attachment: fixed;
            background-blend-mode: overlay;
        }
        .title {
            text-align:center;
            color:#FFCC70;
            font-size:45px;
            font-weight:900;
            margin-top:10px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }
        .subtitle {
            text-align:center;
            color:#E5E5E5;
            font-size:18px;
            margin-bottom:25px;
        }
        .result-box {
            background: rgba(255,255,255,0.12);
            padding:20px;
            border-radius:16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.3);
            backdrop-filter: blur(8px);
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
# INFO DATA HEWAN
# ==========================
animal_info = {
    "spider": {"nama": "🕷️ Laba-laba", "habitat": "Taman, rumah, pepohonan.",
               "makanan": "Serangga kecil seperti lalat atau nyamuk.",
               "fakta": "Laba-laba membuat jaring sutra yang kuat untuk menangkap mangsanya."},
    "cat": {"nama": "🐱 Kucing", "habitat": "Lingkungan rumah manusia.",
            "makanan": "Ikan, daging, makanan kucing kering.",
            "fakta": "Kucing dapat tidur hingga 16 jam sehari!"},
    "dog": {"nama": "🐶 Anjing", "habitat": "Lingkungan rumah manusia.",
            "makanan": "Daging, tulang, makanan anjing kering.",
            "fakta": "Anjing dikenal sangat setia terhadap pemiliknya."},
    "chicken": {"nama": "🐔 Ayam", "habitat": "Kandang dan ladang peternakan.",
                "makanan": "Biji-bijian dan serangga kecil.",
                "fakta": "Ayam dapat mengenali lebih dari 100 wajah manusia!"},
    "horse": {"nama": "🐴 Kuda", "habitat": "Padang rumput dan peternakan.",
              "makanan": "Rumput, jerami, gandum.",
              "fakta": "Kuda bisa tidur sambil berdiri."},
    "butterfly": {"nama": "🦋 Kupu-kupu", "habitat": "Kebun, hutan, ladang bunga.",
                  "makanan": "Nektar bunga.",
                  "fakta": "Kupu-kupu mencicipi rasa dengan kakinya!"},
    "fish": {"nama": "🐟 Ikan", "habitat": "Air tawar dan laut.",
             "makanan": "Plankton, cacing, serangga air.",
             "fakta": "Beberapa ikan bisa tidur dengan mata terbuka!"}
}

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🐾 Animal Vision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi Citra Hewan — Model Cerdas dan Elegan</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR STATUS
# ==========================
st.sidebar.header("📦 Status Model")
if model is None:
    if info == "no_model":
        st.sidebar.error("❌ Tidak ditemukan file .h5 di folder 'model/'.")
    else:
        st.sidebar.error(f"❌ Gagal memuat model: {info}")
else:
    st.sidebar.success(f"✅ Model berhasil dimuat dari:\n{info}")
    st.sidebar.write(f"📏 Input model: {model.input_shape}")

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

# ==========================
# FIX PREPROCESSING (auto menyesuaikan ukuran input model)
# ==========================
def preprocess_image(pil_img, model):
    input_shape = model.input_shape[1:3] if model and model.input_shape else (224, 224)
    img_resized = pil_img.resize(input_shape)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img, model)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, confidence

# ==========================
# MAIN DISPLAY
# ==========================
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
# FOOTER
# ==========================
st.markdown("""
<footer>
    🐾 <b>Animal Vision AI</b> • by Repa Cantikk 🌷<br>
    Letakkan model klasifikasi hewan kamu di folder <code>model/</code> (format .h5)
</footer>
""", unsafe_allow_html=True)
