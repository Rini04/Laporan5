import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os, glob

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🐾 Animal Vision Pro Premium", layout="wide")

# ==========================
# STYLING PREMIUM
# ==========================
bg_url = "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0"  # contoh: hewan alami background
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            font-family: 'Poppins', sans-serif;
            color: #2d2d2d;
        }}
        .main-title {{
            text-align: center;
            color: white;
            font-size: 48px;
            font-weight: 700;
            text-shadow: 0px 0px 20px rgba(0,0,0,0.6);
            margin-top: 20px;
        }}
        .subtitle {{
            text-align: center;
            color: #f0f0f0;
            font-size: 18px;
            margin-bottom: 40px;
            text-shadow: 0px 0px 10px rgba(0,0,0,0.5);
        }}
        .glass {{
            background: rgba(255, 255, 255, 0.82);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}
        .footer {{
            text-align:center;
            color:#fff;
            margin-top:30px;
            text-shadow:0 0 8px rgba(0,0,0,0.4);
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_classifier():
    model_path = "model/model_Rini_Laporan 2.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

classifier = load_classifier()
class_names = ["butterfly", "cat", "chicken", "dog", "fish", "horse", "spider"]

# ==========================
# DATABASE INFORMASI HEWAN
# ==========================
animal_info = {
    "cat": {"nama": "Kucing 🐱", "jenis": "Domestik, Persia, Siam", "makanan": "Daging, ikan", "habitat": "Rumah", "fakta": "Bisa tidur 12-16 jam per hari."},
    "dog": {"nama": "Anjing 🐶", "jenis": "Labrador, Pomeranian", "makanan": "Daging, tulang", "habitat": "Rumah", "fakta": "Hidungnya sangat tajam."},
    "fish": {"nama": "Ikan 🐟", "jenis": "Koi, Mas, Guppy", "makanan": "Plankton, pelet", "habitat": "Laut, sungai", "fakta": "Bernapas menggunakan insang."},
    "chicken": {"nama": "Ayam 🐔", "jenis": "Kampung, Broiler", "makanan": "Biji-bijian", "habitat": "Kandang", "fakta": "Ayam bisa mengenali wajah."},
    "horse": {"nama": "Kuda 🐴", "jenis": "Arab, Pony", "makanan": "Rumput", "habitat": "Padang rumput", "fakta": "Bisa tidur sambil berdiri."},
    "butterfly": {"nama": "Kupu-kupu 🦋", "jenis": "Monarch, Morpho", "makanan": "Nektar bunga", "habitat": "Hutan, taman", "fakta": "Merasakan rasa lewat kaki."},
    "spider": {"nama": "Laba-laba 🕷️", "jenis": "Tarantula, Jumping Spider", "makanan": "Serangga", "habitat": "Taman, rumah", "fakta": "Jaringnya sangat kuat."}
}

# ==========================
# UI HEADER
# ==========================
st.markdown("<h1 class='main-title'>🐾 Animal Vision Pro — Premium Edition</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis gambar hewan dengan tampilan elegan dan interaktif</p>", unsafe_allow_html=True)

# ==========================
# UPLOAD FILE
# ==========================
uploaded = st.file_uploader("📤 Unggah gambar hewan", type=["jpg", "jpeg", "png"])
if uploaded:
    image_obj = Image.open(uploaded).convert("RGB")
    st.image(image_obj, caption="📷 Gambar diunggah", use_container_width=True)
    with st.spinner("🔎 Memprediksi jenis hewan..."):
        img_resized = image_obj.resize((224, 224))
        arr = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)
        preds = classifier.predict(arr)
        idx = np.argmax(preds)
        label = class_names[idx]
        confidence = float(np.max(preds))

    info = animal_info.get(label, {})
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader(f"🎯 Hasil Prediksi: {info.get('nama', label.title())}")
        st.write(f"**Tingkat Keyakinan:** {confidence*100:.2f}%")
        if info:
            st.write(f"🌿 **Jenis:** {info['jenis']}")
            st.write(f"🍽️ **Makanan:** {info['makanan']}")
            st.write(f"🏞️ **Habitat:** {info['habitat']}")
            st.info(f"💡 Fakta menarik: {info['fakta']}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("📁 Silakan unggah gambar hewan untuk mulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("<p class='footer'>🌸 Dibuat dengan cinta oleh <b>Rini Safariani</b> — AI Animal Classifier Premium</p>", unsafe_allow_html=True)
