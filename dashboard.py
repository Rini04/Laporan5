import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import glob
import os

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="Animal Vision AI by Rini Safariani 🐾", layout="wide")

# =======================
# STYLE
# =======================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #141E30, #243B55);
            background-image: url('https://images.unsplash.com/photo-1508675801603-7a89e9e9c5cf');
            background-size: cover;
            background-attachment: fixed;
            color: #fff;
            font-family: "Poppins", sans-serif;
        }
        .title {
            text-align:center;
            font-size:48px;
            font-weight:800;
            margin-top:15px;
            background: linear-gradient(90deg, #FFD700, #FFF8DC, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 3s infinite linear;
        }
        @keyframes glow {
            from {background-position: 0%;}
            to {background-position: 100%;}
        }
        .result-box {
            background: rgba(0,0,0,0.4);
            padding:20px;
            border-radius:12px;
            backdrop-filter: blur(6px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# LOAD MODEL
# =======================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_model():
    path = find_first("*.h5")
    if not path:
        return None
    return tf.keras.models.load_model(path)

model = load_model()

# =======================
# DATA HEWAN
# =======================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]
info = {
    "cat": {"nama": "🐱 Kucing", "habitat": "Rumah dan sekitar manusia.",
            "makanan": "Ikan, daging.", "fakta": "Kucing bisa tidur 16 jam/hari."},
    "dog": {"nama": "🐶 Anjing", "habitat": "Sekitar manusia.",
            "makanan": "Daging, tulang.", "fakta": "Sangat setia kepada pemiliknya."},
    "spider": {"nama": "🕷️ Laba-laba", "habitat": "Rumah, taman.",
               "makanan": "Serangga.", "fakta": "Jaringnya sangat kuat."},
    "chicken": {"nama": "🐔 Ayam", "habitat": "Peternakan.", 
                "makanan": "Biji-bijian.", "fakta": "Kenal lebih dari 100 wajah manusia!"},
    "horse": {"nama": "🐴 Kuda", "habitat": "Padang rumput.",
              "makanan": "Rumput.", "fakta": "Bisa tidur sambil berdiri."},
    "butterfly": {"nama": "🦋 Kupu-kupu", "habitat": "Kebun, hutan.",
                  "makanan": "Nektar.", "fakta": "Mencicipi rasa dengan kakinya."},
    "fish": {"nama": "🐟 Ikan", "habitat": "Laut dan air tawar.",
             "makanan": "Plankton.", "fakta": "Tidur dengan mata terbuka."}
}

# =======================
# HEADER
# =======================
st.markdown("<div class='title'>🐾 Animal Vision AI by Rini Safariani 🦋</div>", unsafe_allow_html=True)
st.write("### 🌿 Sistem cerdas untuk mengenali berbagai jenis hewan menggunakan model AI")

# =======================
# NAVIGASI
# =======================
menu = st.sidebar.radio("📂 Navigasi", ["Upload Gambar", "Tentang"])

# =======================
# UPLOAD
# =======================
if menu == "Upload Gambar":
    uploaded = st.file_uploader("📸 Unggah gambar hewan", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", width=500)

        if model:
            img_resized = img.resize((224, 224))
            arr = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)
            preds = model.predict(arr)
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = class_names[idx]

            data = info.get(label, {})
            st.markdown(f"""
            <div class='result-box'>
                <h3>{data.get("nama", label.title())}</h3>
                <b>🌍 Habitat:</b> {data.get("habitat","-")}<br>
                <b>🍽️ Makanan:</b> {data.get("makanan","-")}<br>
                <b>💡 Fakta:</b> {data.get("fakta","-")}<br>
                <b>🔮 Keyakinan Model:</b> {conf*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Tidak menemukan file model (.h5) di folder 'model/'.")

# =======================
# TENTANG
# =======================
else:
    st.markdown("""
    ### 🦋 Tentang Aplikasi
    **Animal Vision AI by Rini Safariani** adalah aplikasi berbasis AI yang bisa mengenali hewan dari gambar.
    
    **Fitur:**
    - Deteksi otomatis hewan (kucing, anjing, ikan, ayam, laba-laba, kupu-kupu, kuda)
    - Informasi habitat dan fakta unik
    - Desain elegan bertema hutan 🌿
    """)

    st.markdown("---")
    st.markdown("🌸 *Dibuat dengan cinta oleh Rini Safariani 💖*")

