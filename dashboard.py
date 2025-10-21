import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# =========================================================
# KONFIGURASI DASAR
# =========================================================
st.set_page_config(page_title="🐾 Animal Vision Pro", layout="wide", page_icon="🐾")

# CSS background
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
div.block-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.2);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_classifier():
    try:
        model = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

classifier = load_classifier()

# =========================================================
# UI
# =========================================================
st.sidebar.title("🔧 Status Model & Pengaturan")
if classifier is not None:
    st.sidebar.success("✅ Model berhasil dimuat")
    st.sidebar.write(f"Input shape model: {classifier.input_shape}")
else:
    st.sidebar.error("❌ Model tidak ditemukan")

st.title("🐾 Animal Vision Pro — Smart Animal Classifier")
st.caption("Klasifikasi gambar hewan lengkap dengan informasi biologis dan tampilan elegan 🌸")

uploaded = st.file_uploader("📤 Unggah gambar hewan", type=["jpg", "jpeg", "png"])

# =========================================================
# DATA HEWAN
# =========================================================
animal_info = {
    "cat": {"Jenis": "Mamalia", "Makanan": "Daging, ikan", "Habitat": "Rumah & lingkungan manusia"},
    "dog": {"Jenis": "Mamalia", "Makanan": "Daging & makanan anjing", "Habitat": "Rumah & perkotaan"},
    "chicken": {"Jenis": "Unggas", "Makanan": "Biji-bijian", "Habitat": "Peternakan"},
    "horse": {"Jenis": "Mamalia", "Makanan": "Rumput & biji", "Habitat": "Padang rumput"},
    "butterfly": {"Jenis": "Serangga", "Makanan": "Nektar bunga", "Habitat": "Taman & hutan"},
    "fish": {"Jenis": "Vertebrata air", "Makanan": "Plankton, serangga kecil", "Habitat": "Laut & sungai"},
    "spider": {"Jenis": "Arachnida", "Makanan": "Serangga kecil", "Habitat": "Taman & hutan"}
}

class_names = list(animal_info.keys())

# =========================================================
# PROSES KLASIFIKASI
# =========================================================
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

    if classifier is not None:
        with st.spinner("🔍 Sedang menganalisis gambar..."):
            # DETEKSI OTOMATIS UKURAN INPUT
            model_shape = classifier.input_shape
            if len(model_shape) == 4:
                target_size = (model_shape[1], model_shape[2])
            else:
                target_size = (128, 128)  # fallback aman

            img_resized = img.resize(target_size)
            arr = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)

            # CEGAH SHAPE MISMATCH SECARA OTOMATIS
            try:
                preds = classifier.predict(arr)
            except ValueError:
                # fallback: cari ukuran yang bisa diterima
                found = False
                for s in [64, 96, 128, 160, 192, 224]:
                    try:
                        resized = img.resize((s, s))
                        arr = np.expand_dims(image.img_to_array(resized) / 255.0, axis=0)
                        preds = classifier.predict(arr)
                        found = True
                        break
                    except Exception:
                        continue
                if not found:
                    st.error("⚠️ Tidak bisa mencocokkan ukuran input model.")
                    st.stop()

            idx = np.argmax(preds)
            label = class_names[idx]
            confidence = float(np.max(preds))

        st.success(f"🎯 **Prediksi:** {label.capitalize()} ({confidence*100:.2f}%)")

        info = animal_info[label]
        st.markdown("### 🌿 Informasi Hewan")
        st.write(f"**Jenis:** {info['Jenis']}")
        st.write(f"**Makanan:** {info['Makanan']}")
        st.write(f"**Habitat:** {info['Habitat']}")
        st.markdown("---")
        st.subheader("📸 Contoh Hewan Sejenis")
        st.image(f"https://source.unsplash.com/600x400/?{label}", caption=f"Gambar acak {label}")
    else:
        st.warning("⚠️ Model belum dimuat.")
else:
    st.info("📁 Silakan unggah gambar untuk mulai klasifikasi.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("💠 Dibuat oleh **Rini Safariani** • Animal Vision Pro 🐾")
