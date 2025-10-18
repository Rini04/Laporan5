import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
from sklearn.cluster import KMeans

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🌸 Flower Vision Pro", layout="wide")

# ==========================
# CUSTOM BACKGROUND & STYLE
# ==========================
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://i.ibb.co/Fmfxjwr/flower-bg.jpg');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #2c2c2c;
            font-family: 'Poppins', sans-serif;
        }
        .title {
            text-align: center;
            color: #3a0ca3;
            font-size: 42px;
            font-weight: 700;
            margin-top: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #560bad;
            margin-bottom: 25px;
        }
        .result-box {
            background-color: rgba(255,255,255,0.85);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        footer {
            text-align: center;
            color: #6a0572;
            font-size: 15px;
            margin-top: 40px;
            background: rgba(255, 255, 255, 0.6);
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Rini Safariani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🌸 Flower Vision Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>YOLOv8 + TensorFlow + Tools Analisis Gambar</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Menu Fitur")
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Filter Gambar", "Ubah Ukuran/Crop", "Analisis Warna"]
)
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# FUNGSI TAMBAHAN
# ==========================
def get_dominant_colors(img, n_colors=5):
    img_np = np.array(img)
    img_np = img_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_np)
    colors = np.array(kmeans.cluster_centers_, dtype=int)
    return colors

# ==========================
# PROSES GAMBAR
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    # ==========================
    # MODE 1: YOLO DETECTION
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="🧩 Hasil Deteksi", use_container_width=True)

    # ==========================
    # MODE 2: KLASIFIKASI
    # ==========================
    elif menu == "Klasifikasi Gambar":
        st.subheader("🧾 Hasil Klasifikasi")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        st.success(f"🎯 Prediksi: {class_index}")
        st.info(f"📊 Probabilitas: {confidence:.2%}")

    # ==========================
    # MODE 3: FILTER GAMBAR
    # ==========================
    elif menu == "Filter Gambar":
        st.subheader("🎨 Terapkan Filter Gambar")
        filter_option = st.selectbox("Pilih Filter:", ["Asli", "Grayscale", "Blur", "Sharpen", "Edge Detection"])
        if filter_option == "Grayscale":
            img_filtered = ImageOps.grayscale(img)
        elif filter_option == "Blur":
            img_filtered = img.filter(ImageFilter.BLUR)
        elif filter_option == "Sharpen":
            img_filtered = img.filter(ImageFilter.SHARPEN)
        elif filter_option == "Edge Detection":
            img_filtered = img.filter(ImageFilter.FIND_EDGES)
        else:
            img_filtered = img
        st.image(img_filtered, caption=f"Hasil Filter: {filter_option}", use_container_width=True)

    # ==========================
    # MODE 4: CROP & RESIZE
    # ==========================
    elif menu == "Ubah Ukuran/Crop":
        st.subheader("✂️ Crop dan Ubah Ukuran")
        width, height = img.size
        new_w = st.slider("Lebar Baru", 50, width, width)
        new_h = st.slider("Tinggi Baru", 50, height, height)
        img_resized = img.resize((new_w, new_h))
        st.image(img_resized, caption="📐 Hasil Resize", use_container_width=True)

    # ==========================
    # MODE 5: ANALISIS WARNA
    # ==========================
    elif menu == "Analisis Warna":
        st.subheader("🎨 Analisis Warna Dominan")
        colors = get_dominant_colors(img, n_colors=5)
        st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)
        st.write("🌈 Warna Dominan:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            color = tuple(colors[i])
            hex_color = '#%02x%02x%02x' % color
            col.markdown(f"<div style='background-color:{hex_color}; width:100%; height:80px; border-radius:8px;'></div>", unsafe_allow_html=True)
            col.write(f"{hex_color}")

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🌼 <b>Flower Vision Pro</b> — Aplikasi multi-fungsi karya <b>Rini Safariani</b>.<br>
    Menggabungkan YOLOv8, TensorFlow, dan Image Analysis.<br>
    <span style="font-size:13px;">© 2025 All rights reserved.</span>
</footer>
""", unsafe_allow_html=True)
