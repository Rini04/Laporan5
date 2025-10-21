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
# CUSTOM CSS — Pastel Floral Glow Theme
# ==========================
st.markdown("""
    <style>
        /* Background gradient pastel elegan */
        .stApp {
            background: linear-gradient(135deg, #fde2e4 0%, #fad2e1 30%, #e2ece9 70%, #bee1e6 100%);
            background-attachment: fixed;
            color: #2c2c2c;
            font-family: 'Poppins', sans-serif;
        }

        /* Efek bunga lembut di background */
        .stApp::before {
            content: "";
            position: fixed;
            top: -100px;
            left: -100px;
            width: 200%;
            height: 200%;
            background-image: radial-gradient(circle at 20% 30%, rgba(255,192,203,0.25) 10%, transparent 40%),
                              radial-gradient(circle at 80% 70%, rgba(173,216,230,0.25) 15%, transparent 40%),
                              radial-gradient(circle at 60% 40%, rgba(255,182,193,0.25) 10%, transparent 40%);
            z-index: -1;
        }

        /* Header */
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

        /* Card hasil */
        .result-box {
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(8px);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }

        /* Tombol sidebar */
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
st.markdown("<div class='subtitle'>Aplikasi Deteksi, Klasifikasi, dan Analisis Gambar dengan Sentuhan Estetik</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🌼 Menu Fitur")
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

    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("Sedang mendeteksi objek... 🌸"):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="🧩 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        st.subheader("🧾 Hasil Klasifikasi Gambar")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        st.success(f"🎯 Prediksi: {class_index}")
        st.info(f"📊 Probabilitas: {confidence:.2%}")

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

    elif menu == "Ubah Ukuran/Crop":
        st.subheader("✂️ Crop dan Ubah Ukuran")
        width, height = img.size
        new_w = st.slider("Lebar Baru", 50, width, width)
        new_h = st.slider("Tinggi Baru", 50, height, height)
        img_resized = img.resize((new_w, new_h))
        st.image(img_resized, caption="📐 Hasil Resize", use_container_width=True)

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
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk mulai.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🌷 <b>Flower Vision Pro</b> by <b>Rini Safariani</b><br>
    YOLOv8 • TensorFlow • Image Analysis • Esthetic UI<br>
    <span style="font-size:13px;">© 2025 All rights reserved.</span>
</footer>
""", unsafe_allow_html=True)
