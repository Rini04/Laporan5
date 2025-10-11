import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🧠 Image Classification & Object Detection", layout="wide")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Rini Safariani_Laporan 4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")  # Model Klasifikasi
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")
st.write("Aplikasi ini menggunakan **YOLOv8** untuk deteksi objek dan **TensorFlow** untuk klasifikasi gambar.")

menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES FILE YANG DI-UPLOAD
# ==========================
if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    # ==========================
    # MODE DETEKSI OBJEK (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek (YOLOv8)")
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        st.image(result_img, caption="🧩 Hasil Deteksi Objek", use_container_width=True)

    # ==========================
    # MODE KLASIFIKASI GAMBAR
    # ==========================
    elif menu == "Klasifikasi Gambar":
        st.subheader("🧾 Hasil Klasifikasi Gambar")
        with st.spinner("Sedang mengklasifikasi gambar..."):
            # Preprocessing
            img_resized = img.resize((224, 224))  # sesuaikan dengan input model
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"**Prediksi:** {class_index}")
        st.info(f"**Probabilitas:** {confidence:.2%}")

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk mulai.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("👩‍💻 Dibuat oleh **Rini Safariani** — Menggabungkan YOLOv8 & TensorFlow untuk Analisis Gambar.")
