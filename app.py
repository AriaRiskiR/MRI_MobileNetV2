import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Judul Aplikasi
st.title("Deteksi Tumor Otak MRI")
st.write("Unggah gambar MRI Otak untuk deteksi tumor.")

# Memuat model (pastikan 'mobilenetv2_mri.h5' ada di direktori yang sama)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mobilenetv2_mri.h5')
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        st.stop()

model = load_model()

if model is None:
    st.write("Model tidak dapat dimuat. Silakan periksa file model Anda.")
else:
    # Mengunggah gambar
    uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        st.write("")
        st.write("Menganalisis...")

        # Pra-pemrosesan gambar
        # Model MobileNetV2 biasanya mengharapkan input 224x224
        try:
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0  # Normalisasi ke [0, 1]
            img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
        except Exception as e:
            st.error(f"Error dalam pra-pemrosesan gambar: {e}")
            st.stop()

        # Melakukan prediksi
        try:
            predictions = model.predict(img_array)
            
            # --- BAGIAN PENTING YANG DIMODIFIKASI ---
            # Definisikan nama kelas sesuai dengan urutan output model Anda
            # PENTING: Pastikan urutan ini sesuai dengan bagaimana model Anda dilatih!
            class_names = ["No Tumor", "Glioma", "Meningioma", "Pituitary"] 
            # Sesuaikan urutan ini jika model Anda mengeluarkan probabilitas dalam urutan yang berbeda.
            # Contoh: Jika output model Anda adalah [prob_glioma, prob_meningioma, prob_notumor, prob_pituitary]
            # maka class_names harus disesuaikan menjadi ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]

            st.write(f"**Prediksi:** {class_names[predicted_class_index]}")
            st.write(f"**Keyakinan:** {confidence:.2f}")

            # Memberikan pesan berdasarkan prediksi
            if class_names[predicted_class_index] == "No Tumor":
                st.success("Gambar ini diprediksi **tidak ada tumor**.")
            else:
                st.warning(f"Gambar ini diprediksi memiliki **{class_names[predicted_class_index]}**. Silakan berkonsultasi dengan profesional medis.")

        except Exception as e:
            st.error(f"Error dalam melakukan prediksi: {e}")
            st.write("Pastikan model Anda menghasilkan output yang sesuai dengan penanganan prediksi (misalnya, jumlah kelas).")