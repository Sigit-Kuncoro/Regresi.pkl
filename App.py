import streamlit as st
import pandas as pd
import numpy as np
import joblib # Mengubah dari pickle ke joblib

# --- Bagian Memuat Model Anda yang Sebenarnya ---
# Di aplikasi nyata, Anda akan memuat model Anda yang sudah dilatih di sini.
try:
    with open('regresi.pkl', 'rb') as file:
        model = joblib.load(file)
    st.success("Model 'regresi.pkl' berhasil dimuat!")
except FileNotFoundError:
    st.error("File model 'regresi.pkl' tidak ditemukan. Pastikan ada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak dapat dimuat
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()


# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Prediksi Income",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Aplikasi Prediksi Income")
st.markdown("""
    Aplikasi ini memprediksi *Income* (pendapatan) berdasarkan *Age* (usia) dan *Experience* (pengalaman) Anda.
    
    *Catatan: Model yang digunakan di sini adalah model contoh. Untuk penggunaan nyata, Anda akan memuat model terlatih Anda sendiri.*
""")

st.header("Masukkan Data Baru")

# --- Input Pengguna menggunakan Widget Streamlit ---
col1, col2 = st.columns(2)

with col1:
    new_age = st.number_input(
        "Masukkan nilai Age (Usia):",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Usia dalam tahun."
    )

with col2:
    new_experience = st.number_input(
        "Masukkan nilai Experience (Pengalaman):",
        min_value=0,
        max_value=60,
        value=5,
        step=1,
        help="Pengalaman kerja dalam tahun."
    )

# --- Tombol untuk Prediksi ---
if st.button("Prediksi Income"):
    try:
        # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
        new_data_df = pd.DataFrame([[new_age, new_experience]], columns=['Age', 'Experience'])

        # Lakukan prediksi menggunakan model yang sudah dilatih
        predicted_income = model.predict(new_data_df)

        st.subheader("Hasil Prediksi")
        st.info(f"Untuk Age = **{new_age}** dan Experience = **{new_experience}**:")
        # predicted_income adalah array, ambil nilai tunggalnya
        st.success(f"Prediksi Income adalah: **${predicted_income[0]:,.2f}**")
        st.balloons() # Efek balon saat prediksi sukses

    except ValueError:
        st.error("Input tidak valid. Harap masukkan angka yang benar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e) # Menampilkan detail error untuk debugging
